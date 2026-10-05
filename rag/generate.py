"""
rag/generate.py
===============
Retrieval-augmented generation with a local LLM (Ollama).

Why generation timed out ("Ollama timed out after 120s") and what changed
-------------------------------------------------------------------------
  * ONE 120-second limit covered loading the model + reading ~1700 tokens of context +
    writing up to 800 tokens.  A 7B model on a laptop (4 GB GPU or CPU) needs minutes.
    Now the answer STREAMS: the only timeout is "no new token for READ_TIMEOUT seconds",
    and the UI can show text as it appears (st.write_stream).
  * Smaller default model: the first installed one of llama3.2:3b, qwen2.5:3b, phi3.5,
    gemma2:2b, ... mistral (see backend/config.py).  A 3B model fits a 4 GB GPU entirely.
  * Context cut to the 2 best passages, per-type output caps, keep_alive so the model
    stays loaded between requests.
  * Retrieval is limited to the chosen subject (before it searched everything).

Backwards compatible: generate_content(...) has the same signature and result keys.
"""
import json
import re
import threading
import time
from datetime import datetime, timedelta, timezone

import requests

from backend.config import OLLAMA_URL, LLM_PREFERENCE, CONCEPTS_PATH
from backend.db import get_collection
from rag import store
from rag.ingest import is_index_available            # re-exported: the Study page imports it from here too

FLASHCARD, SUMMARY, QUIZ, CODING_TASK = "flashcard", "summary", "quiz", "coding_task"
VALID_CONTENT_TYPES = [FLASHCARD, SUMMARY, QUIZ, CODING_TASK]

READ_TIMEOUT = 240            # seconds to wait for the NEXT token (first token includes model loading)
CONTEXT_PASSAGES = 2
CONTEXT_WORDS = 520
BEGINNER_BELOW, INTERMEDIATE_BELOW = 40.0, 65.0

TOKEN_CAPS = {FLASHCARD: 200, SUMMARY: 380, QUIZ: 650, CODING_TASK: 520,
              "flashcards": 450, "section_quiz": 650, "section_summary": 380, "answer": 400}


# =========================================================================== Ollama
def installed_models() -> list:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return [m["name"] for m in r.json().get("models", [])] if r.status_code == 200 else []
    except Exception:
        return []


def is_ollama_running() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


def pick_model(installed=None):
    """First model in LLM_PREFERENCE that is installed (prefix match: 'mistral' -> 'mistral:latest')."""
    installed = installed if installed is not None else installed_models()
    for pref in LLM_PREFERENCE:
        for name in installed:
            if name == pref or name.startswith(pref):
                return name
    return installed[0] if installed else None


def ollama_status() -> dict:
    """{'running', 'installed', 'model', 'loaded': [{'name','gpu_percent'}], 'hint'}"""
    st = {"running": False, "installed": [], "model": None, "loaded": [], "hint": ""}
    if not is_ollama_running():
        st["hint"] = "Ollama is not running. Start it with:  ollama serve"
        return st
    st["running"], st["installed"] = True, installed_models()
    st["model"] = pick_model(st["installed"])
    try:
        ps = requests.get(f"{OLLAMA_URL}/api/ps", timeout=2).json().get("models", [])
        st["loaded"] = [{"name": m["name"],
                         "gpu_percent": round(100 * m.get("size_vram", 0) / max(m.get("size", 1), 1))} for m in ps]
    except Exception:
        pass
    if not st["installed"]:
        st["hint"] = "No model installed. Run:  ollama pull llama3.2:3b"
    elif st["model"] and st["model"].startswith("mistral"):
        st["hint"] = "Mistral 7B is slow on most laptops. A smaller model answers much faster:  ollama pull llama3.2:3b"
    return st


def warm_up(background: bool = True):
    """Load the model into memory now so the first real request does not pay for it."""
    def _go():
        model = pick_model()
        if model:
            try:
                requests.post(f"{OLLAMA_URL}/api/generate",
                              json={"model": model, "keep_alive": "30m"}, timeout=300)
            except Exception:
                pass
    (threading.Thread(target=_go, daemon=True).start() if background else _go())


def stream_ollama(prompt: str, kind: str = "answer"):
    """Yield the reply token by token.  Raises RuntimeError with a message fit for the UI."""
    model = pick_model()
    if model is None:
        raise RuntimeError(
            "Ollama is not running or has no model.\n  1. ollama serve\n  2. ollama pull llama3.2:3b")
    payload = {
        "model": model, "prompt": prompt, "stream": True, "keep_alive": "30m",
        "options": {"temperature": 0.4, "top_p": 0.9, "repeat_penalty": 1.1, "num_ctx": 3072,
                    "num_predict": TOKEN_CAPS.get(kind, 400)},
    }
    try:
        with requests.post(f"{OLLAMA_URL}/api/generate", json=payload, stream=True,
                           timeout=(5, READ_TIMEOUT)) as r:
            if r.status_code == 404:
                raise RuntimeError(f"Model '{model}' is not installed. Run:  ollama pull {model}")
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("error"):
                    raise RuntimeError(f"Ollama error: {obj['error']}")
                if obj.get("response"):
                    yield obj["response"]
                if obj.get("done"):
                    return
    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError) as exc:
        # requests reports a stall AFTER the first token as ConnectionError("Read timed out"),
        # so look at the message to tell "stalled" from "Ollama is not running".
        if isinstance(exc, requests.exceptions.ReadTimeout) or "timed out" in str(exc).lower():
            raise RuntimeError(
                f"{model} sent nothing for {READ_TIMEOUT}s. The computer may be out of memory - close other "
                "apps, or try a smaller model:  ollama pull llama3.2:3b")
        raise RuntimeError("Lost the connection to Ollama. Start it with:  ollama serve")


def call_ollama(prompt: str, temperature: float = 0.4, kind: str = "answer") -> str:
    """Whole reply as one string (old signature; `temperature` is kept for old callers)."""
    return "".join(stream_ollama(prompt, kind)).strip()


# ======================================================================== retrieval
def get_difficulty_level(recall_score: float) -> str:
    if recall_score < BEGINNER_BELOW:
        return "beginner"
    return "intermediate" if recall_score < INTERMEDIATE_BELOW else "advanced"


def _concepts() -> dict:
    try:
        with open(CONCEPTS_PATH, encoding="utf-8") as f:
            return {c["concept_id"]: c for c in json.load(f)}
    except Exception:
        return {}


def retrieve_chunks(concept_id: str, query_text: str = None, top_k: int = 4) -> list:
    """Best passages for a subject: from that subject's notes first, then from its prerequisites
    (e.g. process_mgmt falls back to os).  Never returns passages from unrelated subjects."""
    if not store.is_ready():
        return []
    concepts = _concepts()
    query = query_text or concepts.get(concept_id, {}).get("name") or concept_id.replace("_", " ")
    hits = store.search(query, k=top_k, concept_ids=[concept_id])
    if len(hits) < 2:
        prereqs = concepts.get(concept_id, {}).get("prerequisites", [])
        if prereqs:
            hits += store.search(query, k=top_k - len(hits), concept_ids=prereqs)
    return hits


def build_context_string(chunks: list, max_chunks: int = CONTEXT_PASSAGES, max_words: int = CONTEXT_WORDS) -> str:
    parts, used = [], 0
    for c in chunks[:max_chunks]:
        words = c["text"].split()
        room = max_words - used
        if room <= 40:
            break
        parts.append(" ".join(words[:room]))
        used += min(len(words), room)
    return "\n\n".join(parts) if parts else "No context available."


# ========================================================================== prompts
_LEVEL = {
    "beginner": "Keep it basic: definitions and simple facts.",
    "intermediate": "Test understanding: how and why, and comparisons.",
    "advanced": "Make it challenging: edge cases, exceptions and subtle differences.",
}

_FORMATS = {
    FLASHCARD: "Write ONE flashcard.\nReply in exactly this format and nothing else:\nQUESTION: <question>\nANSWER: <answer in 1-3 sentences>",
    SUMMARY: "Write a revision summary of 4-5 bullet points.\nReply in exactly this format:\nSUMMARY: {name} - Key Points\n\n- **<Key term>**: <one-sentence explanation>",
    QUIZ: ("Write exactly 3 multiple-choice questions, 4 options each.\nReply in exactly this format and nothing else:\n"
           "QUIZ: {name}\n\nQ1: <question>\nA) <option>\nB) <option>\nC) <option>\nD) <option>\n"
           "CORRECT: <A, B, C or D>\nEXPLANATION: <one sentence>\n\n(then Q2 and Q3 the same way)"),
    CODING_TASK: ("Write ONE short Python exercise that needs only the standard library and prints a deterministic result.\n"
                  "Reply in exactly this format:\nCODING TASK: {name}\n\nPROBLEM:\n<description>\n\nINPUT:\n<example input>\n\n"
                  "EXPECTED OUTPUT:\n<exact output>\n\nHINT:\n<one hint, no solution>"),
}


def build_prompt(content_type: str, concept_name: str, context: str, difficulty: str) -> str:
    if content_type not in _FORMATS:
        raise ValueError(f"Unknown content type: {content_type}. Use one of {VALID_CONTENT_TYPES}")
    fmt = _FORMATS[content_type].format(name=concept_name)
    return (f"You are a computer-science tutor. Use ONLY the notes below; do not add facts that are not in them.\n"
            f"Subject: {concept_name}. Difficulty: {difficulty}. {_LEVEL[difficulty]}\n\n"
            f"NOTES:\n{context}\n\n{fmt}")


def get_prompt_for_type(content_type, concept_name, context, difficulty):      # old name
    return build_prompt(content_type, concept_name, context, difficulty)


def section_prompt(kind: str, title: str, text: str, max_words: int = 650) -> str:
    """kind: 'flashcards' | 'section_quiz' | 'section_summary'.  `text` is one section of the student's notes."""
    text = " ".join(text.split()[:max_words])
    head = f"You are a study assistant. Use ONLY the text below (the student's own notes).\n\nTEXT - {title}:\n{text}\n\n"
    if kind == "flashcards":
        return head + ("Write 5 flashcards on the most important ideas.\nReply in exactly this format and nothing else:\n"
                       "Q: <question>\nA: <short answer>\n(repeat for all 5)")
    if kind == "section_quiz":
        return head + _FORMATS[QUIZ].format(name=title)
    return head + ("Summarise this section in 5 bullet points a student can revise from in one minute.\n"
                   "Reply with the bullets only, each like: - **<Key term>**: <one sentence>")


def answer_prompt(question: str, passages: list) -> str:
    numbered = "\n\n".join(f"[{i}] ({p.get('section_title', p.get('concept_id', 'notes'))})\n{p['text']}"
                           for i, p in enumerate(passages, 1))
    return ("Answer the question using ONLY the numbered passages from the student's notes. "
            "Be concise. Cite passages like [1]. If the passages do not contain the answer, say so.\n\n"
            f"{numbered}\n\nQUESTION: {question}\nANSWER:")


# ========================================================================== parsers
_Q = re.compile(r"^\W*(?:\d+[.)]\s*)?(?:\*\*)?q(?:uestion)?\s*\d*\s*(?:\*\*)?\s*[:.\-)]\s*(.*)$", re.I)
_A = re.compile(r"^\W*(?:\*\*)?a(?:nswer)?\s*\d*\s*(?:\*\*)?\s*[:.\-)]\s*(.*)$", re.I)


def _plain(s: str) -> str:
    return re.sub(r"\*\*|__", "", s).strip()


def parse_flashcards(text: str) -> list:
    """[{'question', 'answer'}] from 'Q: ... / A: ...' or 'QUESTION: ... / ANSWER: ...' output."""
    cards, cur, field = [], None, None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m_q, m_a = _Q.match(line), _A.match(line)
        if m_q and (cur is None or cur["answer"]):
            cur = {"question": _plain(m_q.group(1)), "answer": ""}
            cards.append(cur); field = "question"
        elif m_a and cur is not None and not cur["answer"]:
            cur["answer"] = _plain(m_a.group(1)); field = "answer"
        elif cur is not None and field:
            cur[field] = (cur[field] + " " + _plain(line)).strip()
    return [c for c in cards if c["question"] and c["answer"]]


_QUESTION = re.compile(r"^\W*(?:\*\*)?(?:q(?:uestion)?\s*)?(\d+)\s*(?:\*\*)?\s*[:.)\-]\s*(.+)$", re.I)
_OPTION = re.compile(r"^\W*\(?([A-Da-d])[).:\-]\s*(.+)$")
_CORRECT = re.compile(r"^\W*(?:\*\*)?(?:correct(?:\s*answer)?|answer)\s*(?:\*\*)?\s*[:\-]\s*(?:\*\*)?\s*\(?([A-Da-d])\b", re.I)
_EXPL = re.compile(r"^\W*(?:\*\*)?explanation\s*(?:\*\*)?\s*[:\-]\s*(?:\*\*)?\s*(.*)$", re.I)


def parse_quiz(text: str) -> list:
    """[{'question', 'options': [('A','..'),...], 'correct': 'B', 'explanation'}] - incomplete questions are dropped."""
    qs, cur, in_expl = [], None, False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        mq, mo, mc, me = _QUESTION.match(line), _OPTION.match(line), _CORRECT.match(line), _EXPL.match(line)
        if mc and cur:
            cur["correct"] = mc.group(1).upper(); in_expl = False
        elif me and cur:
            cur["explanation"] = _plain(me.group(1)); in_expl = True
        elif mo and cur and not in_expl:
            cur["options"].append((mo.group(1).upper(), _plain(mo.group(2))))
        elif mq and not (cur and in_expl and not cur["options"]):
            cur = {"question": _plain(mq.group(2)), "options": [], "correct": "", "explanation": ""}
            qs.append(cur); in_expl = False
        elif cur and in_expl:
            cur["explanation"] = (cur["explanation"] + " " + _plain(line)).strip()
        elif cur and not cur["options"]:
            cur["question"] = (cur["question"] + " " + _plain(line)).strip()
    good = []
    for q in qs:
        letters = [l for l, _ in q["options"]]
        if len(letters) >= 3 and q["correct"] in letters:
            good.append(q)
    return good


_CODE_HEAD = re.compile(r"^\W*(?:\*\*)?(problem|input|expected\s*output|output|hint)\s*(?:\*\*)?\s*:\s*(?:\*\*)?\s*(.*)$", re.I)


def parse_coding_task(text: str) -> dict:
    """{'problem','input','expected','hint'} from the CODING_TASK reply (missing parts are '')."""
    names = {"problem": "problem", "input": "input", "expected output": "expected", "output": "expected", "hint": "hint"}
    parts, cur = {"problem": [], "input": [], "expected": [], "hint": []}, None
    for raw in text.splitlines():
        m = _CODE_HEAD.match(raw.strip())
        if m:
            cur = names[re.sub(r"\s+", " ", m.group(1).lower())]
            if m.group(2):
                parts[cur].append(m.group(2))
        elif cur and not raw.strip().upper().startswith("CODING TASK"):
            parts[cur].append(raw.rstrip())
    clean = {k: "\n".join(v).strip().strip("`").strip() for k, v in parts.items()}
    if not any(clean.values()):
        clean["problem"] = text.strip()
    return clean


# ===================================================================== MongoDB cache
def save_generated_content(user_id, concept_id, content_type, content, difficulty_level, model=None) -> str:
    res = get_collection("generated_content").insert_one({
        "user_id": user_id, "concept_id": concept_id, "content_type": content_type, "content": content,
        "difficulty_level": difficulty_level, "generated_at": datetime.now(timezone.utc),
        "rating": 0, "was_reviewed": False, "model_used": model or pick_model() or "unknown"})
    return str(res.inserted_id)


def get_cached_content(user_id, concept_id, content_type, max_age_hours: int = 24):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    return get_collection("generated_content").find_one(
        {"user_id": user_id, "concept_id": concept_id, "content_type": content_type,
         "generated_at": {"$gte": cutoff}}, {"_id": 0}, sort=[("generated_at", -1)])


def _update_latest(user_id, concept_id, content_type, fields: dict) -> bool:
    # find_one_and_update accepts `sort`; update_one(..., sort=...) does not exist in PyMongo 4.6
    # (the first version of rate_content used it, so thumbs up/down were never saved).
    try:
        get_collection("generated_content").find_one_and_update(
            {"user_id": user_id, "concept_id": concept_id, "content_type": content_type},
            {"$set": fields}, sort=[("generated_at", -1)])
        return True
    except Exception as exc:
        print(f"  could not update generated_content: {exc}")
        return False


def rate_content(user_id, concept_id, content_type, rating: int) -> bool:
    return rating in (-1, 0, 1) and _update_latest(user_id, concept_id, content_type, {"rating": rating})


def mark_content_as_reviewed(user_id, concept_id, content_type) -> bool:
    return _update_latest(user_id, concept_id, content_type, {"was_reviewed": True})


# ================================================================ main entry points
def _passages_for_ui(chunks: list) -> list:
    return [{"title": c.get("section_title") or c.get("concept_id", "notes"),
             "pages": (f"pp. {c['page_start']}-{c['page_end']}" if c.get("page_start") else ""),
             "text": c["text"]} for c in chunks]


def prepare_generation(user_id, concept_id, concept_name, content_type, recall_score, force_regenerate=False) -> dict:
    """Everything except the LLM call - lets a page stream the answer itself:

        prep = prepare_generation(...)
        if prep["content"]:  show it                       # cached
        elif prep["error"]:  show prep["error"] + prep["passages"]
        else:
            text = st.write_stream(stream_ollama(prep["prompt"], content_type))
            finish_generation(user_id, concept_id, content_type, text, prep["difficulty_level"])
    """
    out = {"content": None, "content_type": content_type, "difficulty_level": get_difficulty_level(recall_score),
           "concept_id": concept_id, "concept_name": concept_name, "recall_score": recall_score,
           "from_cache": False, "error": None, "prompt": None, "passages": [], "model": None}
    if content_type not in VALID_CONTENT_TYPES:
        out["error"] = f"Invalid content type '{content_type}'"
        return out
    if not force_regenerate:
        cached = get_cached_content(user_id, concept_id, content_type)
        if cached and cached.get("difficulty_level") == out["difficulty_level"]:
            out.update(content=cached["content"], from_cache=True, model=cached.get("model_used"))
            return out
    chunks = retrieve_chunks(concept_id, concept_name)
    if not chunks:
        out["error"] = (f"There are no notes for '{concept_name}' yet. Upload a PDF in My Notes "
                        f"(or add data/notes/{concept_id}.txt and run: python rag/ingest.py).")
        return out
    out["passages"] = _passages_for_ui(chunks)
    out["prompt"] = build_prompt(content_type, concept_name, build_context_string(chunks), out["difficulty_level"])
    try:                                                    # optional feedback loop from prompt_optimizer.py
        from backend.prompt_optimizer import get_feedback_context, inject_feedback_into_prompt
        out["prompt"] = inject_feedback_into_prompt(out["prompt"], get_feedback_context(user_id, concept_id, content_type))
    except Exception:
        pass
    if not is_ollama_running():
        out["error"] = "Ollama is not running. Start it with:  ollama serve"
    out["model"] = pick_model()
    return out


def finish_generation(user_id, concept_id, content_type, text, difficulty_level):
    if text and text.strip():
        save_generated_content(user_id, concept_id, content_type, text.strip(), difficulty_level)


def generate_content(user_id, concept_id, concept_name, content_type, recall_score, force_regenerate=False) -> dict:
    """Same signature and result keys as before, plus 'model' and 'fallback_passages' (the notes the
    answer would have been built from - show them when generation fails so the student can still revise)."""
    prep = prepare_generation(user_id, concept_id, concept_name, content_type, recall_score, force_regenerate)
    result = {k: prep[k] for k in ("content", "content_type", "difficulty_level", "concept_id", "concept_name",
                                   "recall_score", "from_cache", "error", "model")}
    result["fallback_passages"] = prep["passages"]
    if prep["content"] or prep["error"]:
        return result
    try:
        text = call_ollama(prep["prompt"], kind=content_type)
        if not text:
            result["error"] = "The model returned an empty answer. Try again."
            return result
        finish_generation(user_id, concept_id, content_type, text, prep["difficulty_level"])
        result["content"] = text
    except RuntimeError as exc:
        result["error"] = str(exc)
    except Exception as exc:
        result["error"] = f"Unexpected error: {exc}"
    return result
