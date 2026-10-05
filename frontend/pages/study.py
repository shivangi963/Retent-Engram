"""
Study  -  AI-written revision material from YOUR notes (flashcard, summary, quiz, coding task).

Everything you do here is saved automatically: finish a quiz, grade your flashcards, solve a
coding task or rate a summary and your recall estimate updates - no separate logging step.

Put this file where your current "Study" page lives (overwrite it), or point the "Study" entry
of your st.navigation(...) at it.
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "backend").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

from backend.config import CONCEPTS_PATH
from backend.ml.pipeline import compute_scores_for_user
from backend.study_log import make_logger
from frontend.components import practice
from rag import generate as gen
from rag import store

try:
    st.set_page_config(page_title="Study", page_icon=":material/school:", layout="wide")
except Exception:
    pass

USER_KEYS = ("user_id", "username", "user")
TYPES = {"Flashcard": gen.FLASHCARD, "Summary": gen.SUMMARY, "Quiz (3 questions)": gen.QUIZ, "Coding task": gen.CODING_TASK}
LEVEL_TEXT = {"beginner": "Beginner-level content (your recall is low)",
              "intermediate": "Intermediate content",
              "advanced": "Advanced content (you remember this well)"}


def current_user():
    for k in USER_KEYS:
        if st.session_state.get(k):
            return str(st.session_state[k])
    return None


def load_concepts() -> list:
    try:
        return json.loads(Path(CONCEPTS_PATH).read_text(encoding="utf-8"))
    except Exception:
        return []


@st.cache_data(ttl=10, show_spinner=False)
def llm_status() -> dict:
    return gen.ollama_status()


def request(concept_id, concept_name, content_type, recall, force):
    st.session_state["study_pending"] = dict(cid=concept_id, cname=concept_name, ctype=content_type,
                                             recall=recall, force=force)


# ------------------------------------------------------------------------------- page
st.title("Study")
st.caption("AI-generated review material from your own notes. Everything you do here is saved automatically.")

user_id = current_user()
if not user_id:
    st.warning("Please sign in first.")
    st.stop()
concepts = load_concepts()
if not concepts:
    st.error("data/concepts.json is missing or empty.")
    st.stop()

try:
    scores = {r["concept_id"]: r for r in compute_scores_for_user(user_id)}
except Exception:
    scores = {}
concepts.sort(key=lambda c: scores.get(c["concept_id"], {}).get("recall_score", 50.0))      # weakest first
by_id = {c["concept_id"]: c for c in concepts}
llm = llm_status()
llm_ok = bool(llm["running"] and llm["model"])

left, right = st.columns([1, 1.6])

with left:
    with st.container(border=True):
        st.subheader("Generate content")
        cid = st.selectbox("Concept", list(by_id), key="study_concept",
                           format_func=lambda c: by_id[c]["name"] + ("" if c in scores else "  (no sessions yet)"))
        info = scores.get(cid)
        recall = info["recall_score"] if info else 50.0
        if info:
            st.caption(f"Recall {recall:.0f}% - {info['priority']} priority")
        else:
            st.caption("No study logged for this concept yet.")
        level = gen.get_difficulty_level(recall)
        st.caption(LEVEL_TEXT[level] + " will be generated.")
        ctype_label = st.radio("Content type", list(TYPES), key="study_type")
        force = st.checkbox("Generate fresh content (ignore saved)", key="study_force",
                            help="By default content made in the last 24 hours is reused.")
        st.button("Generate", type="primary", key="study_generate", disabled=not llm_ok,
                  on_click=request, args=(cid, by_id[cid]["name"], TYPES[ctype_label], recall, force))
        if not llm_ok:
            st.caption(llm["hint"] or "Start Ollama to generate content.")
        elif llm["hint"]:
            st.caption(llm["hint"])
        if not store.is_ready():
            st.caption("No notes yet - add a PDF in My Notes so the AI has something to work from.")

with right:
    pending = st.session_state.pop("study_pending", None)
    if pending:
        practice.reset("study:w")
        prep = gen.prepare_generation(user_id, pending["cid"], pending["cname"], pending["ctype"],
                                      pending["recall"], pending["force"])
        result = {"cid": pending["cid"], "cname": pending["cname"], "ctype": pending["ctype"],
                  "level": prep["difficulty_level"], "text": prep["content"], "cached": prep["from_cache"],
                  "error": prep["error"], "passages": prep["passages"], "model": prep["model"]}
        if not prep["content"] and not prep["error"]:
            with st.status("Writing from your notes...", expanded=True) as status:
                try:
                    result["text"] = st.write_stream(gen.stream_ollama(prep["prompt"], pending["ctype"]))
                    gen.finish_generation(user_id, pending["cid"], pending["ctype"], result["text"], prep["difficulty_level"])
                    status.update(label="Ready", state="complete", expanded=False)
                except RuntimeError as exc:
                    result["error"] = str(exc)
                    status.update(label="Could not generate", state="error")
        st.session_state["study_result"] = result
        st.rerun()

    res = st.session_state.get("study_result")
    if not res:
        st.info("Choose a concept and a content type, then press Generate. Content you've already generated "
                "today is reused instantly.")
    elif res["error"] and not res["text"]:
        st.error(res["error"])
        if res["passages"]:
            st.markdown("**You can still revise from your notes meanwhile:**")
            for pg in res["passages"]:
                with st.expander(f"{pg['title']} {pg['pages']}".strip()):
                    st.write(pg["text"])
    else:
        ctype, text = res["ctype"], res["text"]
        st.caption(f"{res['cname']} - {dict((v, k) for k, v in TYPES.items())[ctype]} - {res['level']} level - "
                   + ("from saved content" if res["cached"] else "just generated")
                   + (f" - {res['model']}" if res["model"] else ""))
        logger = make_logger(user_id, res["cid"])
        key = f"study:w:{res['cid']}:{ctype}"
        parsed_ok = True
        if ctype == gen.FLASHCARD:
            cards = gen.parse_flashcards(text)
            parsed_ok = bool(cards)
            if cards:
                practice.flashcards(cards, key, logger, source="study_flashcards", minutes=3)
        elif ctype == gen.QUIZ:
            questions = gen.parse_quiz(text)
            parsed_ok = bool(questions)
            if questions:
                practice.quiz(questions, key, logger, source="study_quiz", minutes=5)
        elif ctype == gen.CODING_TASK:
            practice.coding(gen.parse_coding_task(text), key, logger, source="study_coding", minutes=10)
        else:
            with st.container(border=True):
                st.markdown("\n".join(ln for ln in text.splitlines() if not ln.strip().upper().startswith("SUMMARY:")))
            practice.self_rating(key, logger, prompt="Finished reading? How did it go?",
                                 event_type="reading", source="study_summary", minutes=3)
        if not parsed_ok:
            st.warning("The answer could not be turned into an interactive exercise. Here it is as written:")
            st.write(text)
            practice.self_rating(key, logger, prompt="How did it go?", event_type="quiz",
                                 source="study_" + ("quiz" if ctype == gen.QUIZ else "flashcards"), minutes=3)

        st.divider()
        fb1, fb2, fb3 = st.columns(3)
        with fb1:
            if st.button("Helpful", key="study_up"):
                gen.rate_content(user_id, res["cid"], ctype, 1)
                st.toast("Thanks - noted.")
        with fb2:
            if st.button("Not helpful", key="study_down"):
                gen.rate_content(user_id, res["cid"], ctype, -1)
                st.toast("Noted - try Regenerate for a different version.")
        with fb3:
            st.button("Regenerate", key="study_regen", on_click=request,
                      args=(res["cid"], res["cname"], ctype, scores.get(res["cid"], {}).get("recall_score", 50.0), True))
