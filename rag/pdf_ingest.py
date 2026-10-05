"""
rag/pdf_ingest.py
=================
Your uploaded PDFs become a LIBRARY you can read and revise from - not just vectors.

  upload -> extract pages -> clean -> split into SECTIONS (by headings) -> chunk -> index
            \\__ saved as data/library/<doc_id>.json so the app can show them again

New compared with the first version
  * sections + page numbers (so "My Notes" can show a table of contents and a reader)
  * the same PDF can't be added twice (SHA-256 of the file)
  * documents can be deleted (their chunks leave the index too)
  * the index is created on first upload - no need to run ingest.py first
Old entry points still work: ingest_pdf(...), detect_concept_from_text(...).
"""
import hashlib
import json
import os
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

from backend.config import LIBRARY_DIR, CONCEPTS_PATH
from rag import store

REGISTRY_PATH = LIBRARY_DIR / "registry.json"
MIN_SECTION_WORDS = 45
MAX_SECTION_WORDS = 700
CHUNK_WORDS, CHUNK_OVERLAP = 230, 40

# ------------------------------------------------------------ concept detection
CONCEPT_KEYWORDS = {
    "os": ["operating system", "process", "deadlock", "scheduling", "memory management", "paging",
           "semaphore", "mutex", "virtual memory", "thread", "kernel", "context switch"],
    "dbms": ["database", "normalization", "transaction", "acid", "b-tree", "indexing", "relational",
             "er diagram", "functional dependency", "schema", "tuple", "dbms"],
    "cn": ["network", "tcp", "ip address", "routing", "protocol", "osi", "ethernet", "dns", "http",
           "subnet", "udp", "packet", "bandwidth", "router", "switch", "lan", "wan", "data communication"],
    "dsa": ["data structure", "algorithm", "linked list", "binary tree", "graph", "sorting",
            "dynamic programming", "stack", "queue", "hash table", "complexity"],
    "python_oop": ["class", "object", "inheritance", "polymorphism", "encapsulation", "abstraction",
                   "__init__", "decorator", "python"],
    "sql": ["select", "insert", "update", "delete", "group by", "having", "aggregate", "subquery", "sql"],
    "process_mgmt": ["process creation", "fork", "ipc", "pipe", "process state", "pcb", "context switch"],
    "memory_mgmt": ["page replacement", "lru", "thrashing", "frame", "segmentation", "fragmentation", "tlb"],
}


def detect_concept_from_text(text: str, min_score: int = 6):
    """Concept whose keywords occur most often (each keyword counts at most 8 times so one
    repeated word can't win).  None when nothing is convincing."""
    low = text.lower()
    scores = {cid: sum(min(low.count(k), 8) for k in kws) for cid, kws in CONCEPT_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= min_score else None


# ------------------------------------------------------------------- extraction
def extract_pages(pdf_path) -> list:
    """[(page_no, raw_text)] - needs `pip install pdfplumber`."""
    import pdfplumber
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            pages.append((i, page.extract_text() or ""))
    return pages


_PAGE_NO = re.compile(r"^(page\s*)?\d{1,3}(\s*(of|/)\s*\d{1,3})?$", re.I)
_DOT_LEADER = re.compile(r"\.{4,}|(\.\s){4,}")
_RULE_LINE = re.compile(r"^[\s\-_=*~\u2022\u2013\u2014.]{3,}$")           # ------  ======  * * *


def clean_pages(pages: list) -> list:
    """[(page_no, [lines])]: removes page numbers, dot-leader (table of contents) pages and
    running headers/footers that repeat on many pages."""
    cleaned = []
    for no, text in pages:
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln and not _PAGE_NO.match(ln) and not _RULE_LINE.match(ln)]
        if sum(1 for ln in lines if _DOT_LEADER.search(ln)) >= 5:        # table of contents page
            continue
        cleaned.append((no, [ln for ln in lines if not _DOT_LEADER.search(ln)]))

    if len(cleaned) >= 4:                                               # running headers / footers
        freq = {}
        for _, lines in cleaned:
            for ln in set(re.sub(r"\d+", "#", x) for x in lines if len(x) < 90):
                freq[ln] = freq.get(ln, 0) + 1
        limit = max(3, int(0.4 * len(cleaned)))
        cleaned = [(no, [ln for ln in lines
                         if len(ln) >= 90 or freq.get(re.sub(r"\d+", "#", ln), 0) < limit])
                   for no, lines in cleaned]
    return [(no, lines) for no, lines in cleaned if lines]


# --------------------------------------------------------------------- headings
_MODULE = re.compile(r"^(module|chapter|unit)\s*[-:\u2013\u2014]?\s*(\d+|[ivx]+)\b\s*[-:.\u2013\u2014]?\s*(.*)$", re.I)
_NUM_MULTI = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,3})\.?\s+(\S.{2,100})$")      # 1.1 Title  /  2.3.4 Title
_NUM_SINGLE = re.compile(r"^(\d{1,2})[.)]\s+([A-Z].{2,80})$")                      # 1. Title   /  1) Title
_NOT_HEADING = ("fig", "figure", "table", "note", "example", "output", "input", "source", "page")


def _plausible_title(t: str) -> bool:
    words = t.split()
    if not 1 <= len(words) <= 14 or t[-1] in ".,;:?!":
        return False
    letters = sum(c.isalpha() for c in t)
    return letters >= 3 and letters / len(t) >= 0.55 and t[0].isalpha() and t[0].isupper()


def heading_level(line: str):
    """1 = chapter-level, 2 = '1.1', 3 = '1.1.1'.  None if the line is not a heading."""
    line = line.strip()
    if not line or len(line) > 110 or line.lower().startswith(_NOT_HEADING):
        return None
    if _MODULE.match(line):
        return 1
    m = _NUM_MULTI.match(line)
    if m and _plausible_title(m.group(2)):
        return min(m.group(1).count(".") + 1, 3)
    m = _NUM_SINGLE.match(line)
    if m and _plausible_title(m.group(2)):
        words = m.group(2).split()
        if len(words) <= 8 and sum(w[0].isupper() for w in words) / len(words) >= 0.6:
            return 1
    letters = [c for c in line if c.isalpha()]
    if (len(letters) >= 5 and line.isupper() and len(line.split()) <= 9 and line[0].isalpha()
            and line[-1] not in ".,;:?!)]" and not re.search(r"[|+=\[\](){}]", line)):
        return 1                                                    # ALL-CAPS title line
    return None


def _clean_title(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip(" -:\u2013\u2014")[:90]


# --------------------------------------------------------------------- sections
def _make_section(title, level, items):
    text = "\n".join(ln for _, ln in items)
    pages = [p for p, _ in items]
    return {"title": title, "level": level, "text": text, "words": len(text.split()),
            "page_start": min(pages), "page_end": max(pages)}


def _split_long(sec):
    if sec["words"] <= MAX_SECTION_WORDS * 1.3:
        return [sec]
    lines = sec["text"].split("\n")
    parts, cur, count = [], [], 0
    for ln in lines:
        cur.append(ln)
        count += len(ln.split())
        if count >= MAX_SECTION_WORDS and ln.rstrip().endswith((".", "?", "!", ":")):
            parts.append(cur); cur, count = [], 0
    if cur:
        if parts and count < MIN_SECTION_WORDS:
            parts[-1].extend(cur)
        else:
            parts.append(cur)
    if len(parts) == 1:
        return [sec]
    out, n = [], len(parts)
    for i, ln_list in enumerate(parts, 1):
        text = "\n".join(ln_list)
        out.append({**sec, "title": f"{sec['title']} ({i}/{n})", "text": text,
                    "words": len(text.split())})
    return out


def split_sections(cleaned: list) -> list:
    """cleaned: [(page_no, [lines])]  ->  [{id,title,level,text,words,page_start,page_end}]"""
    sections, cur_title, cur_level, cur_items = [], None, 1, []

    def close():
        nonlocal cur_items
        if cur_items:
            sections.append(_make_section(cur_title or "Introduction", cur_level, cur_items))
        cur_items = []

    n_headings = 0
    for page, lines in cleaned:
        for ln in lines:
            lvl = heading_level(ln)
            if lvl is None:
                cur_items.append((page, ln))
                continue
            n_headings += 1
            words_so_far = sum(len(x.split()) for _, x in cur_items)
            big_enough = words_so_far >= (MIN_SECTION_WORDS * 1.3 if cur_title is None else MIN_SECTION_WORDS)
            if big_enough:
                close()
                cur_title, cur_level, cur_items = _clean_title(ln), lvl, [(page, ln)]
            else:
                # the section so far is only a lead-in (cover line, chapter title...): keep its
                # text, but title the section by the more specific heading that follows
                cur_title, cur_level = _clean_title(ln), lvl
                cur_items.append((page, ln))
    close()

    if n_headings < 2 or len(sections) < 2:                           # no usable headings -> by pages
        sections, group, words = [], [], 0
        for page, lines in cleaned:
            group.extend((page, ln) for ln in lines)
            words += sum(len(x.split()) for x in lines)
            if words >= 450:
                a, b = group[0][0], group[-1][0]
                sections.append(_make_section(f"Page {a}" if a == b else f"Pages {a}\u2013{b}", 1, group))
                group, words = [], 0
        if group:
            a, b = group[0][0], group[-1][0]
            sections.append(_make_section(f"Page {a}" if a == b else f"Pages {a}\u2013{b}", 1, group))

    final = []
    for sec in sections:
        final.extend(_split_long(sec))
    for i, sec in enumerate(final, 1):
        sec["id"] = f"s{i:02d}"
    return [s for s in final if s["words"] >= 25]


# ---------------------------------------------------------------------- chunking
def chunk_section(sec: dict, doc_id: str, concept_id: str) -> list:
    words = sec["text"].split()
    step = CHUNK_WORDS - CHUNK_OVERLAP
    chunks = []
    for k, start in enumerate(range(0, max(len(words), 1), step)):
        piece = words[start:start + CHUNK_WORDS]
        if len(piece) < 25 and chunks:
            break
        text = " ".join(piece)
        chunks.append({
            "chunk_id": f"{doc_id}_{sec['id']}_{k}", "concept_id": concept_id, "source": "pdf",
            "doc_id": doc_id, "section_id": sec["id"], "section_title": sec["title"],
            "page_start": sec["page_start"], "page_end": sec["page_end"],
            "text": text, "word_count": len(piece), "chunk_index": k,
            "embed_text": f"{sec['title']}. {text}",           # title helps retrieval, not stored
        })
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


# ---------------------------------------------------------------------- registry
def _read_registry() -> dict:
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def _write_registry(reg: dict):
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_name("registry.json.tmp")
    tmp.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, REGISTRY_PATH)


def list_documents() -> list:
    """Newest first.  Each entry: doc_id, title, filename, concept_id, pages, sections, chunks, added_at."""
    return sorted(_read_registry().values(), key=lambda d: d.get("added_at", ""), reverse=True)


def get_document(doc_id: str):
    try:
        return json.loads((LIBRARY_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


def delete_document(doc_id: str) -> bool:
    reg = _read_registry()
    if doc_id not in reg:
        return False
    store.remove_where(lambda m: m.get("doc_id") == doc_id)
    (LIBRARY_DIR / f"{doc_id}.json").unlink(missing_ok=True)
    reg.pop(doc_id)
    _write_registry(reg)
    return True


def legacy_chunk_count() -> int:
    """PDF chunks from the first version (no doc_id).  They can't be shown or deleted individually."""
    try:
        _, meta = store.load()
    except Exception:
        return 0
    return sum(1 for m in meta if m.get("source") == "pdf" and not m.get("doc_id"))


def cleanup_legacy() -> int:
    return store.remove_where(lambda m: m.get("source") == "pdf" and not m.get("doc_id"))


# ------------------------------------------------------------------ reader helpers
def estimate_read_minutes(words: int) -> int:
    return max(1, round(words / 200))


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>$~|])")
_BULLET = re.compile(r"^[\u2022\u25cf\u25aa\u25e6\u00b7\u2023\u2043\-\*]\s+")


def reflow_markdown(text: str) -> str:
    """PDF text has a line break every ~90 characters.  Rebuild paragraphs and bullets so
    it reads like a document, and escape characters Markdown would otherwise act on."""
    esc = lambda s: _MD_SPECIAL.sub(r"\\\1", s)
    lines = [ln.strip() for ln in text.splitlines()]
    sizes = [len(ln) for ln in lines if len(ln) > 30]
    typical = statistics.median(sizes) if sizes else 80
    out, cur = [], []

    def flush():
        if cur:
            para = " ".join(cur)
            out.append("\\" + para if para[:1] in "#+" else para)         # a leading # would become a heading
            cur.clear()

    for ln in lines:
        if not ln:
            flush()
            continue
        if _BULLET.match(ln):
            flush()
            out.append("- " + esc(_BULLET.sub("", ln)))
            continue
        if heading_level(ln) is not None and len(ln) < 90:
            flush()
            out.append("**" + esc(ln) + "**")
            continue
        cur.append(esc(ln))
        if ln.endswith((".", "?", "!")) and len(ln) < 0.7 * typical:       # short last line = paragraph end
            flush()
    flush()
    return "\n\n".join(out)


# ------------------------------------------------------------------------ ingest
def _title_from_name(name: str) -> str:
    stem = re.sub(r"\.pdf$", "", name, flags=re.I)
    stem = re.sub(r"[-_]+", " ", stem)
    stem = re.sub(r"\s+pdf$", "", stem, flags=re.I).strip()
    return stem or "Untitled notes"


def ingest_pdf(pdf_path, concept_id=None, save_txt=True, original_name=None) -> dict:
    """
    Returns {"success", "doc_id", "concept_id", "pages", "sections", "chunks", "error",
             "duplicate", "needs_concept"}.   (`save_txt` is accepted for old callers; the
    library JSON replaces the old *_pdf_extracted.txt files.)
    """
    res = {"success": False, "doc_id": None, "concept_id": concept_id, "pages": 0,
           "sections": 0, "chunks": 0, "error": None, "duplicate": False, "needs_concept": False}
    try:
        data = Path(pdf_path).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        doc_id = sha[:12]
        reg = _read_registry()
        if doc_id in reg:
            res.update(doc_id=doc_id, duplicate=True, concept_id=reg[doc_id]["concept_id"],
                       error=f"'{reg[doc_id]['title']}' is already in your library - nothing was added twice.")
            return res

        pages = extract_pages(pdf_path)
        cleaned = clean_pages(pages)
        if not cleaned:
            res["error"] = ("No readable text found. This looks like a scanned (image-only) PDF - "
                            "try a PDF where you can select the text.")
            return res
        sections = split_sections(cleaned)
        if not sections:
            res["error"] = "The PDF has too little text to make sections from."
            return res

        if concept_id is None:
            sample = " ".join(s["text"] for s in sections)
            concept_id = detect_concept_from_text(sample)
            if concept_id is None:
                res.update(needs_concept=True,
                           error="I couldn't tell which subject this belongs to - please pick one.")
                return res
        res["concept_id"] = concept_id

        chunks = [c for s in sections for c in chunk_section(s, doc_id, concept_id)]
        store.add_chunks(chunks)

        title = _title_from_name(original_name) if original_name else (
            _title_from_name(Path(pdf_path).name) if not Path(pdf_path).name.lower().startswith("tmp")
            else sections[0]["title"])
        LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
        (LIBRARY_DIR / f"{doc_id}.json").write_text(json.dumps(
            {"doc_id": doc_id, "title": title, "concept_id": concept_id, "sections": sections},
            ensure_ascii=False), encoding="utf-8")
        reg[doc_id] = {"doc_id": doc_id, "title": title, "filename": original_name or Path(pdf_path).name,
                       "concept_id": concept_id, "pages": len(pages), "sections": len(sections),
                       "chunks": len(chunks), "sha256": sha,
                       "added_at": datetime.now(timezone.utc).isoformat()}
        _write_registry(reg)
        res.update(success=True, doc_id=doc_id, pages=len(pages), sections=len(sections), chunks=len(chunks))
        return res
    except ImportError as exc:
        res["error"] = f"Missing package: {exc.name}. Run: pip install pdfplumber sentence-transformers faiss-cpu"
    except Exception as exc:
        res["error"] = f"Could not process this PDF: {exc}"
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Add a PDF to your Retent Engram library")
    ap.add_argument("--file", required=True)
    ap.add_argument("--concept")
    ap.add_argument("--reset-legacy", action="store_true", help="remove PDF chunks from the first version")
    a = ap.parse_args()
    if a.reset_legacy:
        print("removed", cleanup_legacy(), "legacy chunks")
    r = ingest_pdf(a.file, a.concept, original_name=Path(a.file).name)
    print(r)
