"""
My Notes  -  upload a PDF, READ it section by section, search it, and turn any section into
flashcards or a quiz.  Marking a section studied (or finishing a quiz) is logged automatically,
so your recall scores and Today list update without a separate "Log Session" step.

Put this file where your current "My Notes" page lives (overwrite it), or point the My Notes
entry of your st.navigation(...) at it.  It only uses plain Streamlit widgets, so it picks up
the theme/CSS your main file already applies.
"""
import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "backend").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import CONCEPTS_PATH
from backend.study_log import get_studied_sections, make_logger
from frontend.components import practice
from rag import generate as gen
from rag import pdf_ingest as lib
from rag import store

USER_KEYS = ("user_id", "username", "user")          # session_state keys your login may use

try:
    st.set_page_config(page_title="My Notes", page_icon=":material/menu_book:", layout="wide")
except Exception:
    pass                                              # already set by the main file


# ------------------------------------------------------------------------------ helpers
def current_user():
    for k in USER_KEYS:
        if st.session_state.get(k):
            return str(st.session_state[k])
    return None


def concept_names() -> dict:
    try:
        return {c["concept_id"]: c["name"] for c in json.loads(Path(CONCEPTS_PATH).read_text(encoding="utf-8"))}
    except Exception:
        return {}


def flash(kind: str, msg: str):
    st.session_state["_notes_flash"] = (kind, msg)


def show_flash():
    item = st.session_state.pop("_notes_flash", None)
    if item:
        getattr(st, item[0])(item[1])


@st.cache_data(ttl=10, show_spinner=False)
def llm_status() -> dict:
    return gen.ollama_status()


def open_section(doc_id: str, section_id: str):      # button callbacks run before the rerun, so state is safe to set
    st.session_state["notes_doc"] = doc_id
    st.session_state[f"sec_{doc_id}"] = section_id


# ------------------------------------------------------------------------------ page
st.title("My Notes")
st.caption("Upload your PDFs, read them section by section, and turn any section into flashcards or a quiz.")

user_id = current_user()
if not user_id:
    st.warning("Please sign in first.")
    st.stop()

show_flash()
names = concept_names()
llm = llm_status()
llm_ok = bool(llm["running"] and llm["model"])
if llm_ok and not st.session_state.get("_notes_warmed"):
    gen.warm_up()                                    # load the model in the background while the student reads
    st.session_state["_notes_warmed"] = True

# ---- leftovers from the first version of the importer ---------------------------------------
n_legacy = lib.legacy_chunk_count()
if n_legacy:
    st.warning(
        f"{n_legacy} text chunks from an older PDF import are still in the knowledge base. They can't be "
        "shown here and may contain duplicates. Remove them, then upload the PDF again.")
    if st.button("Remove old chunks"):
        removed = lib.cleanup_legacy()
        flash("success", f"Removed {removed} old chunks. Now upload your PDF below.")
        st.rerun()


# ---- upload ---------------------------------------------------------------------------------------
def render_upload():
    up = st.file_uploader("Choose a PDF with selectable text", type=["pdf"], key="notes_upload")
    subject_ids = [None] + list(names.keys())
    subject = st.selectbox("Subject", subject_ids, key="notes_subject",
                           format_func=lambda cid: "Detect automatically" if cid is None else names[cid])
    if up is not None and st.button("Add to my notes", type="primary", key="notes_add"):
        with st.status("Reading your PDF...", expanded=True) as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(up.getvalue())
                tmp_path = tmp.name
            st.write("Extracting text and splitting it into sections...")
            res = lib.ingest_pdf(tmp_path, concept_id=subject, original_name=up.name)
            Path(tmp_path).unlink(missing_ok=True)
            status.update(label="Done" if res["success"] else "Not added",
                          state="complete" if res["success"] else "error", expanded=False)
        if res["success"]:
            st.session_state["notes_doc"] = res["doc_id"]
            flash("success", f"Added '{up.name}': {res['pages']} pages, {res['sections']} sections, "
                             f"subject {names.get(res['concept_id'], res['concept_id'])}.")
            st.rerun()
        elif res["duplicate"]:
            st.session_state["notes_doc"] = res["doc_id"]
            st.info(res["error"])
        elif res["needs_concept"]:
            st.warning(res["error"] + " Choose the subject above and press the button again.")
        else:
            st.error(res["error"])


docs = lib.list_documents()
if not docs:
    with st.container(border=True):
        st.subheader("Add your first notes")
        st.write("Upload a lecture-notes or textbook PDF. It will be split into sections you can read here, "
                 "and the AI will use it to write flashcards and quizzes grounded in your own material.")
        render_upload()
    st.stop()

with st.expander("Add another PDF"):
    render_upload()

# ---- search ---------------------------------------------------------------------------------------
query = st.text_input("Search your notes", key="notes_query",
                      placeholder="e.g. circuit switching vs packet switching")
if query.strip():
    with st.spinner("Searching your notes..."):
        hits = store.search(query.strip(), k=5)
    if not hits:
        st.info("Nothing found. Try different words.")
    for i, h in enumerate(hits):
        with st.container(border=True):
            where = h.get("section_title") or names.get(h.get("concept_id"), "Built-in note")
            pages = f" - pages {h['page_start']}-{h['page_end']}" if h.get("page_start") else ""
            st.markdown(f"**{where}**{pages}")
            snippet = " ".join(h["text"].split())[:320]
            st.write(snippet + ("..." if len(h["text"]) > 320 else ""))
            if h.get("doc_id") and h.get("section_id"):
                st.button("Open this section", key=f"open_hit_{i}", on_click=open_section,
                          args=(h["doc_id"], h["section_id"]))
    if hits and llm_ok and st.button("Answer my question from these passages (AI)", key="notes_answer"):
        with st.status("Reading your notes...", expanded=True) as status:
            try:
                st.write_stream(gen.stream_ollama(gen.answer_prompt(query.strip(), hits[:3]), "answer"))
                status.update(label="Answer", state="complete")
            except RuntimeError as exc:
                status.update(label="Could not answer", state="error")
                st.error(str(exc))
    st.divider()

# ---- document + reader ------------------------------------------------------------------------------
doc_ids = [d["doc_id"] for d in docs]
labels = {d["doc_id"]: f"{d['title']}  -  {names.get(d['concept_id'], d['concept_id'])}" for d in docs}
if st.session_state.get("notes_doc") not in doc_ids:
    st.session_state["notes_doc"] = doc_ids[0]
doc_id = st.selectbox("Document", doc_ids, key="notes_doc", format_func=lambda d: labels[d])
entry = next(d for d in docs if d["doc_id"] == doc_id)
doc = lib.get_document(doc_id)
if not doc or not doc.get("sections"):
    st.error("This document's sections are missing. Delete it and upload the PDF again.")
    st.stop()

sections = doc["sections"]
sec_ids = [s["id"] for s in sections]
sec_by_id = {s["id"]: s for s in sections}
studied = get_studied_sections(user_id, doc_id) & set(sec_ids)

st.progress(len(studied) / len(sec_ids), text=f"{len(studied)} of {len(sec_ids)} sections studied")

left, right = st.columns([1, 2.4])
with left:
    st.markdown("**Contents**")
    if st.session_state.get(f"sec_{doc_id}") not in sec_ids:
        st.session_state[f"sec_{doc_id}"] = sec_ids[0]
    with st.container(height=560):
        sec_id = st.radio("Sections", sec_ids, key=f"sec_{doc_id}", label_visibility="collapsed",
                          format_func=lambda s: ("\u2713 " if s in studied else "") + sec_by_id[s]["title"])

sec = sec_by_id[sec_id]
minutes = lib.estimate_read_minutes(sec["words"])
gkey = f"gen:{doc_id}:{sec_id}"

with right:
    st.subheader(sec["title"])
    pages = (f"Page {sec['page_start']}" if sec["page_start"] == sec["page_end"]
             else f"Pages {sec['page_start']}-{sec['page_end']}")
    st.caption(f"{pages}  -  about {minutes} min read  -  {sec['words']} words")
    with st.container(height=420, border=True):
        st.markdown(lib.reflow_markdown(sec["text"]))

    logger = make_logger(user_id, entry["concept_id"], doc_id=doc_id, section_id=sec_id)

    # -- reading is logged automatically when you say how it went ---------------------------------------
    if practice.self_rating(f"{gkey}:read", logger, prompt="Finished reading? How well did it go?",
                            event_type="reading", source="notes", minutes=minutes):
        st.rerun()                                   # refresh the tick next to this section

    # -- AI practice from this section (results are saved automatically) ---------------------------------
    st.markdown("**Practice this section**")
    if not llm_ok:
        st.caption(llm["hint"] or "Start Ollama to enable AI practice.")
    elif llm["hint"]:
        st.caption(llm["hint"])
    p1, p2, p3 = st.columns(3)
    kind = None
    with p1:
        if st.button("Flashcards", key=f"btn_fc_{gkey}", disabled=not llm_ok):
            kind = "flashcards"
    with p2:
        if st.button("Quiz me", key=f"btn_qz_{gkey}", disabled=not llm_ok):
            kind = "section_quiz"
    with p3:
        if st.button("Summary", key=f"btn_sm_{gkey}", disabled=not llm_ok):
            kind = "section_summary"

    if kind:
        practice.reset(f"{gkey}:{kind}")
        with st.status("Writing from your notes...", expanded=True) as status:
            try:
                raw = st.write_stream(gen.stream_ollama(gen.section_prompt(kind, sec["title"], sec["text"]), kind))
                status.update(label="Ready", state="complete", expanded=False)
            except RuntimeError as exc:
                status.update(label="Could not generate", state="error")
                st.error(str(exc))
                raw = None
        if raw:
            parsed = (gen.parse_flashcards(raw) if kind == "flashcards"
                      else gen.parse_quiz(raw) if kind == "section_quiz" else raw)
            st.session_state[f"{gkey}:{kind}"] = {"raw": raw, "parsed": parsed}
            st.rerun()

    fc = st.session_state.get(f"{gkey}:flashcards")
    if fc:
        st.markdown("**Flashcards**")
        if not fc["parsed"]:
            st.warning("The model's answer could not be turned into cards. Here it is as written:")
            st.write(fc["raw"])
        elif practice.flashcards(fc["parsed"], f"{gkey}:flashcards:w", logger, source="notes_flashcards"):
            st.rerun()

    qz = st.session_state.get(f"{gkey}:section_quiz")
    if qz:
        st.markdown("**Quiz**")
        if not qz["parsed"]:
            st.warning("The model's answer could not be turned into a quiz. Here it is as written:")
            st.write(qz["raw"])
        else:
            practice.quiz(qz["parsed"], f"{gkey}:section_quiz:w", logger, source="notes_quiz")

    sm = st.session_state.get(f"{gkey}:section_summary")
    if sm:
        st.markdown("**Summary**")
        with st.container(border=True):
            st.markdown(sm["raw"])

# ---- manage -------------------------------------------------------------------------------------------
with st.expander("Manage this document"):
    s = store.stats()["by_doc"].get(doc_id, 0)
    st.caption(f"{entry['filename']}  -  {entry['pages']} pages  -  {entry['sections']} sections  -  {s} searchable chunks")
    if st.button("Delete this document", key=f"del_{doc_id}"):
        st.session_state[f"confirm_del_{doc_id}"] = True
    if st.session_state.get(f"confirm_del_{doc_id}"):
        st.warning("This removes the document and its searchable chunks. Your study history is kept.")
        if st.button("Yes, delete it", key=f"del_yes_{doc_id}", type="primary"):
            lib.delete_document(doc_id)
            st.session_state.pop(f"confirm_del_{doc_id}", None)
            flash("success", "Document deleted.")
            st.rerun()
