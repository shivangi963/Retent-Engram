"""
frontend/components/data.py
===========================
Small shared helpers used by several pages (session, concepts, formatting).
"""
import json
import os
from datetime import datetime, timezone

import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONCEPTS_PATH = os.path.join(ROOT, "data", "concepts.json")

HIGH_THRESHOLD = 40.0
MEDIUM_THRESHOLD = 65.0

PRIORITY_TONE = {"High": "red", "Medium": "amber", "Low": "green"}

# priority -> (button label, content type to open in the Study page)
ACTIONS = {
    "High":   ("Summary + Quiz", "summary"),
    "Medium": ("Flashcards",     "flashcard"),
    "Low":    ("No Action",      None),
}

CONTENT_LABELS = {
    "flashcard":   "Flashcard",
    "summary":     "Summary",
    "quiz":        "Quiz",
    "coding_task": "Coding Task",
}


@st.cache_data
def load_concepts() -> list:
    with open(CONCEPTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def concepts_by_id() -> dict:
    return {c["concept_id"]: c for c in load_concepts()}


def priority_for(score: float) -> str:
    if score < HIGH_THRESHOLD:
        return "High"
    if score < MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def current_user() -> tuple:
    """Returns (user_id, user_name). The app shell guarantees a login."""
    uid = st.session_state.get("user_id", "")
    if not uid:
        st.stop()
    return uid, st.session_state.get("user_name") or uid


def greeting() -> str:
    h = datetime.now().hour
    return "Good morning" if h < 12 else ("Good afternoon" if h < 17 else "Good evening")


def ago(hours: float) -> str:
    """Hours -> friendly 'time ago' string."""
    if hours is None:
        return "Never"
    if hours < 1:
        return "Just now"
    if hours < 24:
        return f"{int(hours)}h ago"
    if hours < 48:
        return "Yesterday"
    return f"{int(hours // 24)} days ago"


def ago_from_dt(dt) -> str:
    if not dt:
        return "Never"
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return ago((datetime.now(timezone.utc) - dt).total_seconds() / 3600)


def flash(message: str, icon: str = ":material/check_circle:"):
    """Show a toast on the NEXT run (survives st.rerun / page switch)."""
    st.session_state["_flash"] = (message, icon)


def show_flash():
    item = st.session_state.pop("_flash", None)
    if item:
        st.toast(item[0], icon=item[1])


def open_study(concept_id: str, content_type: str | None = None):
    """Jump to the Study page with a concept (and content type) pre-selected."""
    st.session_state["study_prefill"] = {"concept_id": concept_id, "content_type": content_type}
    st.switch_page("views/study.py")
