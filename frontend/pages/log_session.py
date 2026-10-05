"""
Log study  -  for study done OUTSIDE the app only.

Anything you do inside Retent Engram (quizzes, flashcards, coding tasks, reading a section in
My Notes) is saved automatically.  This page is for the rest: learning a new concept somewhere
else, or re-reading something you already knew.  Five quick choices, one click to save.

Put this file where your current "Log Session" page lives (overwrite it), or point the
"Log Session" entry of your st.navigation(...) at it.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "backend").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import CONCEPTS_PATH
from backend.study_log import (MANUAL, log_study, recent_events, delete_event, describe_event,
                               describe_due, count_concept_events, to_local)

try:
    st.set_page_config(page_title="Log study", page_icon=":material/edit_note:", layout="wide")
except Exception:
    pass

USER_KEYS = ("user_id", "username", "user")
ACTIVITIES = {"Learned something new": "learned", "Revised something I knew": "revised"}
OUTCOMES = {"Understood it well": 0.85, "Mostly": 0.60, "Struggled": 0.30}
DURATIONS = [15, 30, 45, 60, 90, "Other"]
WHEN = ["Just now", "Earlier today", "Yesterday", "Pick a date"]


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


def duration_label(d) -> str:
    return "Other" if d == "Other" else (f"{d} min" if d < 60 else f"{d / 60:g} h")


def default_activity(user_id: str, concept_id: str) -> str:
    """Nothing logged for this concept yet -> probably the first time you learn it."""
    return "Learned something new" if count_concept_events(user_id, concept_id) == 0 else "Revised something I knew"


def resolve_when(ss):
    """-> (utc datetime, None) or (None, error message)"""
    now = datetime.now().astimezone()
    choice = ss["log_when"]
    if choice == "Just now":
        return now.astimezone(timezone.utc), None
    if choice == "Earlier today":
        day, clock = now.date(), ss["log_time_today"]
    elif choice == "Yesterday":
        day, clock = now.date() - timedelta(days=1), ss["log_time_yesterday"]
    else:
        day, clock = ss["log_date"], ss["log_time_other"]
    ts = datetime.combine(day, clock).astimezone()           # naive local time -> aware
    if ts > now + timedelta(minutes=1):
        return None, "That time has not happened yet - pick an earlier time."
    return ts.astimezone(timezone.utc), None


def save_cb(user_id: str, names: dict):
    """Runs BEFORE the page reruns, so it may reset the form widgets."""
    ss = st.session_state
    concept_id = ss["log_concept"]
    ts, err = resolve_when(ss)
    if err:
        ss["_log_flash"] = ("error", err)
        return
    minutes = ss["log_minutes"] if ss["log_duration"] == "Other" else ss["log_duration"]
    res = log_study(user_id, concept_id, OUTCOMES[ss["log_outcome"]], "reading", minutes=minutes,
                    source=MANUAL, timestamp=ts, activity=ACTIVITIES[ss["log_activity"]],
                    note=(ss.get("log_note") or "").strip() or None)
    if not res:
        ss["_log_flash"] = ("error", "Could not save to the database. Is MongoDB running?")
        return
    msg = f"Saved: {names[concept_id]}."
    if res.get("recall") is not None:
        due = describe_due(res.get("due_in_hours"))
        msg += f" Recall is now {res['recall']:.0f}%. " + ("It is due for review now." if due == "now" else f"Next review {due}.")
    ss["_log_flash"] = ("success", msg)
    ss["log_last_concept"] = concept_id
    ss.update(log_outcome="Mostly", log_duration=30, log_when="Just now", log_note="",
              log_activity=default_activity(user_id, concept_id))


def ask_remove(event_id: str):
    st.session_state["log_confirm_rm"] = event_id


def do_remove(user_id: str, event_id, event_key: str):
    ok = delete_event(user_id, event_id)
    st.session_state["_log_flash"] = ("success", "Entry removed.") if ok else ("error", "Could not remove it.")
    st.session_state.pop("log_confirm_rm", None)


def cancel_remove():
    st.session_state.pop("log_confirm_rm", None)


# ------------------------------------------------------------------------------- page
st.title("Log study")
st.caption("Studied somewhere else? Log it here in a few clicks. Everything you do inside Retent Engram - "
           "quizzes, flashcards, coding tasks, reading in My Notes - is saved automatically, so don't log those.")

user_id = current_user()
if not user_id:
    st.warning("Please sign in first.")
    st.stop()
names = concept_names()
if not names:
    st.error("data/concepts.json is missing or empty.")
    st.stop()

flash = st.session_state.pop("_log_flash", None)
if flash:
    getattr(st, flash[0])(flash[1])

ss = st.session_state
ids = list(names)
if ss.get("log_concept") not in ids:
    ss["log_concept"] = ss.get("log_last_concept") if ss.get("log_last_concept") in ids else ids[0]
now_local = datetime.now().astimezone()
for k, v in {"log_outcome": "Mostly", "log_duration": 30, "log_when": "Just now", "log_note": "",
             "log_minutes": 30, "log_date": now_local.date() - timedelta(days=2),
             "log_time_today": max(now_local - timedelta(hours=1), now_local.replace(hour=0, minute=0)).time().replace(second=0, microsecond=0),
             "log_time_yesterday": now_local.replace(hour=18, minute=0).time().replace(second=0, microsecond=0),
             "log_time_other": now_local.replace(hour=18, minute=0).time().replace(second=0, microsecond=0)}.items():
    ss.setdefault(k, v)
if ss.get("_log_prev_concept") != ss["log_concept"] or "log_activity" not in ss:
    ss["log_activity"] = default_activity(user_id, ss["log_concept"])      # smart default, still changeable
    ss["_log_prev_concept"] = ss["log_concept"]

left, right = st.columns([1.2, 1])

with left:
    with st.container(border=True):
        st.subheader("Study done outside the app")
        st.selectbox("What did you study?", ids, key="log_concept", format_func=lambda c: names[c])
        st.radio("What did you do?", list(ACTIVITIES), key="log_activity", horizontal=True)
        st.radio("How did it go?", list(OUTCOMES), key="log_outcome", horizontal=True)
        st.radio("How long?", DURATIONS, key="log_duration", horizontal=True, format_func=duration_label)
        if ss["log_duration"] == "Other":
            st.number_input("Minutes", min_value=1, max_value=600, step=5, key="log_minutes")
        st.radio("When?", WHEN, key="log_when", horizontal=True)
        if ss["log_when"] == "Earlier today":
            st.time_input("Around what time?", key="log_time_today")
        elif ss["log_when"] == "Yesterday":
            st.time_input("Around what time?", key="log_time_yesterday")
        elif ss["log_when"] == "Pick a date":
            d1, d2 = st.columns(2)
            with d1:
                st.date_input("Which day?", key="log_date", max_value=now_local.date())
            with d2:
                st.time_input("Around what time?", key="log_time_other")
        with st.expander("Add a note (optional)"):
            st.text_input("Note", key="log_note", max_chars=140, placeholder="e.g. watched the lecture on deadlocks")
        st.button("Save", type="primary", key="log_save", on_click=save_cb, args=(user_id, names))
    st.caption("Your recall estimate and your Today list update as soon as you save.")

with right:
    st.subheader("Recent study")
    events = recent_events(user_id, 8)
    if not events:
        st.info("Nothing logged yet. Study in the app, or log something you studied elsewhere.")
    for e in events:
        label, by_hand = describe_event(e)
        eid = str(e["_id"])
        mins = e.get("response_time_min")
        if mins is None and e.get("response_time_sec") is not None:
            mins = e["response_time_sec"] / 60
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**{names.get(e['concept_id'], e['concept_id'])}**")
                detail = f"{label} - {to_local(e['timestamp']).strftime('%d %b, %I:%M %p')}"
                detail += f" - {mins:.0f} min" if mins else ""
                detail += f" - {e['note']}" if e.get("note") else ""
                st.caption(detail)
            with c2:
                st.markdown(f"**{float(e.get('score', 0)):.0%}**")
            if by_hand and ss.get("log_confirm_rm") == eid:
                st.warning("Remove this entry? Your recall estimate will be recalculated.")
                y, n = st.columns(2)
                with y:
                    st.button("Yes, remove", key=f"rm_yes_{eid}", on_click=do_remove, args=(user_id, e["_id"], eid))
                with n:
                    st.button("Keep it", key=f"rm_no_{eid}", on_click=cancel_remove)
            elif by_hand:
                st.button("Remove", key=f"rm_{eid}", on_click=ask_remove, args=(eid,))
    if events:
        st.caption(f"Showing the last {len(events)} entries. Entries made inside the app can't be removed here.")
