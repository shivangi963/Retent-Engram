import streamlit as st
import json
import os
import sys
import pandas as pd
from datetime import datetime

# Let Python find the db/ folder
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.db import get_or_create_user, get_recent_events, insert_event
from backend.models.event import create_event


st.set_page_config(page_title="Log Event", layout="centered")
st.title("Log Study Event")
st.caption("Record a study session to track your recall over time.")

CONCEPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "concepts.json")

@st.cache_data
def load_concepts():
    with open(CONCEPTS_PATH, "r") as f:
        return json.load(f)

concepts = load_concepts()
concept_name_to_id = {c["name"]: c["concept_id"] for c in concepts}


if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

st.subheader("Who are you?")
col1, col2 = st.columns(2)
with col1:
    name_input = st.text_input("Your name")
with col2:
    uid_input = st.text_input("User ID (no spaces)")

if st.button("Set User / Login"):
    if not name_input.strip() or not uid_input.strip():
        st.warning("Please fill in both fields.")
    elif " " in uid_input.strip():
        st.warning("User ID must not contain spaces.")
    else:
        get_or_create_user(uid_input.strip(), name_input.strip())
        st.session_state.user_id = uid_input.strip().lower()
        st.session_state.user_name = name_input.strip()
        st.success(f"Logged in as **{name_input}**  •  ID: `{uid_input.lower()}`")

if not st.session_state.user_id:
    st.info("Set your user above to start logging events.")
    st.stop()

st.divider()
st.subheader(f"Log a session — {st.session_state.user_name}")

with st.form("log_event_form", clear_on_submit=True):
    concept_name = st.selectbox("Concept studied", options=list(concept_name_to_id.keys()))

    event_type = st.selectbox("Type of session", ["reading", "quiz", "coding"])

    score = st.slider(
        "Score (0.0 = couldn't recall, 1.0 = perfect)",
        min_value=0.0, max_value=1.0, value=0.7, step=0.05
    )

    response_time = st.number_input(
        "Time spent (minutes)", min_value=1, max_value=300, value=20, step=5
    )

    hints_used = st.number_input(
        "Hints / references used", min_value=0, max_value=20, value=0,
        help="How many times you looked something up"
    )

    submitted = st.form_submit_button("Save Event", type="primary", use_container_width=True)

if submitted:
    concept_id = concept_name_to_id[concept_name]
    event_doc = create_event(
        user_id=st.session_state.user_id,
        concept_id=concept_id,
        event_type=event_type,
        score=score,
        response_time_min=response_time,
        hints_used=hints_used
    )
    insert_event(event_doc)
    st.success(f"Saved! **{concept_name}** — {event_type} — score {score:.0%}")
    st.balloons()


st.divider()
st.subheader("Recent Events")

events = get_recent_events(st.session_state.user_id, limit=20)
id_to_name = {c["concept_id"]: c["name"] for c in concepts}

if not events:
    st.info("No events yet. Log your first session above!")
else:
    rows = []
    for e in events:
        ts = e.get("timestamp", "")
        rows.append({
            "Concept": id_to_name.get(e.get("concept_id", ""), e.get("concept_id", "")),
            "Type": e.get("event_type", ""),
            "Score": f"{e.get('score', 0):.0%}",
            "Time (min)": e.get("response_time_min", e.get("response_time_sec", "—")),
            "Hints": e.get("hints_used", 0),
            "When": ts.strftime("%d %b %Y, %I:%M %p") if hasattr(ts, "strftime") else str(ts)
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"Showing last {len(rows)} events")