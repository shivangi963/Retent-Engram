import streamlit as st
from datetime import datetime
import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from db.mongo_client import insert_event, get_events_by_user

st.set_page_config(page_title="PKDP - Knowledge Tracker", layout="centered")
st.title("📚 Personal Knowledge Decay Predictor")
st.subheader("Log a Study Event")

with st.form("event_form"):
    user_id = st.text_input("Your User ID", value="user_001")
    concept = st.selectbox("Concept", ["binary_search", "recursion", "dynamic_programming", "os_scheduling"])
    event_type = st.selectbox("Event Type", ["reading", "quiz", "coding"])
    score = st.slider("Score (0 = forgot, 1 = perfect)", 0.0, 1.0, 0.7, step=0.1)
    response_time = st.number_input("Response Time (seconds)", min_value=1, value=30)
    difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"])
    hints = st.number_input("Hints Used", min_value=0, value=0)
    submitted = st.form_submit_button("💾 Log Event")

if submitted:
    event = {
        "user_id": user_id,
        "concept_id": concept,
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "score": score,
        "response_time_sec": response_time,
        "difficulty": difficulty,
        "hints_used": hints
    }
    doc_id = insert_event(event)
    st.success(f"✅ Event logged! MongoDB ID: {doc_id}")

st.divider()
st.subheader("📊 Your Event History")
uid = st.text_input("Enter User ID to view history", value="user_001")
if st.button("Fetch Events"):
    events = get_events_by_user(uid)
    if events:
        df = pd.DataFrame(events)
        st.dataframe(df)
    else:
        st.warning("No events found for this user.")