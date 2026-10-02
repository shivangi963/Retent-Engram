"""
frontend/views/log_session.py
==============================
Log a study session (reading / quiz / coding) and see recent sessions.
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.db import get_recent_events, insert_event
from backend.models.event import create_event

from frontend.components.theme import html, page_header, section_title, card, icon, empty_state, TONES
from frontend.components.data import concepts_by_id, current_user, flash

user_id, user_name = current_user()
concepts = concepts_by_id()
name_to_id = {c["name"]: cid for cid, c in concepts.items()}

page_header("Log Session", "Record a study session so Retent Engram can track your recall.")

left, right = st.columns([1, 1.3], gap="large")

with left:
    with card("log_form"):
        html('<div style="font-weight:700;color:#0F1F3D;font-size:1.02rem;margin-bottom:.6rem">New session</div>')
        with st.form("log_event_form", clear_on_submit=True, border=False):
            concept_name = st.selectbox("Concept studied", options=list(name_to_id.keys()))
            event_type = st.segmented_control("Type of session", options=["reading", "quiz", "coding"],
                                                default="reading")
            score = st.slider("Score", 0.0, 1.0, 0.7, 0.05,
                               help="0.0 = couldn't recall it at all · 1.0 = perfect recall")
            c1, c2 = st.columns(2)
            with c1:
                response_time = st.number_input("Time spent (min)", 1, 300, 20, 5)
            with c2:
                hints_used = st.number_input("Hints used", 0, 20, 0)
            submitted = st.form_submit_button("Save session", type="primary", width="stretch",
                                               icon=":material/save:")

        if submitted:
            event_type = event_type or "reading"
            concept_id = name_to_id[concept_name]
            event_doc = create_event(
                user_id=user_id, concept_id=concept_id, event_type=event_type,
                score=score, response_time_min=response_time, hints_used=hints_used,
            )
            insert_event(event_doc)
            flash(f"Saved — {concept_name} · {event_type} · {score:.0%}")
            st.rerun()

with right:
    section_title("Recent sessions")
    events = get_recent_events(user_id, limit=20)
    id_to_name = {c["concept_id"]: c["name"] for c in concepts.values()}

    if not events:
        empty_state("edit_note" if False else "inbox", "No sessions yet",
                     "Log your first session on the left to start tracking recall.")
    else:
        with card("recent_list"):
            for e in events:
                ts = e.get("timestamp", "")
                ts_s = ts.strftime("%d %b, %I:%M %p") if hasattr(ts, "strftime") else str(ts)
                score_val = e.get("score", 0)
                tone = "green" if score_val >= 0.8 else ("amber" if score_val >= 0.6 else "red")
                fg = TONES[tone][0]
                cname = id_to_name.get(e.get("concept_id", ""), e.get("concept_id", ""))
                time_min = e.get("response_time_min", e.get("response_time_sec", "—"))
                html(f"""<div class="rt-row">
                    <div><div class="rt-row-n">{cname}</div>
                         <div class="rt-row-s">{e.get('event_type','')} · {ts_s} · {time_min} min</div></div>
                    <div class="rt-row-r"><div class="rt-row-sc" style="color:{fg}">{score_val:.0%}</div></div>
                  </div>""")
        st.caption(f"Showing last {len(events)} sessions")
