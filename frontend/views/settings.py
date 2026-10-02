"""
frontend/views/settings.py
============================
Profile, daily goal, data management, system status, and project info.
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.db import get_collection
from backend.scheduler import set_daily_goal, get_or_create_daily_goal

from frontend.components.theme import html, page_header, section_title, card, pill, icon
from frontend.components.data import concepts_by_id, current_user, flash
from frontend.components.system import all_checks

user_id, user_name = current_user()
concepts = concepts_by_id()

page_header("Settings", "Manage your profile, preferences, and data.")

col_l, col_r = st.columns([1.3, 1], gap="large")

with col_l:
    # ── profile ──────────────────────────────────────────────────────────────
    section_title("Profile")
    with card("profile"):
        users_col = get_collection("users")
        user_doc = users_col.find_one({"user_id": user_id}, {"_id": 0}) or {}
        new_name = st.text_input("Display name", value=user_doc.get("name", user_name))
        if st.button("Save name", type="primary", key="save_name"):
            if new_name.strip():
                users_col.update_one({"user_id": user_id}, {"$set": {"name": new_name.strip()}}, upsert=True)
                st.session_state.user_name = new_name.strip()
                flash("Name updated.")
                st.rerun()
            else:
                st.warning("Name cannot be empty.")

    # ── daily goal ───────────────────────────────────────────────────────────
    section_title("Daily review goal")
    with card("goal"):
        current_goal = get_or_create_daily_goal(user_id)
        new_goal = st.slider("Concepts per day", 1, 10, current_goal, 1, label_visibility="collapsed")
        if st.button("Save goal", key="save_goal"):
            if set_daily_goal(user_id, new_goal):
                flash(f"Daily goal set to {new_goal}.")
                st.rerun()

    # ── data management ─────────────────────────────────────────────────────
    section_title("Data management", "These actions are permanent.")
    with st.expander("Reset events for a specific concept"):
        concept_options = {c["name"]: cid for cid, c in concepts.items()}
        reset_concept_name = st.selectbox("Concept", list(concept_options.keys()), key="reset_concept")
        reset_cid = concept_options[reset_concept_name]
        if st.button(f"Delete all events for '{reset_concept_name}'", key="reset_btn"):
            events_col, scores_col, content_col = (get_collection("events"), get_collection("recall_scores"),
                                                     get_collection("generated_content"))
            deleted = events_col.delete_many({"user_id": user_id, "concept_id": reset_cid}).deleted_count
            scores_col.delete_one({"user_id": user_id, "concept_id": reset_cid})
            content_col.delete_many({"user_id": user_id, "concept_id": reset_cid})
            flash(f"Deleted {deleted} events for '{reset_concept_name}'.")
            st.rerun()

    with st.expander("Clear ALL my data"):
        st.error("This deletes all your events, recall scores, and generated content. This cannot be undone.")
        confirm_text = st.text_input("Type your learner ID to confirm", placeholder=user_id, key="confirm_delete")
        if st.button("Delete all my data", key="delete_all"):
            if confirm_text.strip().lower() == user_id.lower():
                events_col, scores_col, content_col = (get_collection("events"), get_collection("recall_scores"),
                                                         get_collection("generated_content"))
                e = events_col.delete_many({"user_id": user_id}).deleted_count
                s = scores_col.delete_many({"user_id": user_id}).deleted_count
                c = content_col.delete_many({"user_id": user_id}).deleted_count
                get_collection("users").update_one({"user_id": user_id},
                                                     {"$set": {"streak_days": 0, "last_active_date": None}})
                st.session_state.user_id = ""
                st.session_state.user_name = ""
                st.success(f"Deleted {e} events, {s} scores, {c} content pieces.")
                st.rerun()
            else:
                st.error("Learner ID doesn't match — deletion cancelled.")

with col_r:
    # ── system status ────────────────────────────────────────────────────────
    section_title("System status")
    with card("system"):
        for name, ic_name, ok, msg in all_checks():
            tone = "green" if ok else "amber"
            html(f"""<div class="rt-row">
                <div style="display:flex;align-items:center;gap:.6rem">
                  <span style="color:#64748B">{icon(ic_name, 17)}</span>
                  <div class="rt-row-n">{name}</div></div>
                <div class="rt-row-r">{pill(msg, tone, dot=True)}</div></div>""")

    # ── app info ─────────────────────────────────────────────────────────────
    section_title("App info")
    with card("appinfo"):
        info = {"App": "Retent Engram", "Database": "pkdp_db (MongoDB)",
                "Streamlit": st.__version__, "Python": f"{sys.version_info.major}.{sys.version_info.minor}"}
        for k, v in info.items():
            html(f'<div class="rt-row"><div class="rt-row-s">{k}</div><div class="rt-row-n">{v}</div></div>')

        model_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                                   "models", "recall_model.pkl")
        if os.path.exists(model_path):
            st.caption("ML model trained and active.")
        else:
            st.caption("ML model not trained yet — using the Ebbinghaus formula fallback.")

    # ── about / project info ────────────────────────────────────────────────
    with st.expander("About this project"):
        html("""
        <p style="color:#334155;font-size:.92rem;line-height:1.6">
        <b>Retent Engram: Personalised Cognitive Recall Assistor</b> predicts concept-wise forgetting
        for each learner and generates timely, personalised revision material — before recall failure
        happens. It combines event logging, memory-decay / ML scoring, an urgency-based review queue,
        and a RAG pipeline (FAISS + a local LLM) that grounds generated content in your own notes.</p>
        """)
        tech = [
            ("Frontend", "Streamlit"), ("Database", "MongoDB"),
            ("ML — baseline", "Logistic Regression"), ("ML — main", "XGBoost"),
            ("Survival model", "lifelines"), ("Experiment tracking", "MLflow"),
            ("Vector store", "FAISS"), ("Embeddings", "sentence-transformers (all-MiniLM-L6-v2)"),
            ("LLM", "Mistral 7B via Ollama"), ("Data processing", "Pandas, NumPy, SciPy"),
            ("Visualisation", "Plotly"),
        ]
        st.dataframe(pd.DataFrame(tech, columns=["Layer", "Technology"]), width="stretch", hide_index=True)

        html("""<div style="margin-top:.8rem"><b style="color:#0F1F3D">Team</b>
          <div style="color:#334155;font-size:.9rem;line-height:1.7;margin-top:.3rem">
          Shivangi Shukla (1BC23CS056) · S. Harini (1BC23CS049) · Aliya Aiman (1BC23CS006)<br>
          Guide: Mrs. Madhuri, M.E · Bangalore College of Engineering &amp; Technology<br>
          Affiliated to Visvesvaraya Technological University, Belagavi · Dept. of CSE · 2026–2027</div></div>""")
