"""
frontend/pages/6_settings.py
==============================
PHASE 7 — User Settings Page

SECTIONS:
  1. Profile settings   — change display name
  2. Daily goal         — set review target
  3. Data management    — reset events, clear all data
  4. App info           — version, model status
"""

import streamlit as st
import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.db import get_collection
from backend.scheduler import set_daily_goal, get_or_create_daily_goal

# =============================================================================
# PAGE SETUP
# =============================================================================

st.set_page_config(
    page_title="Settings — Retent Engram",
    page_icon="⚙️",
    layout="centered"
)

st.title("⚙️ Settings")
st.caption("Manage your profile, preferences, and data.")

# =============================================================================
# USER CHECK
# =============================================================================

if "user_id" not in st.session_state or not st.session_state.user_id:
    st.warning("⚠️ Set your user ID on the Home page first.")
    st.stop()

user_id   = st.session_state.user_id
user_name = st.session_state.get("user_name", user_id)

st.info(f"Logged in as **{user_name}** (`{user_id}`)")
st.divider()


# =============================================================================
# SECTION 1 — PROFILE SETTINGS
# =============================================================================

st.subheader("👤 Profile")

users_col = get_collection("users")
user_doc  = users_col.find_one({"user_id": user_id}, {"_id": 0}) or {}

new_name = st.text_input(
    "Display Name",
    value=user_doc.get("name", user_name),
    help="This name appears across all pages."
)

if st.button("💾 Save Name", type="primary"):
    if new_name.strip():
        users_col.update_one(
            {"user_id": user_id},
            {"$set": {"name": new_name.strip()}},
            upsert=True
        )
        st.session_state.user_name = new_name.strip()
        st.success(f"Name updated to **{new_name}**!")
    else:
        st.warning("Name cannot be empty.")

st.divider()


# =============================================================================
# SECTION 2 — DAILY GOAL
# =============================================================================

st.subheader("🎯 Daily Review Goal")
st.caption("How many concepts do you want to review each day?")

current_goal = get_or_create_daily_goal(user_id)

new_goal = st.slider(
    "Concepts per day",
    min_value=1,
    max_value=10,
    value=current_goal,
    step=1
)

if st.button("💾 Save Goal"):
    if set_daily_goal(user_id, new_goal):
        st.success(f"Daily goal set to **{new_goal}** concepts/day!")

st.divider()


# =============================================================================
# SECTION 3 — DATA MANAGEMENT
# =============================================================================

st.subheader("🗑️ Data Management")
st.caption("⚠️ These actions are permanent and cannot be undone.")

# ── Reset events for ONE concept ──────────────────────────────────────────────
with st.expander("🔄 Reset events for a specific concept"):
    import json
    CONCEPTS_PATH = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "concepts.json"
    )
    with open(CONCEPTS_PATH) as f:
        concepts = json.load(f)
    concept_options = {c["name"]: c["concept_id"] for c in concepts}

    reset_concept_name = st.selectbox(
        "Select concept to reset",
        list(concept_options.keys()),
        key="reset_concept"
    )
    reset_cid = concept_options[reset_concept_name]

    if st.button(f"🗑️ Delete all events for '{reset_concept_name}'",
                 type="secondary"):
        events_col  = get_collection("events")
        scores_col  = get_collection("recall_scores")
        content_col = get_collection("generated_content")

        deleted_events = events_col.delete_many(
            {"user_id": user_id, "concept_id": reset_cid}
        ).deleted_count
        scores_col.delete_one(
            {"user_id": user_id, "concept_id": reset_cid}
        )
        content_col.delete_many(
            {"user_id": user_id, "concept_id": reset_cid}
        )

        st.success(
            f"Deleted {deleted_events} events and all recall data "
            f"for '{reset_concept_name}'."
        )

# ── Clear ALL data ────────────────────────────────────────────────────────────
with st.expander("💣 Clear ALL my data (nuclear option)"):
    st.error(
        "**WARNING:** This deletes ALL your events, recall scores, and "
        "generated content. This cannot be undone."
    )

    confirm_text = st.text_input(
        f"Type your user ID to confirm deletion",
        placeholder=f"Type: {user_id}",
        key="confirm_delete"
    )

    if st.button("🗑️ DELETE ALL MY DATA", type="primary"):
        if confirm_text.strip().lower() == user_id.lower():
            events_col  = get_collection("events")
            scores_col  = get_collection("recall_scores")
            content_col = get_collection("generated_content")

            e = events_col.delete_many({"user_id": user_id}).deleted_count
            s = scores_col.delete_many({"user_id": user_id}).deleted_count
            c = content_col.delete_many({"user_id": user_id}).deleted_count

            users_col.update_one(
                {"user_id": user_id},
                {"$set": {"streak_days": 0, "last_active_date": None}}
            )

            st.success(
                f"Deleted {e} events, {s} recall scores, "
                f"{c} content pieces. Fresh start!"
            )

            # Clear session state
            st.session_state["user_id"]   = ""
            st.session_state["user_name"] = ""
            st.rerun()
        else:
            st.error("User ID doesn't match. Deletion cancelled.")

st.divider()


# =============================================================================
# SECTION 4 — APP INFO
# =============================================================================

st.subheader("ℹ️ App Info")

info_data = {
    "App":          "Retent Engram",
    "Version":      "Phase 7 — Final",
    "Database":     "pkdp_db (MongoDB)",
    "Streamlit":    st.__version__,
    "Python":       f"{sys.version_info.major}.{sys.version_info.minor}",
}

for k, v in info_data.items():
    c1, c2 = st.columns([2, 3])
    with c1:
        st.caption(f"**{k}**")
    with c2:
        st.caption(v)

# Model status
model_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "recall_model.pkl"
)
if os.path.exists(model_path):
    import json
    meta_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "models", "model_metadata.json"
    )
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        active = meta.get("active_model", "unknown")
        metrics = meta.get(active, {})
        st.success(
            f"**ML Model:** {active}  •  "
            f"AUC: {metrics.get('auc','N/A')}  •  "
            f"Brier: {metrics.get('brier','N/A')}"
        )
else:
    st.warning("**ML Model:** Not trained yet  ·  run `python scripts/run_training.py`")