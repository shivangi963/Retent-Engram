"""
frontend/pages/4_queue.py
===========================
PHASE 5 — Daily Review Queue Page

PURPOSE
-------
This page is the student's daily action list. When they open it they see:
  - EXACTLY which concepts to study today
  - IN WHAT ORDER (most urgent first)
  - HOW LONG it will take (estimated time)
  - WHAT TO DO (suggested study method per concept)
  - PROGRESS toward today's goal ("2 of 3 concepts reviewed")
  - THEIR STREAK ("🔥 5 day streak")

PAGE LAYOUT (top to bottom):
  ┌─────────────────────────────────────────────────────┐
  │ 📋 Daily Review Queue                               │
  │ Hello, Shivangi 👋                                  │
  ├────────────┬──────────────┬────────────┬────────────┤
  │ 🎯 Goal   │ ✅ Reviewed  │ 🔥 Streak  │ ⏱ Time    │
  │ 3 today   │ 1 / 3        │ 5 days     │ ~45 min    │
  ├────────────┴──────────────┴────────────┴────────────┤
  │ Progress bar: ██████░░░░░░ 33%                      │
  ├─────────────────────────────────────────────────────┤
  │ 🔴 [CRITICAL 87.3]  Operating Systems               │
  │    Recall: 28%  •  Last: 3 days ago  •  📝 Flashcard│
  │    [✅ Mark Reviewed]  [😴 Snooze 1d]  [😴 Snooze 3d]│
  ├─────────────────────────────────────────────────────┤
  │ 🟠 [HIGH 62.1]  Computer Networks                   │
  │    Recall: 44%  •  Last: 2 days ago  •  📖 Summary  │
  │    [✅ Mark Reviewed]  [😴 Snooze 1d]  [😴 Snooze 3d]│
  ├─────────────────────────────────────────────────────┤
  │ ▼ Snoozed concepts (1)                              │
  │   OS — snoozed until tomorrow                       │
  └─────────────────────────────────────────────────────┘

ALL INTERACTIONS ARE LIVE:
  When student clicks "Mark as Reviewed" or "Snooze":
    1. scheduler.py function is called
    2. MongoDB is updated immediately
    3. st.rerun() refreshes the page
    4. Queue updates automatically (no manual refresh needed)

SELF-RATING AFTER REVIEW:
  After clicking "Mark as Reviewed", a slider appears:
    "How well did you recall this? 0.0 → 1.0"
  Student rates themselves. This score is saved as a real event.
  This makes the ML model (Phase 4) smarter over time.
"""

import streamlit as st
import json
import os
import sys
import pandas as pd
from datetime import datetime, timezone

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.scheduler import (
    build_and_get_queue,
    mark_as_reviewed,
    snooze_concept,
    unsnooze_concept,
    set_daily_goal,
    get_urgency_emoji,
    compute_urgency_score
)
from backend.db import (
    get_collection,
    get_recent_events_for_concept
)


# =============================================================================
# PAGE SETUP
# =============================================================================

st.set_page_config(
    page_title="Review Queue — Retent Engram",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Daily Review Queue")
st.caption("Your personalized study plan for today — sorted by urgency.")


# =============================================================================
# LOAD CONCEPTS.JSON
# =============================================================================

CONCEPTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "concepts.json"
)

@st.cache_data
def load_concepts() -> dict:
    """
    Loads concepts.json and returns a lookup dict.

    Returns:
        dict: { concept_id: concept_dict }
    """
    with open(CONCEPTS_PATH, "r") as f:
        concepts_list = json.load(f)
    return {c["concept_id"]: c for c in concepts_list}

concepts_lookup = load_concepts()


# =============================================================================
# USER CHECK
# =============================================================================

if "user_id" not in st.session_state or not st.session_state.user_id:
    st.warning("⚠️ Please go to **Log Event** first and set your user ID.")
    st.stop()

user_id   = st.session_state.user_id
user_name = st.session_state.get("user_name", user_id)

st.subheader(f"Hello, {user_name} 👋")


# =============================================================================
# LOAD ALL QUEUE DATA
# =============================================================================

# Show spinner while computing urgency scores + building queue
with st.spinner("🧠 Computing urgency scores and building your queue..."):
    data = build_and_get_queue(user_id, concepts_lookup)

queue           = data["queue"]
all_concepts    = data["all_concepts"]
daily_goal      = data["daily_goal"]
reviewed_today  = data["reviewed_today"]
streak_days     = data["streak_days"]
total_time_min  = data["total_time_min"]
snoozed_list    = data["snoozed_concepts"]
is_new_day      = data["is_new_day"]

# Show new day notification
if is_new_day:
    st.success("🌅 Good morning! Yesterday's progress has been reset. Here's today's queue.")


# =============================================================================
# SECTION 1 — SUMMARY METRIC CARDS
# =============================================================================

st.divider()

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        label="🎯 Daily Goal",
        value=f"{daily_goal} concepts",
        help="Set your goal in the settings section below."
    )

with m2:
    progress_text = f"{reviewed_today} / {daily_goal}"
    if reviewed_today >= daily_goal:
        progress_label = f"✅ {progress_text}"
    else:
        progress_label = f"📚 {progress_text}"
    st.metric(
        label="Reviewed Today",
        value=progress_label,
        help="How many concepts you've reviewed today vs your daily goal."
    )

with m3:
    streak_label = f"🔥 {streak_days}" if streak_days > 0 else "😴 0"
    st.metric(
        label="Study Streak",
        value=f"{streak_label} days",
        help="Consecutive days you've reviewed at least one concept."
    )

with m4:
    if total_time_min == 0:
        time_label = "🎉 All done!"
    elif total_time_min < 60:
        time_label = f"⏱ ~{total_time_min} min"
    else:
        hours = total_time_min // 60
        mins  = total_time_min % 60
        time_label = f"⏱ ~{hours}h {mins}m"
    st.metric(
        label="Estimated Time",
        value=time_label,
        help="Total estimated time to complete today's review queue."
    )

# ── Progress bar ──────────────────────────────────────────────────────────────
if daily_goal > 0:
    progress_fraction = min(reviewed_today / daily_goal, 1.0)
    st.progress(
        progress_fraction,
        text=f"Daily goal progress: {reviewed_today} of {daily_goal} concepts reviewed"
    )

    if reviewed_today >= daily_goal:
        st.balloons()
        st.success(
            f"🎉 Daily goal reached! You've reviewed {reviewed_today} concepts today. "
            f"Come back tomorrow to keep your streak going!"
        )


# =============================================================================
# SECTION 2 — THE REVIEW QUEUE
# =============================================================================

st.divider()

if not queue:
    # Empty queue — good news!
    if reviewed_today > 0:
        st.success(
            "✅ All done for today! You've reviewed everything that needed attention. "
            "Check back tomorrow."
        )
    else:
        st.info(
            "🎉 Nothing to review right now! All your concepts are above the 65% recall threshold. "
            "Keep logging study events to stay on top of your knowledge."
        )
else:
    st.subheader(f"📚 Queue — {len(queue)} concept(s) need review")

    # ── Each concept card ─────────────────────────────────────────────────────
    for idx, item in enumerate(queue):
        concept_id      = item["concept_id"]
        concept_name    = item["concept_name"]
        subject         = item["subject"]
        difficulty      = item["difficulty"]
        recall          = item["recall_score"]
        urgency         = item["urgency_score"]
        urgency_level   = item["urgency_level"]
        priority        = item["priority"]
        hours_since     = item["hours_since"]
        suggested       = item["suggested_content"]
        est_time        = item["estimated_time_min"]
        features        = item.get("features", {})

        # ── Urgency badge color ───────────────────────────────────────────────
        emoji = get_urgency_emoji(urgency_level)

        # ── Time since last review as readable string ─────────────────────────
        if hours_since < 1:
            time_str = "< 1 hour ago"
        elif hours_since < 24:
            time_str = f"{int(hours_since)}h ago"
        elif hours_since < 48:
            time_str = "Yesterday"
        else:
            time_str = f"{int(hours_since // 24)} days ago"

        # ── Difficulty stars ──────────────────────────────────────────────────
        difficulty_stars = "⭐" * difficulty

        # ── Card container ────────────────────────────────────────────────────
        with st.container(border=True):

            # Top row: urgency badge + concept name + recall %
            header_col, recall_col = st.columns([3, 1])

            with header_col:
                st.markdown(
                    f"### {emoji} {concept_name}"
                )
                st.caption(f"Subject: {subject}  •  Difficulty: {difficulty_stars}")

            with recall_col:
                # Recall % displayed prominently
                if recall < 40:
                    st.error(f"**Recall: {recall:.1f}%**")
                elif recall < 65:
                    st.warning(f"**Recall: {recall:.1f}%**")
                else:
                    st.success(f"**Recall: {recall:.1f}%**")

            # Detail row: urgency score + last reviewed + suggested method + time
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.caption(f"🚨 Urgency: **{urgency:.1f}** ({urgency_level})")
            with d2:
                st.caption(f"🕐 Last reviewed: **{time_str}**")
            with d3:
                st.caption(f"📖 Suggested: **{suggested}**")
            with d4:
                st.caption(f"⏱ Est. time: **{est_time} min**")

            # ── Action buttons row ────────────────────────────────────────────
            btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([2, 1, 1, 2])

            with btn_col1:
                # Mark as Reviewed button
                # key must be unique per concept to avoid Streamlit ID conflicts
                if st.button(
                    "✅ Mark as Reviewed",
                    key=f"review_{concept_id}_{idx}",
                    type="primary",
                    use_container_width=True
                ):
                    # Store which concept we're rating in session_state
                    # so the rating slider appears below
                    st.session_state[f"rating_{concept_id}"] = True

            with btn_col2:
                if st.button(
                    "😴 Snooze 1 day",
                    key=f"snooze1_{concept_id}_{idx}",
                    use_container_width=True
                ):
                    success = snooze_concept(user_id, concept_id, days=1)
                    if success:
                        st.success(f"'{concept_name}' snoozed for 1 day.")
                        st.rerun()

            with btn_col3:
                if st.button(
                    "😴 Snooze 3 days",
                    key=f"snooze3_{concept_id}_{idx}",
                    use_container_width=True
                ):
                    success = snooze_concept(user_id, concept_id, days=3)
                    if success:
                        st.success(f"'{concept_name}' snoozed for 3 days.")
                        st.rerun()

            with btn_col4:
                pass  # empty column for spacing

            # ── Self-rating panel (appears after clicking Mark as Reviewed) ───
            # Uses session_state to track which concept is being rated
            if st.session_state.get(f"rating_{concept_id}", False):
                st.divider()
                st.markdown("**How well did you recall this concept?**")
                st.caption(
                    "0.0 = Completely forgot  •  0.5 = Partially remembered  •  1.0 = Perfect recall"
                )

                rating_col1, rating_col2 = st.columns([3, 1])

                with rating_col1:
                    self_score = st.slider(
                        "Self-rating",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.7,
                        step=0.05,
                        key=f"slider_{concept_id}_{idx}",
                        label_visibility="collapsed"
                    )

                with rating_col2:
                    if st.button(
                        "✔ Confirm",
                        key=f"confirm_{concept_id}_{idx}",
                        type="primary",
                        use_container_width=True
                    ):
                        # Log the event with the student's self-rated score
                        success = mark_as_reviewed(user_id, concept_id, score=self_score)
                        if success:
                            # Clear the rating panel from session state
                            del st.session_state[f"rating_{concept_id}"]
                            st.success(f"✅ '{concept_name}' marked as reviewed (score: {self_score:.2f})")
                            # Rerun to refresh the queue
                            st.rerun()
                        else:
                            st.error("Something went wrong. Try again.")

            # ── Expandable history ────────────────────────────────────────────
            with st.expander(f"📊 Recent history for {concept_name}"):
                recent = get_recent_events_for_concept(
                    user_id, concept_id, limit=5
                )
                if not recent:
                    st.caption("No recent events logged for this concept.")
                else:
                    history_rows = []
                    for e in recent:
                        ts = e.get("timestamp", "")
                        ts_str = (
                            ts.strftime("%d %b %Y, %I:%M %p")
                            if hasattr(ts, "strftime") else str(ts)
                        )
                        history_rows.append({
                            "Date":     ts_str,
                            "Type":     e.get("event_type", ""),
                            "Score":    f"{e.get('score', 0):.0%}",
                            "Time":     f"{e.get('response_time_min', '—')} min"
                        })
                    st.dataframe(
                        pd.DataFrame(history_rows),
                        hide_index=True,
                        use_container_width=True
                    )

                # Also show feature breakdown for this concept
                if features:
                    st.markdown("**Feature snapshot:**")
                    fc1, fc2, fc3 = st.columns(3)
                    with fc1:
                        st.caption(f"Total reviews: **{features.get('total_reviews', 0)}**")
                        st.caption(f"Avg score: **{features.get('avg_score', 0):.0%}**")
                    with fc2:
                        st.caption(f"Last score: **{features.get('last_score', 0):.0%}**")
                        st.caption(f"Success streak: **{features.get('success_streak', 0)}/3**")
                    with fc3:
                        h = features.get("hours_since_last", 0)
                        st.caption(f"Hours since last: **{h:.1f}h**")
                        st.caption(f"Avg time: **{features.get('avg_response_time', 0):.0f} min**")

        # Add small space between cards
        st.write("")


# =============================================================================
# SECTION 3 — ALL CONCEPTS URGENCY TABLE (reference)
# =============================================================================

with st.expander("📊 All Concepts Urgency Overview"):
    st.caption(
        "All concepts with computed urgency scores. "
        "Green rows = above review threshold (no action needed). "
        "Red/amber = in your queue above."
    )

    if all_concepts:
        table_rows = []
        for item in all_concepts:
            cid    = item["concept_id"]
            recall = item["recall_score"]
            h      = item["hours_since"]
            diff   = item["difficulty"]
            urg    = item["urgency_score"]
            level  = item["urgency_level"]
            emoji  = get_urgency_emoji(level)

            # Time since label
            if h < 24:
                time_label = f"{h:.0f}h ago"
            else:
                time_label = f"{h/24:.0f} days ago"

            # In queue or not?
            in_queue = any(q["concept_id"] == cid for q in queue)

            table_rows.append({
                "Concept":      item["concept_name"],
                "Recall %":     f"{recall:.1f}%",
                "Urgency":      f"{emoji} {urg:.1f} ({level})",
                "Last Reviewed": time_label,
                "Difficulty":   "⭐" * diff,
                "In Queue":     "✅ Yes" if in_queue else "⬜ No"
            })

        st.dataframe(
            pd.DataFrame(table_rows),
            hide_index=True,
            use_container_width=True
        )


# =============================================================================
# SECTION 4 — SNOOZED CONCEPTS
# =============================================================================

if snoozed_list:
    with st.expander(f"😴 Snoozed Concepts ({len(snoozed_list)})"):
        st.caption("These concepts are hidden from your queue until their snooze expires.")

        for snoozed_item in snoozed_list:
            cid           = snoozed_item.get("concept_id", "")
            snoozed_until = snoozed_item.get("snoozed_until")
            concept_info  = concepts_lookup.get(cid, {})
            concept_name  = concept_info.get("name", cid)

            # Format snooze expiry
            if snoozed_until:
                from backend.scheduler import _make_aware
                su = _make_aware(snoozed_until)
                now = datetime.now(timezone.utc)
                hours_left = (su - now).total_seconds() / 3600
                if hours_left < 24:
                    expiry_str = f"expires in {int(hours_left)}h"
                else:
                    expiry_str = f"expires in {int(hours_left // 24)} days"
            else:
                expiry_str = "unknown"

            snooze_col1, snooze_col2 = st.columns([3, 1])
            with snooze_col1:
                st.caption(f"😴 **{concept_name}** — {expiry_str}")
            with snooze_col2:
                if st.button(
                    "↩ Unsnooze",
                    key=f"unsnooze_{cid}",
                    use_container_width=True
                ):
                    unsnooze_concept(user_id, cid)
                    st.success(f"'{concept_name}' added back to queue.")
                    st.rerun()


# =============================================================================
# SECTION 5 — DAILY GOAL SETTINGS
# =============================================================================

st.divider()

with st.expander("⚙️ Queue Settings"):
    st.subheader("Daily Review Goal")
    st.caption(
        "Set how many concepts you want to review per day. "
        "The progress bar above tracks this."
    )

    goal_col1, goal_col2 = st.columns([2, 1])

    with goal_col1:
        new_goal = st.slider(
            "Concepts per day",
            min_value=1,
            max_value=10,
            value=daily_goal,
            step=1,
            help="How many concepts do you want to review each day?"
        )

    with goal_col2:
        st.write("")   # spacing
        st.write("")
        if st.button("💾 Save Goal", type="primary", use_container_width=True):
            if set_daily_goal(user_id, new_goal):
                st.success(f"Goal updated to {new_goal} concepts/day!")
                st.rerun()

    # Streak info
    st.divider()
    st.subheader("🔥 Study Streak")
    if streak_days == 0:
        st.info(
            "Your streak is at 0. Review a concept today to start your streak!"
        )
    elif streak_days == 1:
        st.success("You're on a 1-day streak! Come back tomorrow to build it.")
    elif streak_days < 7:
        st.success(f"🔥 {streak_days}-day streak! Keep going!")
    else:
        st.success(f"🏆 Amazing! {streak_days}-day streak! You're on fire!")


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

now_str = datetime.now().strftime("%d %b %Y, %I:%M %p")
model_note = "Phase 4 ML model" if os.path.exists(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "recall_model.pkl")
) else "Ebbinghaus formula"

st.caption(
    f"Queue refreshed: {now_str}  •  "
    f"User: `{user_id}`  •  "
    f"Scoring: {model_note}  •  "
    f"Review threshold: 65%"
)