"""
frontend/views/today.py
========================
Daily review queue: what to study right now, sorted by urgency.
"""
import os
import sys
from datetime import datetime, timezone

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.scheduler import (
    build_and_get_queue, mark_as_reviewed, snooze_concept, unsnooze_concept,
    set_daily_goal, _make_aware,
)

from frontend.components.theme import html, page_header, section_title, pill, card, icon, mini_stat, empty_state, TONES
from frontend.components.data import concepts_by_id, current_user, ago, open_study, ACTIONS, flash

user_id, user_name = current_user()
concepts = concepts_by_id()

page_header("Today", f"{user_name}'s personalised study plan — sorted by urgency.")

with st.spinner("Building today's queue…"):
    data = build_and_get_queue(user_id, concepts)

queue = data["queue"]
daily_goal = data["daily_goal"]
reviewed_today = data["reviewed_today"]
streak_days = data["streak_days"]
total_time_min = data["total_time_min"]
snoozed_list = data["snoozed_concepts"]

if data["is_new_day"]:
    st.info("Good morning — yesterday's progress has been reset. Here's today's queue.", icon=":material/wb_sunny:")

# ── summary row ───────────────────────────────────────────────────────────────
time_label = "All done" if total_time_min == 0 else (
    f"~{total_time_min} min" if total_time_min < 60 else f"~{total_time_min // 60}h {total_time_min % 60}m")

m1, m2, m3, m4 = st.columns(4, gap="medium")
with m1:
    html(mini_stat("Daily Goal", f"{daily_goal} concepts", "target", "blue"))
with m2:
    tone = "green" if reviewed_today >= daily_goal else "amber"
    html(mini_stat("Reviewed Today", f"{reviewed_today} / {daily_goal}", "check", tone))
with m3:
    html(mini_stat("Study Streak", f"{streak_days} days", "flame", "amber" if streak_days else "gray"))
with m4:
    html(mini_stat("Est. Time Left", time_label, "clock", "purple"))

if daily_goal > 0:
    frac = min(reviewed_today / daily_goal, 1.0)
    st.progress(frac, text=f"{reviewed_today} of {daily_goal} concepts reviewed today")
    if reviewed_today >= daily_goal:
        st.success("Daily goal reached — nice work! Come back tomorrow to keep your streak going.",
                    icon=":material/celebration:")

# ── queue ──────────────────────────────────────────────────────────────────────
section_title(f"Review Queue — {len(queue)} concept{'s' if len(queue) != 1 else ''}",
              icon_name="layers")

if not queue:
    if reviewed_today > 0:
        empty_state("check", "All done for today", "You've reviewed everything that needed attention.")
    else:
        empty_state("check", "Nothing to review right now", "Every concept is above the 65% recall threshold.")
else:
    for idx, item in enumerate(queue):
        cid = item["concept_id"]
        recall = item["recall_score"]
        urgency_level = item["urgency_level"]
        priority = item["priority"]
        suggested = item["suggested_content"].split(" ", 1)[-1] if " " in item["suggested_content"] else item["suggested_content"]
        est_time = item["estimated_time_min"]
        rt = {"Critical": "red", "High": "red", "Medium": "amber", "Low": "green"}.get(urgency_level, "gray")
        recall_fg = TONES["red" if recall < 40 else ("amber" if recall < 65 else "green")][0]

        with card(f"q_{cid}"):
            top_l, top_r = st.columns([3, 1], vertical_alignment="top")
            with top_l:
                html(f'<div class="rt-q-name">{item["concept_name"]}</div>')
                html(f"""<div class="rt-q-meta">
                    {pill(urgency_level, rt, dot=True)}
                    <span class="rt-chip">{icon('clock', 13)} {ago(item['hours_since'])}</span>
                    <span class="rt-chip">{icon('book', 13)} {suggested}</span>
                    <span class="rt-chip">{icon('target', 13)} ~{est_time} min</span>
                    <span class="rt-chip">{'⭐' * item['difficulty']}</span>
                  </div>""")
            with top_r:
                html(f'<div class="rt-recall-box"><div class="n" style="color:{recall_fg}">{recall:.0f}%</div>'
                     f'<div class="l">recall</div></div>')

            html('<div style="height:.7rem"></div>')
            b1, b2, b3 = st.columns([1.5, 1.3, 1], gap="small")
            with b1:
                _, action_type = ACTIONS.get(priority, (None, "flashcard"))
                if st.button("Study now", key=f"study_{cid}_{idx}", type="primary", width="stretch", icon=":material/auto_stories:"):
                    open_study(cid, action_type or "flashcard")
            with b2:
                if st.button("Mark as reviewed", key=f"markrev_{cid}_{idx}", width="stretch"):
                    st.session_state[f"rating_{cid}"] = True
            with b3:
                with st.popover("Snooze", width="stretch"):
                    if st.button("1 day", key=f"sn1_{cid}_{idx}", width="stretch"):
                        if snooze_concept(user_id, cid, days=1):
                            flash(f"'{item['concept_name']}' snoozed for 1 day.")
                            st.rerun()
                    if st.button("3 days", key=f"sn3_{cid}_{idx}", width="stretch"):
                        if snooze_concept(user_id, cid, days=3):
                            flash(f"'{item['concept_name']}' snoozed for 3 days.")
                            st.rerun()

            if st.session_state.get(f"rating_{cid}", False):
                html('<div style="height:.5rem"></div>')
                with card(f"rate_{cid}"):
                    html('<div style="font-weight:600;color:#0F1F3D;font-size:.9rem;margin-bottom:.2rem">'
                         'How well did you recall this concept?</div>'
                         '<div style="color:#64748B;font-size:.8rem;margin-bottom:.5rem">'
                         '0.0 = completely forgot · 0.5 = partial · 1.0 = perfect recall</div>')
                    rc1, rc2 = st.columns([3, 1])
                    with rc1:
                        self_score = st.slider("Self-rating", 0.0, 1.0, 0.7, 0.05,
                                                key=f"slider_{cid}_{idx}", label_visibility="collapsed")
                    with rc2:
                        if st.button("Confirm", key=f"confirm_{cid}_{idx}", type="primary", width="stretch"):
                            if mark_as_reviewed(user_id, cid, score=self_score):
                                del st.session_state[f"rating_{cid}"]
                                flash(f"'{item['concept_name']}' marked as reviewed.")
                                st.rerun()

        html('<div style="height:.65rem"></div>')

# ── snoozed ─────────────────────────────────────────────────────────────────
if snoozed_list:
    with st.expander(f"Snoozed concepts ({len(snoozed_list)})"):
        for s in snoozed_list:
            cid = s.get("concept_id", "")
            name = concepts.get(cid, {}).get("name", cid)
            until = s.get("snoozed_until")
            if until:
                su = _make_aware(until)
                hrs = (su - datetime.now(timezone.utc)).total_seconds() / 3600
                expiry = f"expires in {int(hrs)}h" if hrs < 24 else f"expires in {int(hrs // 24)}d"
            else:
                expiry = "unknown"
            sc1, sc2 = st.columns([3, 1], vertical_alignment="center")
            with sc1:
                st.caption(f"**{name}** — {expiry}")
            with sc2:
                if st.button("Unsnooze", key=f"unsnooze_{cid}", width="stretch"):
                    unsnooze_concept(user_id, cid)
                    flash(f"'{name}' added back to the queue.")
                    st.rerun()

# ── daily goal ────────────────────────────────────────────────────────────────
with st.expander("Daily goal"):
    g1, g2 = st.columns([3, 1], vertical_alignment="bottom")
    with g1:
        new_goal = st.slider("Concepts per day", 1, 10, daily_goal, 1)
    with g2:
        if st.button("Save", key="save_goal_today", type="primary", width="stretch"):
            if set_daily_goal(user_id, new_goal):
                flash("Daily goal updated.")
                st.rerun()

st.caption(f"Queue refreshed {datetime.now().strftime('%d %b %Y, %I:%M %p')} · Review threshold: 65%")
