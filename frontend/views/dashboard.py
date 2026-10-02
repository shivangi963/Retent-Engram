"""
frontend/views/dashboard.py
============================
Home screen: recall summary, concept table with recommended actions,
visual overview, and a preview of generated review material.
"""
import math
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.ml.pipeline import compute_scores_for_user
from backend.db import get_event_counts_by_concept, get_last_event_per_concept, get_total_events_count
from backend.scheduler import build_and_get_queue
from rag.generate import get_cached_content, is_ollama_running
from rag.ingest import is_index_available

from frontend.components.theme import (
    html, page_header, section_title, stat_card, pill, card, icon, empty_state,
    recall_tone, concept_icon, TONES, PRIMARY, INK,
)
from frontend.components.data import (
    concepts_by_id, current_user, ago, ago_from_dt, open_study, ACTIONS,
)
from frontend.components.parsers import parse_flashcard, summary_blurb, parse_coding, plain
from frontend.components.charts import build_recall_bar_chart, build_priority_donut, build_forgetting_curve
from frontend.components.system import all_checks

user_id, user_name = current_user()
concepts = concepts_by_id()

# ── system status pill (compact — full detail lives in Settings) ────────────
checks = all_checks()
n_ok = sum(1 for c in checks if c[2])
status_pill = (pill("All systems ready", "green", dot=True) if n_ok == len(checks)
               else pill(f"{n_ok}/{len(checks)} services ready", "amber", dot=True))

page_header("Dashboard", f"Your live knowledge health overview, {user_name}.", right=status_pill)

# ── data ──────────────────────────────────────────────────────────────────
with st.spinner("Computing your latest recall scores…"):
    results = compute_scores_for_user(user_id)
    event_counts = get_event_counts_by_concept(user_id)
    last_events = get_last_event_per_concept(user_id)
    total_events = get_total_events_count(user_id)
    queue_data = build_and_get_queue(user_id, concepts)

if not results:
    empty_state("inbox", "No study events yet",
                "Log your first reading, quiz or coding session to unlock your dashboard.")
    if st.button("Log a study session", icon=":material/edit_note:", type="primary"):
        st.switch_page("views/log_session.py")
    st.stop()

rows = []
for r in results:
    cid = r["concept_id"]
    info = concepts.get(cid, {})
    rows.append({
        "concept_id": cid,
        "concept_name": info.get("name", cid),
        "subject": info.get("subject", "—"),
        "difficulty": info.get("difficulty", 1),
        "recall_score": r["recall_score"],
        "priority": r["priority"],
        "sessions": event_counts.get(cid, 0),
        "last_ts": last_events.get(cid),
        "features": r["features"],
    })
df = pd.DataFrame(rows).sort_values("recall_score", ascending=True).reset_index(drop=True)

avg_recall = df["recall_score"].mean()
weak_topics = int((df["priority"] == "High").sum())
upcoming = len(queue_data["queue"])

# ── summary cards (reference layout) ─────────────────────────────────────────
c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    html(stat_card("Recall Score Summary", f"{avg_recall:.0f}%", "Average recall score", "trending-up", "green"))
with c2:
    html(stat_card("Upcoming Reviews", str(upcoming), "Reviews due today", "calendar", "amber"))
with c3:
    html(stat_card("Weak Topics", str(weak_topics), "Topics need focus", "target", "red"))

# ── concept table ────────────────────────────────────────────────────────────
section_title("Your Concepts", "Sorted by urgency — lowest recall first.")

COLS = [3.1, 1.3, 1.3, 1.7]
with card("tbl"):
    html(f"""<div class="rt-thead-wrap">
      <div style="display:flex;background:{INK if False else '#0B2A5B'};border-radius:14px 14px 0 0;padding:.85rem 1.1rem">
        <div class="rt-th" style="flex:{COLS[0]}">Concept Name</div>
        <div class="rt-th" style="flex:{COLS[1]}">Recall %</div>
        <div class="rt-th" style="flex:{COLS[2]}">Priority</div>
        <div class="rt-th" style="flex:{COLS[3]}">Action</div>
      </div></div>""")

    for _, row in df.iterrows():
        cid = row["concept_id"]
        ic_name, tone = concept_icon(cid)
        fg, bg, bd = TONES[tone]
        rt = recall_tone(row["recall_score"])
        rfg = TONES[rt][0]
        action_label, action_type = ACTIONS[row["priority"]]

        rc = st.columns(COLS, vertical_alignment="center")
        with rc[0]:
            html(f"""<div class="rt-tc">
                <div class="rt-tile" style="background:{bg};color:{fg}">{icon(ic_name, 20)}</div>
                <div><div class="rt-tn">{row['concept_name']}</div>
                     <div class="rt-ts">{row['subject']} · {row['sessions']} session{'s' if row['sessions'] != 1 else ''}</div></div>
              </div>""")
        with rc[1]:
            html(f'<div class="rt-recall" style="color:{rfg}">{row["recall_score"]:.0f}%</div>')
        with rc[2]:
            html(pill(row["priority"], rt, dot=True))
        with rc[3]:
            btn_key = f"act_{ 'high' if row['priority']=='High' else ('medium' if row['priority']=='Medium' else 'low') }_{cid}"
            if action_type is None:
                st.button(action_label, key=btn_key, disabled=True, width="stretch")
            else:
                if st.button(action_label, key=btn_key, width="stretch"):
                    open_study(cid, action_type)

# ── visual overview ──────────────────────────────────────────────────────────
section_title("Visual Overview", icon_name="trending-up")
vc1, vc2 = st.columns([1.6, 1], gap="medium")
with vc1:
    with card("chart_bar"):
        html('<div style="font-weight:700;color:#0F1F3D;margin-bottom:.3rem">Recall by concept</div>')
        st.plotly_chart(build_recall_bar_chart(df[["concept_name", "recall_score"]]),
                         width="stretch", config={"displayModeBar": False})
with vc2:
    with card("chart_donut"):
        html('<div style="font-weight:700;color:#0F1F3D;margin-bottom:.3rem">Priority breakdown</div>')
        st.plotly_chart(build_priority_donut(df[["priority"]]),
                         width="stretch", config={"displayModeBar": False})

# ── generated review panel ────────────────────────────────────────────────────
section_title("Generated Review Panel", "AI-ready material for your most urgent concept.", "sparkles")

top = df.iloc[0]
top_cid, top_name = top["concept_id"], top["concept_name"]
html(f'<div style="color:#64748B;font-size:.88rem;margin:-.4rem 0 .9rem">Showing material for '
     f'<b style="color:#0F1F3D">{top_name}</b> — currently your lowest recall concept.</div>')

PANEL = [
    ("summary", "Explanation", "file-text", "blue"),
    ("flashcard", "Flashcard", "layers", "purple"),
    ("coding_task", "Coding Task", "code", "teal"),
]
ready = is_ollama_running() and is_index_available()

for ctype, label, ic_name, tone in PANEL:
    fg, bg, bd = TONES[tone]
    cached = get_cached_content(user_id, top_cid, ctype)
    with card(f"gen_{ctype}"):
        gc1, gc2 = st.columns([0.09, 0.91], gap="small")
        with gc1:
            html(f'<div style="width:44px;height:44px;border-radius:12px;background:{fg};'
                 f'display:grid;place-items:center;color:#fff">{icon(ic_name, 21)}</div>')
        with gc2:
            html(f'<div class="rt-gen-h" style="color:{fg}">{label}</div>')
            if cached:
                content = cached["content"]
                if ctype == "flashcard":
                    q, a = parse_flashcard(content)
                    html(f'<div class="rt-gen-p"><b>Q:</b> {plain(q)}<br><b>A:</b> {plain(a)}</div>')
                elif ctype == "summary":
                    html(f'<div class="rt-gen-p">{summary_blurb(content)}</div>')
                else:
                    sec = parse_coding(content)
                    html(f'<div class="rt-gen-p">{plain(sec.get("PROBLEM", content))[:220]}</div>')
                b1, b2 = st.columns([1, 5])
                with b1:
                    if st.button("Open", key=f"open_{ctype}", width="stretch"):
                        open_study(top_cid, ctype)
            else:
                html(f'<div class="rt-gen-p" style="color:#94A3B8">Not generated yet'
                     f'{"" if ready else " — start Ollama and build the knowledge base first"}.</div>')
                if st.button(f"Generate {label.lower()}", key=f"gen_{ctype}", disabled=not ready):
                    open_study(top_cid, ctype)

# ── concept deep dive ─────────────────────────────────────────────────────────
section_title("Concept Deep Dive", "Predicted forgetting curve for a chosen concept.")

dd1, dd2 = st.columns([1, 3])
with dd1:
    selected_name = st.selectbox("Concept", options=df["concept_name"].tolist(), label_visibility="collapsed")
selected = df[df["concept_name"] == selected_name].iloc[0]
matching = next((r for r in results if r["concept_id"] == selected["concept_id"]), None)
features = matching["features"] if matching else {}
reviews = features.get("total_reviews", 1)
streak = features.get("success_streak", 0)
stability = 24 * math.log1p(reviews) * (1 + 0.3 * streak)

dd_l, dd_r = st.columns([1.7, 1], gap="medium")
with dd_l:
    with card("curve"):
        st.plotly_chart(
            build_forgetting_curve(selected_name, selected["recall_score"], stability),
            width="stretch", config={"displayModeBar": False},
        )
with dd_r:
    with card("curve_stats"):
        rt = recall_tone(selected["recall_score"])
        html(f"""<div style="font-weight:700;font-size:1.02rem;color:#0F1F3D">{selected_name}</div>
          <div style="color:#64748B;font-size:.85rem;margin-bottom:.8rem">{selected['subject']} ·
            {'⭐' * int(selected['difficulty'])}</div>""")
        h = features.get("hours_since_last", 0)
        time_str = "< 1 hour" if h < 1 else (f"{h:.1f} hours" if h < 24 else f"{h/24:.1f} days")
        for label, val in [
            ("Time since last review", time_str),
            ("Total sessions", features.get("total_reviews", 0)),
            ("Average score", f"{features.get('avg_score', 0):.0%}"),
            ("Recent streak", f"{features.get('success_streak', 0)}/3"),
        ]:
            html(f'<div style="display:flex;justify-content:space-between;padding:.35rem 0;'
                 f'border-bottom:1px solid #EEF2F8;font-size:.88rem">'
                 f'<span style="color:#64748B">{label}</span><span style="font-weight:700;color:#0F1F3D">{val}</span></div>')
        note = {"High": ("Review this today — recall is critically low.", "red"),
                "Medium": ("Review soon — recall is declining.", "amber"),
                "Low": ("Well remembered — no action needed.", "green")}[selected["priority"]]
        html(f'<div style="margin-top:.8rem">{pill(note[0], note[1])}</div>')

# ── dependency risk (only if something is actually at risk) ─────────────────
try:
    from backend.dependency_graph import get_concepts_at_risk
    urgent_ids = [r["concept_id"] for r in results if r.get("priority") == "High"]
    at_risk = get_concepts_at_risk(urgent_ids) if urgent_ids else {}
except Exception:
    at_risk = {}

if at_risk:
    section_title("Foundational Risk", "Concepts that may be affected by a weak prerequisite.", "alert")
    with card("risk"):
        for cid, info in at_risk.items():
            name = concepts.get(cid, {}).get("name", cid)
            causes = [concepts.get(c, {}).get("name", c) for c in info["at_risk_because"]]
            html(f'<div style="padding:.4rem 0;font-size:.9rem;color:#334155">'
                 f'<b style="color:#0F1F3D">{name}</b> depends on weak: {", ".join(causes)}</div>')

st.caption(f"Last refreshed {datetime.now().strftime('%d %b %Y, %I:%M %p')} · "
           f"{total_events} sessions logged · Model: {results[0].get('model_used', 'formula')}")
