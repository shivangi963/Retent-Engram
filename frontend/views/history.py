"""
frontend/views/history.py
===========================
Full event history: filters, score trend, breakdowns, timeline, and export.
"""
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.db import get_collection

from frontend.components.theme import html, page_header, section_title, card, mini_stat, empty_state, TONES
from frontend.components.data import concepts_by_id, current_user
from frontend.components.charts import build_activity_heatmap, INK, MUTED, GRID, FONT

user_id, user_name = current_user()
concepts = concepts_by_id()
id_to_name = {cid: c["name"] for cid, c in concepts.items()}

page_header("History", "Your complete study timeline — filter, analyse, and export.")


@st.cache_data(ttl=30)
def load_events(uid: str) -> pd.DataFrame:
    col = get_collection("events")
    events = list(col.find({"user_id": uid}, {"_id": 0}).sort("timestamp", -1))
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["concept_name"] = df["concept_id"].map(id_to_name).fillna(df["concept_id"])
    if "response_time_sec" in df.columns:
        df["response_time_min"] = df.get("response_time_min", df["response_time_sec"] / 60)
    df["response_time_min"] = df.get("response_time_min", 20)
    df["response_time_min"] = pd.to_numeric(df["response_time_min"], errors="coerce").fillna(20).round(1)
    df["score_pct"] = (df["score"] * 100).round(1)
    df["date"] = df["timestamp"].dt.date
    return df


df_all = load_events(user_id)

if df_all.empty:
    empty_state("inbox", "No events found", "Log some study sessions first.")
    st.stop()

# ── filters ───────────────────────────────────────────────────────────────────
with card("filters"):
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1], vertical_alignment="bottom")
    with f1:
        opt = ["All concepts"] + sorted(df_all["concept_name"].unique().tolist())
        filter_concept = st.selectbox("Concept", opt)
    with f2:
        opt = ["All types"] + sorted(df_all["event_type"].unique().tolist())
        filter_type = st.selectbox("Event type", opt)
    with f3:
        min_date, max_date = df_all["timestamp"].dt.date.min(), df_all["timestamp"].dt.date.max()
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    with f4:
        clear_filters = st.button("Reset", width="stretch")

df = df_all.copy()
if filter_concept != "All concepts":
    df = df[df["concept_name"] == filter_concept]
if filter_type != "All types":
    df = df[df["event_type"] == filter_type]
if len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)]
if clear_filters:
    st.rerun()

st.caption(f"Showing {len(df)} of {len(df_all)} total events")

# ── summary ───────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5, gap="medium")
with m1:
    html(mini_stat("Events", str(len(df)), "layers", "blue"))
with m2:
    html(mini_stat("Concepts", str(df["concept_id"].nunique()), "book", "purple"))
with m3:
    html(mini_stat("Avg Score", f"{df['score'].mean()*100:.0f}%", "target", "green"))
with m4:
    total_time = df["response_time_min"].sum()
    label = f"{total_time:.0f} min" if total_time < 60 else f"{total_time/60:.1f} hr"
    html(mini_stat("Study Time", label, "clock", "amber"))
with m5:
    today = datetime.now(timezone.utc).date()
    html(mini_stat("Today", str(len(df[df["timestamp"].dt.date == today])), "calendar", "teal"))

# ── score trend ────────────────────────────────────────────────────────────────
section_title("Score Trend Over Time")
if len(df) >= 2:
    available = sorted(df["concept_name"].unique().tolist())
    selected = st.multiselect("Concepts to display", options=available, default=available[: min(4, len(available))],
                               label_visibility="collapsed")
    if selected:
        with card("trend"):
            df_trend = df[df["concept_name"].isin(selected)].sort_values("timestamp")
            fig = go.Figure()
            palette = ["#2563EB", "#16A34A", "#F59E0B", "#7C3AED", "#0E9F8E", "#EF4444"]
            for i, concept in enumerate(selected):
                cdf = df_trend[df_trend["concept_name"] == concept]
                if cdf.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=cdf["timestamp"], y=cdf["score_pct"], mode="lines+markers", name=concept,
                    line=dict(width=2.4, color=palette[i % len(palette)]), marker=dict(size=6),
                    hovertemplate=f"<b>{concept}</b><br>%{{x|%d %b %Y}}<br>%{{y:.1f}}%<extra></extra>",
                ))
            fig.add_hline(y=65, line_dash="dot", line_color="#CBD5E1", line_width=1.3,
                          annotation_text="Review threshold", annotation_font=dict(color=MUTED, size=10, family=FONT))
            fig.update_layout(
                xaxis=dict(title=None, gridcolor=GRID, tickfont=dict(color=MUTED, size=10, family=FONT)),
                yaxis=dict(title=None, range=[0, 105], ticksuffix="%", gridcolor=GRID,
                            tickfont=dict(color=MUTED, size=10, family=FONT)),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11, family=FONT)),
                margin=dict(l=6, r=20, t=30, b=6), font=dict(family=FONT),
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    else:
        st.caption("Select at least one concept to see the trend.")
else:
    st.caption("Log at least 2 events to see a trend.")

# ── breakdowns ─────────────────────────────────────────────────────────────────
b1, b2 = st.columns(2, gap="medium")
with b1:
    with card("by_type"):
        html('<div style="font-weight:700;color:#0F1F3D;margin-bottom:.3rem">Events by type</div>')
        tc = df["event_type"].value_counts().reset_index()
        tc.columns = ["Event Type", "Count"]
        colors = {"reading": "#2563EB", "quiz": "#16A34A", "coding": "#F59E0B", "review": "#7C3AED"}
        fig_type = px.bar(tc, x="Event Type", y="Count", color="Event Type", color_discrete_map=colors)
        fig_type.update_traces(marker=dict(cornerradius=6))
        fig_type.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=260,
                                showlegend=False, margin=dict(l=6, r=6, t=10, b=6), font=dict(family=FONT),
                                xaxis=dict(title=None, tickfont=dict(color=MUTED, family=FONT)),
                                yaxis=dict(title=None, gridcolor=GRID, tickfont=dict(color=MUTED, family=FONT)))
        st.plotly_chart(fig_type, width="stretch", config={"displayModeBar": False})
with b2:
    with card("by_concept"):
        html('<div style="font-weight:700;color:#0F1F3D;margin-bottom:.3rem">Events by concept</div>')
        cc = df["concept_name"].value_counts().head(8).reset_index()
        cc.columns = ["Concept", "Count"]
        fig_c = px.bar(cc, x="Count", y="Concept", orientation="h", color="Count",
                        color_continuous_scale=["#C7D9FB", "#2563EB"])
        fig_c.update_traces(marker=dict(cornerradius=6))
        fig_c.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=260,
                             showlegend=False, coloraxis_showscale=False, margin=dict(l=6, r=6, t=10, b=6),
                             font=dict(family=FONT),
                             xaxis=dict(title=None, gridcolor=GRID, tickfont=dict(color=MUTED, family=FONT)),
                             yaxis=dict(title=None, tickfont=dict(color=INK, family=FONT)))
        st.plotly_chart(fig_c, width="stretch", config={"displayModeBar": False})

# ── activity pattern ───────────────────────────────────────────────────────────
with st.expander("Study activity pattern — when do you study?"):
    events_col = get_collection("events")
    raw_events = list(events_col.find({"user_id": user_id}, {"_id": 0}))
    st.plotly_chart(build_activity_heatmap(raw_events), width="stretch", config={"displayModeBar": False})

# ── timeline ────────────────────────────────────────────────────────────────────
section_title("Event Timeline")
display_rows = []
for _, row in df.iterrows():
    ts = row["timestamp"]
    ts_str = ts.strftime("%d %b %Y, %I:%M %p") if hasattr(ts, "strftime") else str(ts)
    sv = row.get("score", 0)
    tone = "green" if sv >= 0.8 else ("amber" if sv >= 0.6 else "red")
    display_rows.append({
        "When": ts_str, "Concept": row.get("concept_name", row.get("concept_id", "")),
        "Type": row.get("event_type", ""), "Score": f"{sv*100:.0f}%",
        "Time (min)": f"{row.get('response_time_min', 0):.0f}", "Hints": int(row.get("hints_used", 0)),
    })
with card("timeline"):
    st.dataframe(pd.DataFrame(display_rows), width="stretch", hide_index=True, height=380)

# ── export ────────────────────────────────────────────────────────────────────
section_title("Export Data")
e1, e2 = st.columns(2, gap="medium")
with e1:
    with card("export_events"):
        html('<div style="font-weight:700;color:#0F1F3D">Event log</div>'
             '<div style="color:#64748B;font-size:.85rem;margin:.2rem 0 .8rem">Download your filtered history as CSV.</div>')
        export_df = df[["timestamp", "concept_name", "event_type", "score", "response_time_min", "hints_used"]].copy()
        export_df.columns = ["Timestamp", "Concept", "Event Type", "Score (0-1)", "Time (min)", "Hints Used"]
        export_df["Timestamp"] = export_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.download_button("Download events CSV", data=export_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"retent_engram_events_{user_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv", width="stretch", type="primary", icon=":material/download:")
with e2:
    with card("export_scores"):
        html('<div style="font-weight:700;color:#0F1F3D">Recall scores</div>'
             '<div style="color:#64748B;font-size:.85rem;margin:.2rem 0 .8rem">Download your current recall report as CSV.</div>')
        try:
            scores_docs = list(get_collection("recall_scores").find(
                {"user_id": user_id}, {"_id": 0, "user_id": 0, "features": 0}))
            if scores_docs:
                scores_df = pd.DataFrame(scores_docs)
                scores_df["concept_name"] = scores_df["concept_id"].map(id_to_name).fillna(scores_df["concept_id"])
                if "last_computed" in scores_df.columns:
                    scores_df["last_computed"] = pd.to_datetime(scores_df["last_computed"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
                cols = [c for c in ["concept_name", "concept_id", "recall_score", "priority", "urgency_score",
                                     "urgency_level", "last_computed", "model_used"] if c in scores_df.columns]
                st.download_button("Download scores CSV", data=scores_df[cols].to_csv(index=False).encode("utf-8"),
                                    file_name=f"retent_engram_recall_{user_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv", width="stretch", icon=":material/download:")
            else:
                st.caption("No recall scores yet — visit the Dashboard first.")
        except Exception as e:
            st.error(f"Could not load recall scores: {e}")

st.caption(f"{len(df_all)} total events · {len(df)} shown · Last updated {datetime.now().strftime('%H:%M:%S')}")
