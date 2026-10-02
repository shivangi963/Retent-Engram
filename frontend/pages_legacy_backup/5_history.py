"""
frontend/pages/5_history.py
=============================
PHASE 7 — Event History Page

PURPOSE
-------
Shows the student a complete log of everything they have ever studied,
with filters, charts, and CSV export. Useful for:
  - Reviewing past performance
  - Showing the guide/examiner real data during demo
  - Exporting data for the VTU report

SECTIONS:
  1. Filter bar   — concept, event type, date range
  2. Summary row  — total events, unique concepts, avg score
  3. Score trend  — line chart of score over time per concept
  4. Event table  — scrollable timeline of all events
  5. Export       — download events as CSV
"""

import streamlit as st
import pandas as pd
import os
import sys
import json
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
import plotly.express as px
import io

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.db import get_collection

# =============================================================================
# PAGE SETUP
# =============================================================================

st.set_page_config(
    page_title="History — Retent Engram",
    page_icon="📅",
    layout="wide"
)

st.title("📅 Event History")
st.caption("Your complete study timeline — filter, analyse, and export.")

# =============================================================================
# USER CHECK
# =============================================================================

if "user_id" not in st.session_state or not st.session_state.user_id:
    st.warning("⚠️ Set your user ID on the Home page first.")
    st.stop()

user_id   = st.session_state.user_id
user_name = st.session_state.get("user_name", user_id)

# =============================================================================
# LOAD CONCEPTS FOR NAME LOOKUP
# =============================================================================

CONCEPTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "concepts.json"
)

@st.cache_data
def load_concepts_map():
    with open(CONCEPTS_PATH) as f:
        data = json.load(f)
    return {c["concept_id"]: c["name"] for c in data}

id_to_name = load_concepts_map()

# =============================================================================
# LOAD ALL EVENTS FROM MONGODB
# =============================================================================

@st.cache_data(ttl=30)
def load_events(uid: str) -> pd.DataFrame:
    """
    Fetches all events for user from MongoDB and returns as DataFrame.
    Cached for 30 seconds to avoid repeated queries during filter changes.
    """
    col = get_collection("events")
    events = list(
        col.find({"user_id": uid}, {"_id": 0})
           .sort("timestamp", -1)
    )
    if not events:
        return pd.DataFrame()

    df = pd.DataFrame(events)

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Map concept_id to display name
    df["concept_name"] = df["concept_id"].map(id_to_name).fillna(df["concept_id"])

    # Unify response_time to minutes
    if "response_time_sec" in df.columns and "response_time_min" not in df.columns:
        df["response_time_min"] = df["response_time_sec"] / 60
    elif "response_time_sec" in df.columns and "response_time_min" in df.columns:
        # Fill missing response_time_min from response_time_sec
        mask = df["response_time_min"].isna()
        df.loc[mask, "response_time_min"] = df.loc[mask, "response_time_sec"] / 60

    df["response_time_min"] = df.get("response_time_min", 20).fillna(20).round(1)

    # Readable score percentage
    df["score_pct"] = (df["score"] * 100).round(1)

    # Date only column for grouping
    df["date"] = df["timestamp"].dt.date

    return df

df_all = load_events(user_id)

if df_all.empty:
    st.info("📝 No events found. Log some study sessions first!")
    st.stop()

# =============================================================================
# FILTER BAR
# =============================================================================

st.subheader("🔍 Filters")

f1, f2, f3, f4 = st.columns([2, 2, 2, 1])

with f1:
    all_concepts = ["All Concepts"] + sorted(df_all["concept_name"].unique().tolist())
    filter_concept = st.selectbox("Concept", all_concepts)

with f2:
    all_types = ["All Types"] + sorted(df_all["event_type"].unique().tolist())
    filter_type = st.selectbox("Event Type", all_types)

with f3:
    min_date = df_all["timestamp"].dt.date.min()
    max_date = df_all["timestamp"].dt.date.max()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

with f4:
    st.write("")
    st.write("")
    clear_filters = st.button("🔄 Reset", use_container_width=True)

# Apply filters
df = df_all.copy()

if filter_concept != "All Concepts":
    df = df[df["concept_name"] == filter_concept]

if filter_type != "All Types":
    df = df[df["event_type"] == filter_type]

if len(date_range) == 2:
    start_date, end_date = date_range
    df = df[
        (df["timestamp"].dt.date >= start_date) &
        (df["timestamp"].dt.date <= end_date)
    ]

if clear_filters:
    st.rerun()

st.caption(f"Showing **{len(df)}** of {len(df_all)} total events")

# =============================================================================
# SUMMARY METRIC CARDS
# =============================================================================

st.divider()

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("Total Events", len(df))
with m2:
    st.metric("Unique Concepts", df["concept_id"].nunique())
with m3:
    avg_score = df["score"].mean() * 100
    st.metric("Avg Score", f"{avg_score:.1f}%")
with m4:
    total_time = df["response_time_min"].sum()
    if total_time < 60:
        time_str = f"{total_time:.0f} min"
    else:
        time_str = f"{total_time/60:.1f} hr"
    st.metric("Total Study Time", time_str)
with m5:
    today = datetime.now(timezone.utc).date()
    today_events = len(df[df["timestamp"].dt.date == today])
    st.metric("Today's Events", today_events)

# =============================================================================
# SCORE TREND CHART — line chart per concept over time
# =============================================================================

st.divider()
st.subheader("📈 Score Trend Over Time")
st.caption("Track how your score for each concept has changed over time.")

if len(df) >= 2:
    # Let user pick which concepts to show on the trend
    available_concepts = sorted(df["concept_name"].unique().tolist())
    selected_for_trend = st.multiselect(
        "Select concepts to display",
        options=available_concepts,
        default=available_concepts[:min(4, len(available_concepts))],
        help="Select up to 6 concepts for a readable chart."
    )

    if selected_for_trend:
        df_trend = df[df["concept_name"].isin(selected_for_trend)].copy()
        df_trend = df_trend.sort_values("timestamp")

        fig = go.Figure()

        for concept in selected_for_trend:
            concept_df = df_trend[df_trend["concept_name"] == concept]
            if len(concept_df) < 1:
                continue

            fig.add_trace(go.Scatter(
                x=concept_df["timestamp"],
                y=concept_df["score_pct"],
                mode="lines+markers",
                name=concept,
                line=dict(width=2),
                marker=dict(size=7),
                hovertemplate=(
                    f"<b>{concept}</b><br>"
                    "Date: %{x|%d %b %Y}<br>"
                    "Score: %{y:.1f}%<br>"
                    "<extra></extra>"
                )
            ))

        # Add reference line at 65% (review threshold)
        fig.add_hline(
            y=65,
            line_dash="dash",
            line_color="#F59E0B",
            line_width=1.5,
            annotation_text="Review threshold (65%)",
            annotation_position="right",
            annotation_font_color="#F59E0B"
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Score %",
            yaxis=dict(range=[0, 105], ticksuffix="%"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=20, r=80, t=20, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one concept above to see the trend chart.")
else:
    st.info("Log at least 2 events to see the score trend chart.")

# =============================================================================
# EVENT TYPE BREAKDOWN — bar chart
# =============================================================================

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("📊 Events by Type")
    type_counts = df["event_type"].value_counts().reset_index()
    type_counts.columns = ["Event Type", "Count"]

    fig_type = px.bar(
        type_counts,
        x="Event Type",
        y="Count",
        color="Event Type",
        color_discrete_map={
            "reading": "#6C63FF",
            "quiz":    "#10B981",
            "coding":  "#F59E0B",
            "review":  "#EF4444"
        }
    )
    fig_type.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=40)
    )
    st.plotly_chart(fig_type, use_container_width=True)

with col_chart2:
    st.subheader("📊 Events by Concept")
    concept_counts = df["concept_name"].value_counts().head(8).reset_index()
    concept_counts.columns = ["Concept", "Count"]

    fig_concept = px.bar(
        concept_counts,
        x="Count",
        y="Concept",
        orientation="h",
        color="Count",
        color_continuous_scale="Viridis"
    )
    fig_concept.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=40)
    )
    st.plotly_chart(fig_concept, use_container_width=True)

# =============================================================================
# EVENT TIMELINE TABLE
# =============================================================================

st.divider()
st.subheader("📋 Event Timeline")

# Build display dataframe
display_rows = []
for _, row in df.iterrows():
    ts = row["timestamp"]
    ts_str = ts.strftime("%d %b %Y, %I:%M %p") if hasattr(ts, "strftime") else str(ts)

    score_val = row.get("score", 0)
    if score_val >= 0.8:
        score_display = f"🟢 {score_val*100:.0f}%"
    elif score_val >= 0.6:
        score_display = f"🟡 {score_val*100:.0f}%"
    else:
        score_display = f"🔴 {score_val*100:.0f}%"

    display_rows.append({
        "When":         ts_str,
        "Concept":      row.get("concept_name", row.get("concept_id", "")),
        "Type":         row.get("event_type", ""),
        "Score":        score_display,
        "Time (min)":   f"{row.get('response_time_min', 0):.0f}",
        "Hints":        int(row.get("hints_used", 0))
    })

display_df = pd.DataFrame(display_rows)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    height=400,
    column_config={
        "When":       st.column_config.TextColumn(width="medium"),
        "Concept":    st.column_config.TextColumn(width="large"),
        "Type":       st.column_config.TextColumn(width="small"),
        "Score":      st.column_config.TextColumn(width="small"),
        "Time (min)": st.column_config.TextColumn(width="small"),
        "Hints":      st.column_config.NumberColumn(width="small"),
    }
)

# =============================================================================
# EXPORT FEATURE — download as CSV
# =============================================================================

st.divider()
st.subheader("📥 Export Data")

ex1, ex2 = st.columns(2)

with ex1:
    st.markdown("**Export Events**")
    st.caption("Download your filtered event history as a CSV file.")

    # Build export dataframe (raw data, no emoji formatting)
    export_events_df = df[[
        "timestamp", "concept_name", "event_type",
        "score", "response_time_min", "hints_used"
    ]].copy()
    export_events_df.columns = [
        "Timestamp", "Concept", "Event Type",
        "Score (0-1)", "Time (min)", "Hints Used"
    ]
    export_events_df["Timestamp"] = export_events_df["Timestamp"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    csv_events = export_events_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Events CSV",
        data=csv_events,
        file_name=f"retent_engram_events_{user_id}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary"
    )

with ex2:
    st.markdown("**Export Recall Scores**")
    st.caption("Download your current recall score report as CSV.")

    try:
        scores_col = get_collection("recall_scores")
        scores_docs = list(scores_col.find(
            {"user_id": user_id},
            {"_id": 0, "user_id": 0, "features": 0}
        ))

        if scores_docs:
            scores_df = pd.DataFrame(scores_docs)

            # Map concept_id to name
            scores_df["concept_name"] = scores_df["concept_id"].map(id_to_name).fillna(
                scores_df["concept_id"]
            )

            # Clean up timestamp
            if "last_computed" in scores_df.columns:
                scores_df["last_computed"] = pd.to_datetime(
                    scores_df["last_computed"], utc=True
                ).dt.strftime("%Y-%m-%d %H:%M:%S")

            # Reorder columns
            cols = ["concept_name", "concept_id", "recall_score", "priority",
                    "urgency_score", "urgency_level", "last_computed", "model_used"]
            cols = [c for c in cols if c in scores_df.columns]
            scores_df = scores_df[cols]

            csv_scores = scores_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Recall Scores CSV",
                data=csv_scores,
                file_name=f"retent_engram_recall_{user_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No recall scores yet. Visit the Dashboard first.")

    except Exception as e:
        st.error(f"Could not load recall scores: {e}")

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption(
    f"History for `{user_id}` · "
    f"Total events in DB: {len(df_all)} · "
    f"Filtered: {len(df)} · "
    f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
)