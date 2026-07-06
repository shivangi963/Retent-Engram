import streamlit as st
import pandas as pd
import json
import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.ml.pipeline import compute_scores_for_user
from backend.db import (
    get_recall_scores,
    get_event_counts_by_concept,
    get_last_event_per_concept,
    get_total_events_count,
    get_collection
)
from frontend.components.charts import (
    build_recall_bar_chart,
    build_priority_donut,
    build_forgetting_curve,
    build_activity_heatmap
)

st.set_page_config(
    page_title="Dashboard — Retent Engram",
    layout="wide"       # wide layout gives more space for charts side by side
)

st.title("Recall Dashboard")
st.caption("Your live knowledge health overview. Scores refresh every time you open this page.")


CONCEPTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "concepts.json"
)

@st.cache_data
def load_concepts() -> list:
    
    with open(CONCEPTS_PATH, "r") as f:
        return json.load(f)

concepts = load_concepts()

# Build lookup dictionaries for fast access by concept_id
id_to_name    = {c["concept_id"]: c["name"]       for c in concepts}
id_to_subject = {c["concept_id"]: c["subject"]    for c in concepts}
id_to_diff    = {c["concept_id"]: c["difficulty"] for c in concepts}

# Get all unique subjects for the filter dropdown
all_subjects = sorted(set(c["subject"] for c in concepts))



# Check if user logged in from the Event Logger page
# session_state persists across pages within the same browser session
if "user_id" not in st.session_state or not st.session_state.user_id:
    st.warning("Please go to **Log Event** first and set your user ID.")
    st.info("Your user ID needs to be set before scores can be computed.")
    st.stop()   # st.stop() halts execution — nothing below this runs

user_id = st.session_state.user_id
user_name = st.session_state.get("user_name", user_id)

st.subheader(f"Hello, {user_name}")


# st.spinner() shows a loading indicator while the indented code runs.
# This is important because compute_scores_for_user() queries MongoDB
# and runs math — might take 1–2 seconds with many events.
with st.spinner("Computing your latest recall scores..."):
    # Run the full Phase 2 pipeline for this user
    # This SAVES updated scores to MongoDB AND returns them
    pipeline_results = compute_scores_for_user(user_id)

    # Also fetch supporting data for the table
    event_counts  = get_event_counts_by_concept(user_id)   # { concept_id: count }
    last_events   = get_last_event_per_concept(user_id)    # { concept_id: datetime }
    total_events  = get_total_events_count(user_id)        # int

    # Fetch raw events for heatmap
    events_col = get_collection("events")
    raw_events = list(events_col.find({"user_id": user_id}, {"_id": 0}))

# If no pipeline results, the student hasn't logged any events yet
if not pipeline_results:
    st.info("No study events found. Go to **Log Event** and log your first session!")
    st.stop()


rows = []
for result in pipeline_results:
    cid          = result["concept_id"]
    recall       = result["recall_score"]
    features     = result["features"]
    last_ts      = last_events.get(cid)
    reviews      = event_counts.get(cid, 0)

    # Compute how long ago the last review was, as a readable string
    if last_ts:
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        hours_ago = (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600
        if hours_ago < 1:
            last_reviewed_str = "Just now"
        elif hours_ago < 24:
            last_reviewed_str = f"{int(hours_ago)}h ago"
        elif hours_ago < 48:
            last_reviewed_str = "Yesterday"
        else:
            last_reviewed_str = f"{int(hours_ago // 24)} days ago"
    else:
        last_reviewed_str = "Never"

    rows.append({
        "concept_id":     cid,
        "concept_name":   id_to_name.get(cid, cid),
        "subject":        id_to_subject.get(cid, "Unknown"),
        "difficulty":     id_to_diff.get(cid, 1),
        "recall_score":   recall,
        "priority":       result["priority"],
        "sessions":       reviews,
        "last_reviewed":  last_reviewed_str,
        "stability_hrs":  features.get("stability_hours",
                          24 * (1 + 0.693 * features.get("total_reviews", 1))),
        "hours_since":    features.get("hours_since_last", 0),
        "avg_score":      features.get("avg_score", 0),
        "last_score":     features.get("last_score", 0),
        "streak":         features.get("success_streak", 0),
    })

df = pd.DataFrame(rows)



st.divider()
st.subheader("Summary")

# Create 4 equal columns for the 4 metric cards
col1, col2, col3, col4 = st.columns(4)

# Card 1: Average Recall 
avg_recall = df["recall_score"].mean()
# Color the number based on where it falls
if avg_recall < 40:
    avg_color = "🔴"
elif avg_recall < 65:
    avg_color = "🟡"
else:
    avg_color = "🟢"

with col1:
    st.metric(
        label="Average Recall",
        value=f"{avg_color} {avg_recall:.1f}%",
        help="Average recall score across all your concepts right now."
    )

#  Card 2: Concepts Needing Review 
# Count concepts where recall < 65% (below medium threshold)
needs_review = len(df[df["recall_score"] < 65])
with col2:
    st.metric(
        label="Need Review",
        value=f"{needs_review}",
        help="Concepts with recall below 65% that should be reviewed soon."
    )

#  Card 3: Total Sessions Logged 
with col3:
    st.metric(
        label="Sessions Logged",
        value=f"{total_events}",
        help="Total number of study sessions you have logged across all concepts."
    )

#  Card 4: Total Concepts Tracked
with col4:
    st.metric(
        label="Concepts Tracked",
        value=f"{len(df)}",
        help="Number of unique concepts you have studied at least once."
    )


st.divider()
st.subheader("Visual Overview")

# Two columns for charts: bar chart on left (wider), donut on right (narrower)
chart_col1, chart_col2 = st.columns([2, 1])

with chart_col1:
    # Bar chart: recall per concept (all concepts, no filter yet)
    fig_bar = build_recall_bar_chart(df[["concept_name", "recall_score"]])
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_col2:
    # Donut: priority breakdown
    fig_donut = build_priority_donut(df[["priority"]])
    st.plotly_chart(fig_donut, use_container_width=True)



st.divider()
st.subheader("Concept Detail Table")

# Filters row 
filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])

with filter_col1:
    # Subject filter — dropdown to show only one subject
    selected_subject = st.selectbox(
        "Filter by subject",
        options=["All Subjects"] + all_subjects
    )

with filter_col2:
    # Priority filter — show only High/Medium/Low or all
    selected_priority = st.selectbox(
        "Filter by priority",
        options=["All", "High", "Medium", "Low"]
    )

with filter_col3:
    # Sort control
    sort_by = st.selectbox(
        "Sort by",
        options=["Recall (low→high)", "Recall (high→low)", "Concept Name"]
    )

# Apply filters to the DataFrame
filtered_df = df.copy()

if selected_subject != "All Subjects":
    filtered_df = filtered_df[filtered_df["subject"] == selected_subject]

if selected_priority != "All":
    filtered_df = filtered_df[filtered_df["priority"] == selected_priority]

# Apply sort
if sort_by == "Recall (low→high)":
    filtered_df = filtered_df.sort_values("recall_score", ascending=True)
elif sort_by == "Recall (high→low)":
    filtered_df = filtered_df.sort_values("recall_score", ascending=False)
else:
    filtered_df = filtered_df.sort_values("concept_name", ascending=True)

# Show how many results are visible after filtering
st.caption(f"Showing {len(filtered_df)} of {len(df)} concepts")

#  Build display table
# We build a separate display DataFrame — cleaner column names + formatted values

def priority_badge(p: str) -> str:
    """Converts priority string to emoji badge for the table."""
    return {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}.get(p, p)

def difficulty_stars(d: int) -> str:
    """Converts difficulty int (1–5) to star string."""
    return "⭐" * d

display_rows = []
for _, row in filtered_df.iterrows():
    display_rows.append({
        "Concept":          row["concept_name"],
        "Subject":          row["subject"],
        "Difficulty":       difficulty_stars(row["difficulty"]),
        "Recall %":         f"{row['recall_score']:.1f}%",
        "Priority":         priority_badge(row["priority"]),
        "Sessions":         int(row["sessions"]),
        "Last Reviewed":    row["last_reviewed"],
        "Avg Score":        f"{row['avg_score']:.0%}",
        "Urgency":          f"{r.get('urgency_score', 0):.1f}" if "urgency_score" in r else "—",
    })

display_df = pd.DataFrame(display_rows)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    # Column width configuration
    column_config={
        "Concept":       st.column_config.TextColumn(width="large"),
        "Recall %":      st.column_config.TextColumn(width="small"),
        "Priority":      st.column_config.TextColumn(width="medium"),
        "Sessions":      st.column_config.NumberColumn(width="small"),
        "Last Reviewed": st.column_config.TextColumn(width="medium"),
    }
)



st.divider()
st.subheader(" Concept Deep Dive")
st.caption("Select a concept to see its forgetting curve and detailed features.")

# Dropdown to pick one concept for the deep dive
selected_concept_name = st.selectbox(
    "Select a concept",
    options=df["concept_name"].tolist(),
    index=0   # default to index 0 = most urgent (lowest recall, sorted first)
)

# Find the row for the selected concept
selected_row = df[df["concept_name"] == selected_concept_name].iloc[0]

# Layout: forgetting curve on left, feature stats on right
deep_col1, deep_col2 = st.columns([3, 2])

with deep_col1:
    # Forgetting curve — shows predicted decay over 14 days
    # We need stability_hours for the formula. Re-derive it from features stored
    # in pipeline_results (or compute from total_reviews + streak)
    matching = next(
        (r for r in pipeline_results if r["concept_id"] == selected_row["concept_id"]),
        None
    )
    if matching:
        features = matching["features"]
        import math
        reviews = features.get("total_reviews", 1)
        streak  = features.get("success_streak", 0)
        stability = 24 * math.log1p(reviews) * (1 + 0.3 * streak)
    else:
        stability = 24.0

    fig_curve = build_forgetting_curve(
        concept_name=selected_concept_name,
        current_recall=selected_row["recall_score"],
        stability_hours=stability
    )
    st.plotly_chart(fig_curve, use_container_width=True)

with deep_col2:
    # Feature stats panel
    st.markdown(f"### {selected_concept_name}")
    st.markdown(f"**Subject:** {selected_row['subject']}")
    st.markdown(f"**Difficulty:** {difficulty_stars(selected_row['difficulty'])}")

    st.divider()

    # Show the 6 features as metrics
    if matching:
        f = matching["features"]

        # Hours since last review — format nicely
        h = f.get("hours_since_last", 0)
        if h < 1:
            time_str = "< 1 hour"
        elif h < 24:
            time_str = f"{h:.1f} hours"
        else:
            time_str = f"{h/24:.1f} days"

        st.metric("Time Since Last Review", time_str)
        st.metric("Total Sessions",         f.get("total_reviews", 0))
        st.metric("Average Score",          f"{f.get('avg_score', 0):.0%}")
        st.metric("Last Session Score",     f"{f.get('last_score', 0):.0%}")
        st.metric("Recent Streak (of 3)",   f"{f.get('success_streak', 0)}/3")
        st.metric("Avg Session Length",     f"{f.get('avg_response_time', 0):.0f} min")

    # Recommendation based on priority
    st.divider()
    priority = selected_row["priority"]
    if priority == "High":
        st.error("**Urgent** — Review this concept today. Your recall is critically low.")
    elif priority == "Medium":
        st.warning("**Review Soon** — Recall is declining. Study within the next day or two.")
    else:
        st.success("**Well Remembered** — No immediate action needed. Keep it up!")



with st.expander("Study Activity Pattern (when do you study?)"):
    st.caption("Shows which days and times you study most. Darker = more sessions.")
    fig_heat = build_activity_heatmap(raw_events)
    st.plotly_chart(fig_heat, use_container_width=True)

with st.expander("🕸️ Concept Dependency Graph"):
    st.caption(
        "Shows how concepts are connected. "
        "Weaker foundation concepts (red) may affect dependent ones."
    )

    from backend.dependency_graph import (
        build_dependency_graph_figure,
        get_concepts_at_risk
    )

    # Build and show graph
    fig_graph = build_dependency_graph_figure(pipeline_results)
    st.plotly_chart(fig_graph, use_container_width=True)

    # Show at-risk concepts if any
    urgent_ids = [
        r["concept_id"] for r in pipeline_results
        if r.get("priority") == "High"
    ]

    if urgent_ids:
        at_risk = get_concepts_at_risk(urgent_ids)

        if at_risk:
            st.markdown("**⚠️ Concepts at risk due to weak foundations:**")

            for cid, info in at_risk.items():
                causes = [
                    id_to_name.get(c, c)
                    for c in info["at_risk_because"]
                ]
                name = id_to_name.get(cid, cid)

                st.caption(
                    f"• **{name}** — depends on weak: "
                    f"{', '.join(causes)}"
                )
                
st.divider()
st.caption(
    f"Last refreshed: {datetime.now().strftime('%d %b %Y, %I:%M %p')}  •  "
    f"User: `{user_id}`  •  "
    f"Model: Ebbinghaus formula (Phase 4 will upgrade to XGBoost)"
)