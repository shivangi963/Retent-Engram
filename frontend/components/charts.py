import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timezone

# Threshold constants (must match scorer.py)
HIGH_THRESHOLD   = 40.0
MEDIUM_THRESHOLD = 65.0

# Color palette
COLOR_HIGH   = "#EF4444"   # red
COLOR_MEDIUM = "#F59E0B"   # amber
COLOR_LOW    = "#10B981"   # green
COLOR_BG     = "rgba(0,0,0,0)"   # transparent


def get_color_for_score(score: float) -> str:
    if score < HIGH_THRESHOLD:
        return COLOR_HIGH
    elif score < MEDIUM_THRESHOLD:
        return COLOR_MEDIUM
    else:
        return COLOR_LOW


def get_colors_for_scores(scores: list) -> list:
    
    return [get_color_for_score(s) for s in scores]


def build_recall_bar_chart(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        # Return an empty figure with a message if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No recall data yet. Log some study events first.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG)
        return fig

    # Sort by recall_score ascending: lowest recall = most urgent = top of chart
    df_sorted = df.sort_values("recall_score", ascending=True).reset_index(drop=True)

    # Build a color list: one color per bar
    bar_colors = get_colors_for_scores(df_sorted["recall_score"].tolist())

    # Build the label for each bar: "52.1%" shown inside or beside the bar
    bar_labels = [f"{s:.1f}%" for s in df_sorted["recall_score"]]

    # Create the horizontal bar chart
    fig = go.Figure(
        go.Bar(
            # x = the bar lengths (recall scores)
            x=df_sorted["recall_score"],

            # y = the concept names (horizontal axis)
            y=df_sorted["concept_name"],

            # orientation='h' makes it horizontal
            orientation="h",

            # color each bar individually
            marker_color=bar_colors,

            # text shows the score on each bar
            text=bar_labels,
            textposition="auto",    # auto = inside if bar is long, outside if short
            textfont=dict(size=13, color="white"),

            # hover text shown when mouse is over a bar
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Recall: %{x:.1f}%<br>"
                "<extra></extra>"    # removes the default trace name box
            )
        )
    )

    # Add vertical threshold lines
    fig.add_vline(
        x=HIGH_THRESHOLD,
        line_dash="dash",
        line_color=COLOR_HIGH,
        line_width=1.5,
        annotation_text="High threshold (40%)",
        annotation_position="top right",
        annotation_font_color=COLOR_HIGH
    )
    fig.add_vline(
        x=MEDIUM_THRESHOLD,
        line_dash="dash",
        line_color=COLOR_MEDIUM,
        line_width=1.5,
        annotation_text="Review threshold (65%)",
        annotation_position="top right",
        annotation_font_color=COLOR_MEDIUM
    )

    # Chart height: at least 300px, grows with number of concepts
    # Each concept needs about 50px of vertical space
    chart_height = max(300, len(df_sorted) * 55)

    fig.update_layout(
        title=dict(text="Recall Score per Concept", font=dict(size=16)),
        xaxis=dict(
            title="Recall %",
            range=[0, 105],         # extra 5 so 100% labels don't get cut off
            ticksuffix="%",
            gridcolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(
            title="",               # no Y axis title needed, concept names speak for themselves
            automargin=True,        # auto-expand margin to fit long concept names
        ),
        paper_bgcolor=COLOR_BG,     # transparent outer background
        plot_bgcolor=COLOR_BG,      # transparent inner background
        height=chart_height,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )

    return fig



def build_priority_donut(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data yet.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Count concepts per priority level
    # value_counts() returns a Series like: High→3, Medium→2, Low→5
    priority_counts = df["priority"].value_counts()

    # Ensure all three categories appear (even if count is 0)
    labels   = ["High", "Medium", "Low"]
    values   = [priority_counts.get(label, 0) for label in labels]
    colors   = [COLOR_HIGH, COLOR_MEDIUM, COLOR_LOW]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.5,           # 0.5 = 50% hole → donut shape (0 = full pie, 1 = all hole)
            marker_colors=colors,

            # Format: show "High\n3 (60%)" on each slice
            texttemplate="%{label}<br>%{value} (%{percent})",
            textposition="auto",

            hovertemplate=(
                "<b>%{label}</b><br>"
                "Concepts: %{value}<br>"
                "Share: %{percent}<br>"
                "<extra></extra>"
            ),

            # Pull the "High" slice slightly outward for emphasis
            pull=[0.05 if label == "High" else 0 for label in labels]
        )
    )

    fig.update_layout(
        title=dict(text="Priority Breakdown", font=dict(size=16)),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        height=320,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(l=10, r=10, t=50, b=40)
    )

    return fig



def build_forgetting_curve(concept_name: str, current_recall: float, stability_hours: float) -> go.Figure:
    
    import math

    # Generate time points from 0 to 14 days (in hours)
    # We use 6-hour intervals for a smooth curve
    time_points_hours = list(range(0, 337, 6))   # 0, 6, 12, ..., 336

    # Compute predicted recall at each future time point
    predicted_recall = []
    for t in time_points_hours:
        if stability_hours <= 0:
            recall_at_t = 0.0
        else:
            decay = math.exp(-t / stability_hours)
            recall_at_t = current_recall * decay
        # Clamp to 0–100
        predicted_recall.append(round(max(0.0, min(100.0, recall_at_t)), 2))

    # Convert hours to more readable labels for the X axis
    # We'll show: "Now", "6h", "12h", "1d", "2d", ..., "14d"
    x_labels = []
    for h in time_points_hours:
        if h == 0:
            x_labels.append("Now")
        elif h < 24:
            x_labels.append(f"{h}h")
        else:
            days = h // 24
            x_labels.append(f"{days}d")

    # Build the line chart
    fig = go.Figure()

    # Main decay line
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=predicted_recall,
        mode="lines",           # line only, no dots at each point
        line=dict(
            color="#6366F1",    # indigo color for the decay line
            width=2.5
        ),
        fill="tozeroy",         # fill area between line and x-axis
        fillcolor="rgba(99, 102, 241, 0.1)",  # semi-transparent indigo fill
        name=concept_name,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Predicted recall: %{y:.1f}%<br>"
            "<extra></extra>"
        )
    ))

    # Horizontal reference line at MEDIUM threshold (65%)
    fig.add_hline(
        y=MEDIUM_THRESHOLD,
        line_dash="dash",
        line_color=COLOR_MEDIUM,
        line_width=1.5,
        annotation_text="Review zone (65%)",
        annotation_position="right",
        annotation_font_color=COLOR_MEDIUM
    )

    # Horizontal reference line at HIGH threshold (40%)
    fig.add_hline(
        y=HIGH_THRESHOLD,
        line_dash="dash",
        line_color=COLOR_HIGH,
        line_width=1.5,
        annotation_text="Urgent zone (40%)",
        annotation_position="right",
        annotation_font_color=COLOR_HIGH
    )

    fig.update_layout(
        title=dict(
            text=f"Predicted Forgetting Curve — {concept_name}",
            font=dict(size=15)
        ),
        xaxis=dict(
            title="Time from Now",
            # Only show every 4th label to avoid crowding (every 24 hours = 1 day)
            tickmode="array",
            tickvals=x_labels[::4],   # every 4th label
            ticktext=x_labels[::4]
        ),
        yaxis=dict(
            title="Predicted Recall %",
            range=[0, 105],
            ticksuffix="%",
            gridcolor="rgba(255,255,255,0.1)"
        ),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        height=350,
        showlegend=False,
        margin=dict(l=20, r=80, t=50, b=40)
    )

    return fig


def build_activity_heatmap(events: list) -> go.Figure:
    if not events:
        fig = go.Figure()
        fig.add_annotation(
            text="No events yet to show activity pattern.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Day labels for Y axis
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hour_labels = [f"{h:02d}:00" for h in range(24)]

    # Build a 7×24 matrix filled with zeros
    # matrix[day][hour] = count of events
    matrix = [[0] * 24 for _ in range(7)]

    for event in events:
        ts = event.get("timestamp")
        if ts is None:
            continue
        # Make timezone-aware if needed
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        day = ts.weekday()    # 0=Monday, 6=Sunday
        hour = ts.hour        # 0–23
        matrix[day][hour] += 1

    fig = go.Figure(
        go.Heatmap(
            z=matrix,                      # 7×24 matrix of counts
            x=hour_labels,                 # X axis: hours
            y=day_labels,                  # Y axis: days
            colorscale="Blues",            # light=few events, dark=many events
            showscale=True,
            hovertemplate=(
                "<b>%{y}, %{x}</b><br>"
                "Sessions: %{z}<br>"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=dict(text="Study Activity Pattern", font=dict(size=16)),
        xaxis=dict(title="Hour of Day"),
        yaxis=dict(title="Day of Week"),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        height=280,
        margin=dict(l=60, r=20, t=50, b=60)
    )

    return fig