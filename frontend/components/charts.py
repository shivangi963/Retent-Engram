"""
frontend/components/charts.py
==============================
Plotly figures, restyled for the light "Retent Engram" theme.
Same public function names/signatures as before so pages don't change:
  build_recall_bar_chart, build_priority_donut, build_forgetting_curve,
  build_activity_heatmap
"""
import math
from datetime import timezone

import pandas as pd
import plotly.graph_objects as go

HIGH_THRESHOLD = 40.0
MEDIUM_THRESHOLD = 65.0

INK = "#0F1F3D"
MUTED = "#64748B"
GRID = "#EEF2F8"
COLOR_HIGH = "#EF4444"
COLOR_MEDIUM = "#F59E0B"
COLOR_LOW = "#16A34A"
COLOR_BG = "rgba(0,0,0,0)"
FONT = "Inter, 'Segoe UI', sans-serif"


def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                        showarrow=False, font=dict(size=13, color=MUTED))
    fig.update_layout(paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, height=220,
                       xaxis=dict(visible=False), yaxis=dict(visible=False),
                       margin=dict(l=10, r=10, t=10, b=10))
    return fig


def get_color_for_score(score: float) -> str:
    if score < HIGH_THRESHOLD:
        return COLOR_HIGH
    if score < MEDIUM_THRESHOLD:
        return COLOR_MEDIUM
    return COLOR_LOW


def build_recall_bar_chart(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        return _empty("No recall data yet — log a study event first.")

    df_sorted = df.sort_values("recall_score", ascending=True).reset_index(drop=True)
    colors = [get_color_for_score(s) for s in df_sorted["recall_score"]]
    labels = [f"{s:.0f}%" for s in df_sorted["recall_score"]]

    fig = go.Figure(go.Bar(
        x=df_sorted["recall_score"], y=df_sorted["concept_name"], orientation="h",
        marker=dict(color=colors, cornerradius=6),
        text=labels, textposition="outside", textfont=dict(size=12, color=INK, family=FONT),
        hovertemplate="<b>%{y}</b><br>Recall: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=MEDIUM_THRESHOLD, line_dash="dot", line_color="#CBD5E1", line_width=1.4)

    fig.update_layout(
        xaxis=dict(title=None, range=[0, 112], ticksuffix="%", gridcolor=GRID, zeroline=False,
                    tickfont=dict(color=MUTED, size=11, family=FONT)),
        yaxis=dict(title=None, automargin=True, tickfont=dict(color=INK, size=12, family=FONT)),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        height=max(230, len(df_sorted) * 42), showlegend=False,
        margin=dict(l=0, r=10, t=6, b=6), font=dict(family=FONT),
    )
    return fig


def build_priority_donut(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        return _empty("No data yet.")

    counts = df["priority"].value_counts()
    labels = ["High", "Medium", "Low"]
    values = [int(counts.get(l, 0)) for l in labels]
    colors = [COLOR_HIGH, COLOR_MEDIUM, COLOR_LOW]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.68, marker=dict(colors=colors, line=dict(color="#fff", width=2)),
        texttemplate="%{value}", textposition="inside", textfont=dict(size=13, color="#fff", family=FONT),
        hovertemplate="<b>%{label}</b><br>%{value} concepts (%{percent})<extra></extra>",
        sort=False,
    ))
    total = sum(values)
    fig.add_annotation(text=f"<b>{total}</b><br><span style='font-size:11px;color=#64748B'>concepts</span>",
                        x=0.5, y=0.5, showarrow=False, font=dict(size=20, color=INK, family=FONT))
    fig.update_layout(
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, height=230, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.14, x=0.5, xanchor="center",
                    font=dict(size=11, color=MUTED, family=FONT)),
        margin=dict(l=6, r=6, t=6, b=6),
    )
    return fig


def build_forgetting_curve(concept_name: str, current_recall: float, stability_hours: float) -> go.Figure:
    hours = list(range(0, 337, 6))
    recall = []
    for h in hours:
        decay = math.exp(-h / stability_hours) if stability_hours > 0 else 0.0
        recall.append(round(max(0.0, min(100.0, current_recall * decay)), 2))

    labels = ["Now" if h == 0 else (f"{h}h" if h < 24 else f"{h//24}d") for h in hours]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=recall, mode="lines", line=dict(color="#2563EB", width=3),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)", name=concept_name,
        hovertemplate="<b>%{x}</b><br>Predicted recall: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=MEDIUM_THRESHOLD, line_dash="dot", line_color=COLOR_MEDIUM, line_width=1.3,
                  annotation_text="Review zone", annotation_font=dict(color=COLOR_MEDIUM, size=10, family=FONT),
                  annotation_position="right")
    fig.add_hline(y=HIGH_THRESHOLD, line_dash="dot", line_color=COLOR_HIGH, line_width=1.3,
                  annotation_text="Urgent zone", annotation_font=dict(color=COLOR_HIGH, size=10, family=FONT),
                  annotation_position="right")

    fig.update_layout(
        xaxis=dict(title=None, tickmode="array", tickvals=labels[::4], ticktext=labels[::4],
                    tickfont=dict(color=MUTED, size=10, family=FONT), gridcolor=GRID),
        yaxis=dict(title=None, range=[0, 105], ticksuffix="%", gridcolor=GRID,
                    tickfont=dict(color=MUTED, size=10, family=FONT)),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, height=280, showlegend=False,
        margin=dict(l=6, r=60, t=10, b=6), font=dict(family=FONT),
    )
    return fig


def build_activity_heatmap(events: list) -> go.Figure:
    if not events:
        return _empty("No events yet to show activity pattern.")

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hour_labels = [f"{h:02d}:00" for h in range(24)]
    matrix = [[0] * 24 for _ in range(7)]

    for event in events:
        ts = event.get("timestamp")
        if ts is None:
            continue
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        matrix[ts.weekday()][ts.hour] += 1

    fig = go.Figure(go.Heatmap(
        z=matrix, x=hour_labels, y=day_labels,
        colorscale=[[0, "#F4F7FC"], [0.5, "#93B7F5"], [1, "#1D4ED8"]],
        showscale=False, xgap=3, ygap=3,
        hovertemplate="<b>%{y}, %{x}</b><br>Sessions: %{z}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title=None, tickfont=dict(size=9, color=MUTED, family=FONT), dtick=2),
        yaxis=dict(title=None, tickfont=dict(size=11, color=MUTED, family=FONT)),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, height=220,
        margin=dict(l=6, r=6, t=6, b=6), font=dict(family=FONT),
    )
    return fig
