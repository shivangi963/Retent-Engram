"""
backend/dependency_graph.py
=============================
IMPROVEMENT 5 — Concept Dependency Graph

PURPOSE
-------
Concepts in CS are not isolated. "Process Management" depends on "OS".
"SQL" depends on "DBMS". If a student is struggling with OS, there's a
good chance their Process Management knowledge is also at risk.

This module:
  1. Reads prerequisite relationships from concepts.json
  2. Builds a directed dependency graph
  3. When concept X is urgent, flags all concepts that DEPEND ON X
  4. Builds a visual graph using Plotly for the dashboard

EXAMPLE:
  concepts.json has:
    { "concept_id": "process_mgmt", "prerequisites": ["os"] }
    { "concept_id": "memory_mgmt",  "prerequisites": ["os"] }
    { "concept_id": "sql",          "prerequisites": ["dbms"] }

  If "os" is High priority (recall < 40%):
    → flag "process_mgmt" as "secondary review" (depends on weak foundation)
    → flag "memory_mgmt" as "secondary review"

  This doesn't move them to High priority (they might be fine individually),
  but adds a warning badge: "⚠️ Foundation concept 'OS' is weak"

UPDATED concepts.json STRUCTURE:
  Each concept should have a "prerequisites" list.
  Empty list = no prerequisites (standalone concept).

  {
    "concept_id": "process_mgmt",
    "name": "Process Management",
    "difficulty": 4,
    "subject": "CS Core",
    "prerequisites": ["os"]          ← depends on OS
  }
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
from collections import defaultdict, deque


PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONCEPTS_PATH = os.path.join(PROJECT_ROOT, "data", "concepts.json")


# =============================================================================
# LOAD AND BUILD GRAPH
# =============================================================================

def load_dependency_graph() -> tuple:
    """
    Loads concepts.json and builds two graph representations:
      1. prerequisites:  { concept_id → list of concepts it needs }
         e.g. "process_mgmt" → ["os"]
      2. dependents:     { concept_id → list of concepts that need it }
         e.g. "os" → ["process_mgmt", "memory_mgmt"]

    The "dependents" graph is what we need to answer:
    "If OS is weak, which concepts are at risk?"

    Returns:
        tuple: (concepts_list, prerequisites_map, dependents_map)
    """
    with open(CONCEPTS_PATH) as f:
        concepts = json.load(f)

    # Build prerequisites map: concept → what it needs
    prerequisites = {
        c["concept_id"]: c.get("prerequisites", [])
        for c in concepts
    }

    # Build dependents map: concept → what needs it (reverse graph)
    dependents = defaultdict(list)
    for concept_id, prereqs in prerequisites.items():
        for prereq_id in prereqs:
            dependents[prereq_id].append(concept_id)

    return concepts, dict(prerequisites), dict(dependents)


# =============================================================================
# FIND CONCEPTS AT RISK
# =============================================================================

def get_concepts_at_risk(urgent_concept_ids: list) -> dict:
    """
    Given a list of urgent concepts (High priority), returns
    all concepts that build on them and are therefore "at risk".

    DOES A BFS (Breadth-First Search) through the dependency graph:
      If "os" is urgent:
        - Direct dependents: process_mgmt, memory_mgmt
        - Their dependents: (anything that needs process_mgmt or memory_mgmt)
        - And so on...

    WHY BFS AND NOT JUST ONE LEVEL?
      In a deep dependency chain: A → B → C → D
      If A is weak, D is at risk too — even though D only directly depends on C.
      BFS traverses the full chain.

    RETURN FORMAT:
      {
        "process_mgmt": {
          "at_risk_because": ["os"],
          "depth": 1          ← 1 = direct dependent, 2 = indirect, etc.
        },
        "memory_mgmt": {
          "at_risk_because": ["os"],
          "depth": 1
        }
      }

    Args:
        urgent_concept_ids: list of concept_ids currently at High priority

    Returns:
        dict: { concept_id: { "at_risk_because": [...], "depth": int } }
    """
    _, _, dependents = load_dependency_graph()

    at_risk = {}
    queue   = deque()

    # Seed the queue with direct dependents of urgent concepts
    for urgent_id in urgent_concept_ids:
        for dependent_id in dependents.get(urgent_id, []):
            if dependent_id not in urgent_concept_ids:  # don't re-flag urgent ones
                queue.append((dependent_id, urgent_id, 1))

    # BFS through the graph
    visited = set()
    while queue:
        concept_id, caused_by, depth = queue.popleft()

        if concept_id in visited:
            continue
        visited.add(concept_id)

        if concept_id not in at_risk:
            at_risk[concept_id] = {"at_risk_because": [], "depth": depth}

        at_risk[concept_id]["at_risk_because"].append(caused_by)

        # Add this concept's dependents to the queue (next level)
        for next_dependent in dependents.get(concept_id, []):
            if next_dependent not in visited and next_dependent not in urgent_concept_ids:
                queue.append((next_dependent, concept_id, depth + 1))

    return at_risk


# =============================================================================
# ENRICH QUEUE WITH DEPENDENCY WARNINGS
# =============================================================================

def enrich_queue_with_dependency_info(queue_items: list,
                                       recall_scores: list) -> list:
    """
    Adds dependency warning flags to queue items.

    HOW IT WORKS:
      1. Find which concepts are currently at High priority
      2. Call get_concepts_at_risk() to find at-risk dependents
      3. For each queue item, check if it's in the at-risk dict
      4. If yes, add "dependency_warning" field with the cause

    ADDS TO EACH QUEUE ITEM:
      {
        ...existing fields...,
        "dependency_warning": "⚠️ 'OS' (a foundation concept) is weak",
        "at_risk_depth": 1    ← 1=direct, 2=indirect
      }
      OR
      {
        ...existing fields...,
        "dependency_warning": None,   ← no warning
        "at_risk_depth": 0
      }

    Args:
        queue_items:   list from scheduler.build_review_queue()
        recall_scores: list from pipeline.compute_scores_for_user()
                       (to know which concepts are High priority)

    Returns:
        list: enriched queue_items with dependency warnings added
    """
    concepts, id_to_name = _load_concept_names()

    # Identify urgent concepts (High priority = recall < 40)
    urgent_ids = [
        s["concept_id"]
        for s in recall_scores
        if s.get("priority") == "High"
    ]

    if not urgent_ids:
        # No urgent concepts → no dependency warnings possible
        for item in queue_items:
            item["dependency_warning"] = None
            item["at_risk_depth"]      = 0
        return queue_items

    # Get concepts at risk
    at_risk = get_concepts_at_risk(urgent_ids)

    # Enrich each queue item
    for item in queue_items:
        cid = item["concept_id"]
        if cid in at_risk:
            risk_info   = at_risk[cid]
            caused_by   = risk_info["at_risk_because"]
            depth        = risk_info["depth"]

            # Build human-readable warning
            cause_names = [
                id_to_name.get(c, c) for c in caused_by
            ]
            cause_str = " + ".join(f"'{n}'" for n in cause_names[:2])

            item["dependency_warning"] = (
                f"⚠️ Foundation concept {cause_str} is weak — "
                f"this may also be affected"
            )
            item["at_risk_depth"] = depth
        else:
            item["dependency_warning"] = None
            item["at_risk_depth"]      = 0

    return queue_items


def _load_concept_names() -> tuple:
    """Helper: load concepts list and id→name map."""
    with open(CONCEPTS_PATH) as f:
        concepts = json.load(f)
    id_to_name = {c["concept_id"]: c["name"] for c in concepts}
    return concepts, id_to_name


# =============================================================================
# PLOTLY DEPENDENCY GRAPH VISUALIZATION
# =============================================================================

def build_dependency_graph_figure(recall_scores: list = None):
    """
    Builds an interactive Plotly network graph showing concept dependencies.

    NODES:
      - Each concept is a node
      - Color: red=High priority, amber=Medium, green=Low, gray=Not tracked
      - Size: proportional to recall score (bigger = better remembered)

    EDGES:
      - Arrow from prerequisite → dependent
      - "OS" → "Process Management" means OS must be understood first

    HOVER:
      Hovering a node shows: concept name, recall %, priority

    CALLED BY:
      2_dashboard.py in an expander section

    Args:
        recall_scores: list from pipeline.compute_scores_for_user()
                       (if None, all nodes appear gray)

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    import math

    concepts, prerequisites, dependents = load_dependency_graph()

    # Build recall lookup
    recall_lookup = {}
    priority_lookup = {}
    if recall_scores:
        for s in recall_scores:
            recall_lookup[s["concept_id"]]   = s.get("recall_score", 50)
            priority_lookup[s["concept_id"]] = s.get("priority", "Unknown")

    # Layout: arrange nodes in a circle
    n = len(concepts)
    positions = {}
    for i, concept in enumerate(concepts):
        angle = 2 * math.pi * i / n
        positions[concept["concept_id"]] = (
            math.cos(angle) * 2,
            math.sin(angle) * 2
        )

    # Build edge traces
    edge_x, edge_y = [], []
    for concept_id, prereqs in prerequisites.items():
        if concept_id in positions:
            x2, y2 = positions[concept_id]
            for prereq_id in prereqs:
                if prereq_id in positions:
                    x1, y1 = positions[prereq_id]
                    edge_x += [x1, x2, None]
                    edge_y += [y1, y2, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="#444"),
        hoverinfo="none"
    )

    # Build node traces
    node_x, node_y   = [], []
    node_text         = []
    node_hover        = []
    node_colors       = []
    node_sizes        = []

    color_map = {
        "High":    "#EF4444",
        "Medium":  "#F59E0B",
        "Low":     "#10B981",
        "Unknown": "#6B7280"
    }

    for concept in concepts:
        cid  = concept["concept_id"]
        name = concept["name"]

        if cid not in positions:
            continue

        x, y = positions[cid]
        node_x.append(x)
        node_y.append(y)
        node_text.append(name)

        recall   = recall_lookup.get(cid)
        priority = priority_lookup.get(cid, "Unknown")
        color    = color_map.get(priority, color_map["Unknown"])

        if recall is not None:
            size  = 20 + recall * 0.3   # 20–50px based on recall
            hover = f"{name}<br>Recall: {recall:.1f}%<br>Priority: {priority}"
        else:
            size  = 20
            hover = f"{name}<br>(no data yet)"

        node_colors.append(color)
        node_sizes.append(size)
        node_hover.append(hover)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        hovertext=node_hover,
        text=node_text,
        textposition="top center",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color="#1E2130"),
            opacity=0.9
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Concept Dependency Graph",
            showlegend=False,
            hovermode="closest",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=500,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=50, b=20),
            annotations=[
                dict(
                    text="🔴 High  🟡 Medium  🟢 Low  ⬛ No data",
                    xref="paper", yref="paper",
                    x=0.01, y=-0.02,
                    showarrow=False,
                    font=dict(size=12)
                )
            ]
        )
    )

    return fig