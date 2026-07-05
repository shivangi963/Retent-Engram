import streamlit as st
import json
import os
import sys
import re
from datetime import datetime, timezone

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from rag.generate import (
    generate_content,
    rate_content,
    mark_content_as_reviewed,
    is_ollama_running,
    FLASHCARD, SUMMARY, QUIZ, CODING_TASK,
    VALID_CONTENT_TYPES
)
from rag.ingest import is_index_available
from backend.db import get_recall_scores, get_collection
from backend.ml.pipeline import compute_scores_for_user


# =============================================================================
# PAGE SETUP
# =============================================================================

st.set_page_config(
    page_title="Review — Retent Engram",
    page_icon="📖",
    layout="wide"
)

st.title("📖 Review Content")
st.caption("AI-generated study materials tailored to your current recall level.")


# =============================================================================
# LOAD CONCEPTS
# =============================================================================

CONCEPTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "concepts.json"
)

@st.cache_data
def load_concepts():
    with open(CONCEPTS_PATH) as f:
        return json.load(f)

concepts = load_concepts()
id_to_name    = {c["concept_id"]: c["name"]    for c in concepts}
id_to_subject = {c["concept_id"]: c["subject"] for c in concepts}


# =============================================================================
# USER CHECK
# =============================================================================

if "user_id" not in st.session_state or not st.session_state.user_id:
    st.warning("⚠️ Please go to **Log Event** first and set your user ID.")
    st.stop()

user_id   = st.session_state.user_id
user_name = st.session_state.get("user_name", user_id)


# =============================================================================
# SYSTEM STATUS CHECK
# =============================================================================

status_col1, status_col2 = st.columns(2)

with status_col1:
    if is_index_available():
        st.success("✅ Knowledge base: Ready")
    else:
        st.error("❌ Knowledge base: Not built — run `python rag/ingest.py`")

with status_col2:
    if is_ollama_running():
        st.success("✅ Ollama (Mistral): Running")
    else:
        st.warning("⚠️ Ollama: Not running — run `ollama serve` in terminal")

st.divider()


# =============================================================================
# FETCH RECALL SCORES
# =============================================================================

@st.cache_data(ttl=60)   # cache for 60 seconds to avoid hammering MongoDB
def get_user_recall_scores(uid: str):
    """
    Returns recall scores sorted lowest first.
    Cached for 60s so rapid page interactions don't recompute.
    """
    scores = get_recall_scores(uid)
    return {doc["concept_id"]: doc for doc in scores}

recall_map = get_user_recall_scores(user_id)


# =============================================================================
# TWO-COLUMN LAYOUT: Controls (left) + Content (right)
# =============================================================================

left_col, right_col = st.columns([1, 2])


# =============================================================================
# LEFT COLUMN — CONTROLS
# =============================================================================

with left_col:
    st.subheader("⚙️ Generate Content")

    # ── Concept selector ──────────────────────────────────────────────────────
    # Sort concepts by recall (lowest first) so most urgent is at top
    def sort_key(c):
        score = recall_map.get(c["concept_id"], {}).get("recall_score", 50)
        return score

    sorted_concepts = sorted(concepts, key=sort_key)
    concept_options = {c["name"]: c["concept_id"] for c in sorted_concepts}

    selected_name = st.selectbox(
        "Select Concept",
        options=list(concept_options.keys()),
        help="Concepts are sorted by recall (lowest first = most urgent)."
    )
    selected_cid = concept_options[selected_name]

    # ── Show recall info for selected concept ─────────────────────────────────
    if selected_cid in recall_map:
        recall_doc = recall_map[selected_cid]
        recall_score = recall_doc.get("recall_score", 50.0)
        priority     = recall_doc.get("priority", "Medium")

        # Color the metric based on priority
        if priority == "High":
            st.error(f"Recall: **{recall_score:.1f}%** — 🔴 High Priority")
        elif priority == "Medium":
            st.warning(f"Recall: **{recall_score:.1f}%** — 🟡 Medium Priority")
        else:
            st.success(f"Recall: **{recall_score:.1f}%** — 🟢 Low Priority")

        # Difficulty level label
        if recall_score < 40:
            level_label = "📗 Beginner content will be generated"
        elif recall_score < 65:
            level_label = "📘 Intermediate content will be generated"
        else:
            level_label = "📕 Advanced content will be generated"
        st.caption(level_label)
    else:
        recall_score = 50.0
        st.info("No recall data yet. Log some events first.")

    st.divider()

    # ── Content type selector ─────────────────────────────────────────────────
    content_type_labels = {
        FLASHCARD:   "📝 Flashcard",
        SUMMARY:     "📖 Summary",
        QUIZ:        "❓ Quiz (3 MCQs)",
        CODING_TASK: "💻 Coding Task"
    }

    selected_label = st.radio(
        "Content Type",
        options=list(content_type_labels.values()),
        help=(
            "Flashcard: single Q&A\n"
            "Summary: 5 bullet points\n"
            "Quiz: 3 multiple choice\n"
            "Coding Task: Python problem"
        )
    )

    # Map label back to content type key
    selected_type = {v: k for k, v in content_type_labels.items()}[selected_label]

    st.divider()

    # ── Force regenerate option ───────────────────────────────────────────────
    force_regen = st.checkbox(
        "🔄 Force regenerate",
        value=False,
        help="Skip the cache and generate fresh content even if recent content exists."
    )

    # ── Generate button ───────────────────────────────────────────────────────
    generate_clicked = st.button(
        "🚀 Generate",
        type="primary",
        use_container_width=True,
        disabled=not is_ollama_running() or not is_index_available()
    )

    if not is_ollama_running():
        st.caption("⚠️ Start Ollama to enable generation: `ollama serve`")
    if not is_index_available():
        st.caption("⚠️ Build knowledge base: `python rag/ingest.py`")


# =============================================================================
# RIGHT COLUMN — CONTENT DISPLAY
# =============================================================================

with right_col:

    # ── Trigger generation when button is clicked ─────────────────────────────
    if generate_clicked:
        # Store generation request in session state
        st.session_state["gen_concept_id"]   = selected_cid
        st.session_state["gen_concept_name"] = selected_name
        st.session_state["gen_content_type"] = selected_type
        st.session_state["gen_recall_score"] = recall_score
        st.session_state["gen_force"]        = force_regen
        st.session_state["gen_result"]       = None  # clear previous result

    # ── Run generation if triggered ───────────────────────────────────────────
    if st.session_state.get("gen_concept_id") and st.session_state.get("gen_result") is None:

        cid        = st.session_state["gen_concept_id"]
        cname      = st.session_state["gen_concept_name"]
        ctype      = st.session_state["gen_content_type"]
        recall     = st.session_state["gen_recall_score"]
        force      = st.session_state["gen_force"]

        # Show spinner during generation (can take 30-90 seconds)
        spinner_msg = (
            "🤖 Retrieving from knowledge base + asking Mistral 7B... "
            "(this can take 30–90 seconds on CPU)"
        )

        with st.spinner(spinner_msg):
            result = generate_content(
                user_id=user_id,
                concept_id=cid,
                concept_name=cname,
                content_type=ctype,
                recall_score=recall,
                force_regenerate=force
            )

        # Store result in session state for display
        st.session_state["gen_result"] = result

    # ── Display generated content ─────────────────────────────────────────────
    result = st.session_state.get("gen_result")

    if result is None:
        # No generation yet — show prompt
        st.info(
            "👈 Select a concept and content type, then click **Generate**.\n\n"
            "Content is cached for 24 hours — instant if already generated today."
        )

    elif result.get("error"):
        # Error during generation
        st.error(f"Generation failed:\n\n{result['error']}")

    else:
        # ── Success — render content ──────────────────────────────────────────
        content          = result["content"]
        content_type     = result["content_type"]
        difficulty_level = result.get("difficulty_level", "intermediate")
        from_cache       = result.get("from_cache", False)

        # Header badges
        badge_col1, badge_col2, badge_col3 = st.columns(3)
        with badge_col1:
            type_emoji = {
                FLASHCARD: "📝", SUMMARY: "📖",
                QUIZ: "❓", CODING_TASK: "💻"
            }.get(content_type, "📄")
            st.markdown(f"**{type_emoji} {content_type.replace('_', ' ').title()}**")
        with badge_col2:
            diff_color = {"beginner": "🟢", "intermediate": "🔵", "advanced": "🔴"}
            st.markdown(f"{diff_color.get(difficulty_level,'⚪')} **{difficulty_level.title()}** level")
        with badge_col3:
            if from_cache:
                st.caption("⚡ From cache (< 24h old)")
            else:
                st.caption("✨ Freshly generated")

        st.divider()


        # ── RENDER BASED ON CONTENT TYPE ─────────────────────────────────────

        if content_type == FLASHCARD:
            render_flashcard(content)

        elif content_type == SUMMARY:
            render_summary(content)

        elif content_type == QUIZ:
            render_quiz(content, selected_cid)

        elif content_type == CODING_TASK:
            render_coding_task(content)

        else:
            st.markdown(content)

        # ── FEEDBACK ROW ─────────────────────────────────────────────────────
        st.divider()
        st.markdown("**Was this content helpful?**")

        fb_col1, fb_col2, fb_col3, fb_col4 = st.columns([1, 1, 1, 2])

        with fb_col1:
            if st.button("👍 Good", key="thumbs_up", use_container_width=True):
                rate_content(user_id, selected_cid, selected_type, 1)
                st.success("Thanks for the feedback!")

        with fb_col2:
            if st.button("👎 Bad", key="thumbs_down", use_container_width=True):
                rate_content(user_id, selected_cid, selected_type, -1)
                st.warning("Got it. Try regenerating for different content.")

        with fb_col3:
            if st.button("🔄 Regenerate", key="regen", use_container_width=True):
                # Mark as force regenerate and clear current result
                st.session_state["gen_force"]  = True
                st.session_state["gen_result"] = None
                st.rerun()

        with fb_col4:
            if st.button("✅ Mark as Studied", key="studied",
                         type="primary", use_container_width=True):
                mark_content_as_reviewed(user_id, selected_cid, selected_type)
                st.success(f"'{selected_name}' marked as studied!")

        # ── CONTENT HISTORY ───────────────────────────────────────────────────
        with st.expander("📚 Generation History for this concept"):
            from backend.db import get_collection as _gc
            history = list(
                _gc("generated_content").find(
                    {"user_id": user_id, "concept_id": selected_cid},
                    {"_id": 0}
                ).sort("generated_at", -1).limit(8)
            )

            if not history:
                st.caption("No generation history yet.")
            else:
                for h in history:
                    ts   = h.get("generated_at", "")
                    ts_s = ts.strftime("%d %b, %I:%M %p") if hasattr(ts, "strftime") else str(ts)
                    rating = h.get("rating", 0)
                    rating_str = "👍" if rating == 1 else ("👎" if rating == -1 else "⬜")
                    ctype_h = h.get("content_type", "")
                    diff_h  = h.get("difficulty_level", "")
                    st.caption(f"{rating_str} {ctype_h} · {diff_h} · {ts_s}")


# =============================================================================
# CONTENT RENDERING FUNCTIONS
# =============================================================================

def render_flashcard(content: str):
    """
    Renders flashcard content with a reveal mechanic.

    FORMAT EXPECTED FROM LLM:
      QUESTION: [question text]
      ANSWER: [answer text]

    REVEAL MECHANIC:
      Answer is hidden by default.
      Student clicks "Reveal Answer" to see it.
      This simulates real flashcard behavior (active recall).
    """
    # Parse QUESTION and ANSWER from LLM output
    question = ""
    answer   = ""

    lines = content.split("\n")
    current_section = None
    q_lines = []
    a_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("QUESTION:"):
            current_section = "Q"
            q_text = stripped[9:].strip()
            if q_text:
                q_lines.append(q_text)
        elif stripped.upper().startswith("ANSWER:"):
            current_section = "A"
            a_text = stripped[7:].strip()
            if a_text:
                a_lines.append(a_text)
        elif stripped and current_section == "Q":
            q_lines.append(stripped)
        elif stripped and current_section == "A":
            a_lines.append(stripped)

    question = " ".join(q_lines) if q_lines else content
    answer   = " ".join(a_lines) if a_lines else "Answer not parsed correctly."

    # Display
    st.markdown("### 💭 Question")
    st.info(question)

    # Answer reveal button using session state
    reveal_key = f"reveal_flashcard_{hash(question) % 10000}"
    if not st.session_state.get(reveal_key, False):
        if st.button("👁️ Reveal Answer", key=f"btn_{reveal_key}"):
            st.session_state[reveal_key] = True
            st.rerun()
    else:
        st.markdown("### ✅ Answer")
        st.success(answer)
        if st.button("🙈 Hide Answer", key=f"hide_{reveal_key}"):
            st.session_state[reveal_key] = False
            st.rerun()


def render_summary(content: str):
    """
    Renders summary bullet points with clean formatting.

    FORMAT EXPECTED FROM LLM:
      SUMMARY: Concept Name — Key Points

      • **Key Term**: explanation
      • **Key Term**: explanation
    """
    st.markdown("### 📋 Summary")

    # Clean up the content and display as markdown
    # The LLM uses bullet points which Streamlit's markdown renders natively
    lines = content.split("\n")
    rendered_lines = []

    for line in lines:
        stripped = line.strip()
        # Remove the "SUMMARY:" header line (we have our own header above)
        if stripped.upper().startswith("SUMMARY:"):
            continue
        rendered_lines.append(line)

    cleaned = "\n".join(rendered_lines).strip()

    # Display in an info box for visual distinction
    with st.container(border=True):
        st.markdown(cleaned)


def render_quiz(content: str, concept_id: str):
    """
    Renders a 3-question MCQ quiz with interactive answer selection.

    FORMAT EXPECTED FROM LLM:
      Q1: [question]
      A) option1
      B) option2
      C) option3
      D) option4
      CORRECT: B
      EXPLANATION: why B is correct

    INTERACTION:
      Student sees all 3 questions.
      For each question, they pick an option using radio buttons.
      "Check Answers" button reveals correct/incorrect with explanations.
    """
    st.markdown("### ❓ Quiz")

    # Parse questions from LLM output
    questions = parse_quiz_content(content)

    if not questions:
        # Fallback: show raw content if parsing fails
        st.markdown(content)
        return

    # Display each question
    for i, q in enumerate(questions):
        st.markdown(f"**Q{i+1}: {q['question']}**")

        # Radio button for answer selection
        user_answer = st.radio(
            f"Select answer for Q{i+1}",
            options=q["options"],
            key=f"quiz_q{i}_{concept_id}",
            label_visibility="collapsed"
        )

        # Check answer button per question
        if st.button(f"Check Q{i+1}", key=f"check_q{i}_{concept_id}"):
            # Extract letter from user selection (e.g. "B) option" → "B")
            user_letter = user_answer[0] if user_answer else ""
            correct_letter = q.get("correct", "")

            if user_letter == correct_letter:
                st.success(f"✅ Correct! {q.get('explanation', '')}")
            else:
                st.error(
                    f"❌ Incorrect. Correct answer: **{correct_letter}**\n\n"
                    f"{q.get('explanation', '')}"
                )

        st.write("")   # spacing between questions


def parse_quiz_content(content: str) -> list:
    """
    Parses quiz output from Mistral into structured question dicts.

    Returns:
        list of dicts: [{question, options, correct, explanation}]
    """
    questions = []

    # Split by Q1/Q2/Q3 pattern
    question_blocks = re.split(r'Q\d+:', content)[1:]   # split on Q1: Q2: Q3:

    for block in question_blocks:
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        if not lines:
            continue

        q_dict = {
            "question":    "",
            "options":     [],
            "correct":     "",
            "explanation": ""
        }

        q_dict["question"] = lines[0]

        for line in lines[1:]:
            # Match option lines: A) text, B) text, etc.
            if re.match(r'^[A-D]\)', line):
                q_dict["options"].append(line)
            elif line.upper().startswith("CORRECT:"):
                q_dict["correct"] = line.split(":", 1)[1].strip().upper()
            elif line.upper().startswith("EXPLANATION:"):
                q_dict["explanation"] = line.split(":", 1)[1].strip()

        if q_dict["question"] and q_dict["options"]:
            questions.append(q_dict)

    return questions


def render_coding_task(content: str):
    """
    Renders a coding task with code block and hidden expected output.

    FORMAT EXPECTED FROM LLM:
      CODING TASK: Concept — Topic

      PROBLEM:
      [description]

      INPUT:
      [example]

      EXPECTED OUTPUT:
      [output]

      HINT:
      [hint]
    """
    st.markdown("### 💻 Coding Task")

    # Parse sections
    sections = {}
    current_section = None
    current_lines   = []

    for line in content.split("\n"):
        stripped = line.strip()
        upper    = stripped.upper()

        if upper.startswith("CODING TASK:"):
            continue   # skip title line

        matched_section = None
        for keyword in ["PROBLEM:", "INPUT:", "EXPECTED OUTPUT:", "HINT:"]:
            if upper.startswith(keyword):
                if current_section:
                    sections[current_section] = "\n".join(current_lines).strip()
                current_section = keyword.rstrip(":")
                current_lines   = [stripped[len(keyword):].strip()]
                matched_section = keyword
                break

        if matched_section is None and current_section is not None:
            current_lines.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_lines).strip()

    # Render each section
    if "PROBLEM" in sections:
        st.markdown("**📋 Problem:**")
        st.markdown(sections["PROBLEM"])

    if "INPUT" in sections:
        st.markdown("**📥 Input:**")
        st.code(sections["INPUT"], language="python")

    if "HINT" in sections:
        with st.expander("💡 Show Hint"):
            st.info(sections["HINT"])

    # Expected output is HIDDEN by default
    if "EXPECTED OUTPUT" in sections:
        with st.expander("🎯 Show Expected Output (try first!)"):
            st.code(sections["EXPECTED OUTPUT"], language="python")

    # Python code editor area
    st.markdown("**✍️ Write your solution:**")
    st.code_input = st.text_area(
        "Python code",
        value="# Write your solution here\n\n",
        height=200,
        key=f"code_editor_{hash(content) % 10000}"
    )

    if not sections:
        # Fallback: show raw content in code block
        st.code(content, language="markdown")