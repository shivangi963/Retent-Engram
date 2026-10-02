"""
frontend/views/study.py
========================
Generate and work through AI review material for one concept:
flashcards, summaries, quizzes, and coding tasks — grounded in your notes.

(Rewrite of the old 3_review.py. Render helpers are defined before they are
used, which fixes a NameError the old page hit on every generation.)
"""
import os
import re
import sys

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rag.generate import (
    generate_content, rate_content, mark_content_as_reviewed, is_ollama_running,
    FLASHCARD, SUMMARY, QUIZ, CODING_TASK, VALID_CONTENT_TYPES,
)
from rag.ingest import is_index_available
from backend.db import get_recall_scores, get_collection

from frontend.components.theme import html, page_header, section_title, pill, card, icon, empty_state, TONES
from frontend.components.data import concepts_by_id, current_user, priority_for, CONTENT_LABELS, flash
from frontend.components.parsers import parse_flashcard, summary_body, parse_quiz, parse_coding, plain

user_id, user_name = current_user()
concepts_list = list(concepts_by_id().values())
id_to_name = {c["concept_id"]: c["name"] for c in concepts_list}

TYPE_META = {
    FLASHCARD:   ("layers", "purple"),
    SUMMARY:     ("file-text", "blue"),
    QUIZ:        ("target", "amber"),
    CODING_TASK: ("code", "teal"),
}

# ── render helpers (defined first, used later — this is the actual fix) ─────

def render_flashcard(content: str, box_key: str):
    question, answer = parse_flashcard(content)
    reveal_key = f"reveal_{box_key}"
    revealed = st.session_state.get(reveal_key, False)

    html(f"""<div class="rt-flash">
        <div class="rt-flash-l">Question</div>
        <div class="rt-flash-q">{plain(question)}</div>
      </div>""")
    if revealed:
        html(f"""<div class="rt-flash-a"><div class="rt-flash-l">Answer</div>
              <div class="t">{plain(answer)}</div></div>""")
        if st.button("Hide answer", key=f"hide_{box_key}", width="stretch"):
            st.session_state[reveal_key] = False
            st.rerun()
    else:
        if st.button("Reveal answer", key=f"show_{box_key}", type="primary", width="stretch", icon=":material/visibility:"):
            st.session_state[reveal_key] = True
            st.rerun()


def render_summary(content: str):
    with card("summary_box"):
        st.markdown(summary_body(content))


def render_quiz(content: str, box_key: str):
    questions = parse_quiz(content)
    if not questions:
        st.markdown(content)
        return
    for i, q in enumerate(questions):
        with card(f"quiz_{box_key}_{i}"):
            html(f"""<div class="rt-qcard"><div class="rt-qnum">Q{i+1}</div>
                  <div class="rt-qtext">{plain(q['question'])}</div></div>""")
            user_answer = st.radio(f"Q{i+1}", options=q["options"], key=f"quiz_{box_key}_{i}",
                                    label_visibility="collapsed")
            if st.button(f"Check answer", key=f"check_{box_key}_{i}"):
                user_letter = user_answer[0] if user_answer else ""
                correct = q.get("correct", "")
                if user_letter == correct:
                    html(f'<div class="rt-result" style="background:{TONES["green"][1]};'
                         f'border-color:{TONES["green"][2]};color:{TONES["green"][0]}">'
                         f'<b>Correct.</b> {plain(q.get("explanation", ""))}</div>')
                else:
                    html(f'<div class="rt-result" style="background:{TONES["red"][1]};'
                         f'border-color:{TONES["red"][2]};color:{TONES["red"][0]}">'
                         f'<b>Not quite</b> — correct answer is {correct}. {plain(q.get("explanation", ""))}</div>')


def render_coding_task(content: str, box_key: str):
    sections = parse_coding(content)

    with card(f"code_{box_key}"):
        if "PROBLEM" in sections:
            html(f'<div style="font-weight:700;color:#0F1F3D;margin-bottom:.35rem">Problem</div>'
                 f'<div style="color:#334155;font-size:.93rem;line-height:1.55">{plain(sections["PROBLEM"])}</div>')
        if "INPUT" in sections:
            html('<div class="rt-code-label">Input</div>')
            st.code(sections["INPUT"], language="python")
        if "HINT" in sections:
            with st.expander("Show hint"):
                st.info(plain(sections["HINT"]))
        if "EXPECTED OUTPUT" in sections:
            with st.expander("Show expected output (try first)"):
                st.code(sections["EXPECTED OUTPUT"], language="python")

        html('<div class="rt-code-label">Your solution</div>')
        editor_key = f"editor_{box_key}"
        user_code = st.text_area("Python code", value="# Write your solution here\n\n", height=220,
                                  key=editor_key, label_visibility="collapsed")

        run_col, _ = st.columns([1, 3])
        with run_col:
            run_clicked = st.button("Run code", key=f"run_{box_key}", type="primary",
                                     width="stretch", icon=":material/play_arrow:")

        if run_clicked and user_code.strip():
            from backend.sandbox import run_code, compare_output
            with st.spinner("Running your code…"):
                result = run_code(user_code)
            html('<div class="rt-code-label">Output</div>')
            if result["timed_out"]:
                st.error("Timed out after 10 seconds — check for infinite loops.")
            elif result["error"]:
                st.code(result["error"], language="text")
                if result["stdout"]:
                    st.code(result["stdout"], language="text")
            else:
                st.code(result["stdout"] or "(no output)", language="text")
                expected = sections.get("EXPECTED OUTPUT", "")
                if expected and result["stdout"]:
                    comparison = compare_output(result["stdout"], expected)
                    tone = "green" if comparison["passed"] else ("amber" if comparison["similarity"] > 0.8 else "red")
                    fg, bg, bd = TONES[tone]
                    html(f'<div class="rt-result" style="background:{bg};border-color:{bd};color:{fg}">'
                         f'{comparison["message"]}</div>')
                    if not comparison["passed"]:
                        with st.expander("See expected vs. actual"):
                            ec, ac = st.columns(2)
                            with ec:
                                st.caption("Expected")
                                st.code(expected, language="text")
                            with ac:
                                st.caption("Your output")
                                st.code(result["stdout"], language="text")
        elif run_clicked:
            st.warning("Write some code first.")

    if not sections:
        st.code(content, language="markdown")


# ── page ──────────────────────────────────────────────────────────────────────

page_header("Study", "AI-generated review material, tailored to your current recall level.")

kb_ready, llm_ready = is_index_available(), is_ollama_running()
if not (kb_ready and llm_ready):
    missing = []
    if not kb_ready:
        missing.append("the knowledge base isn't built yet (`python rag/ingest.py`)")
    if not llm_ready:
        missing.append("Ollama isn't running (`ollama serve`)")
    st.warning("Content generation is unavailable — " + " and ".join(missing) + ".", icon=":material/error:")

recall_map = {doc["concept_id"]: doc for doc in get_recall_scores(user_id)}

# ── prefill from a "Study now" / "Generate" link elsewhere in the app ────────
prefill = st.session_state.pop("study_prefill", None)

left, right = st.columns([1, 1.7], gap="large")

with left:
    with card("controls"):
        html('<div style="font-weight:700;color:#0F1F3D;font-size:1.02rem;margin-bottom:.7rem">'
             'Generate content</div>')

        sorted_concepts = sorted(concepts_list, key=lambda c: recall_map.get(c["concept_id"], {}).get("recall_score", 50))
        names = [c["name"] for c in sorted_concepts]
        name_to_id = {c["name"]: c["concept_id"] for c in sorted_concepts}

        default_idx = 0
        if prefill and prefill.get("concept_id") in id_to_name:
            wanted = id_to_name[prefill["concept_id"]]
            if wanted in names:
                default_idx = names.index(wanted)

        selected_name = st.selectbox("Concept", options=names, index=default_idx,
                                      help="Sorted by recall — lowest (most urgent) first.")
        selected_cid = name_to_id[selected_name]

        if selected_cid in recall_map:
            recall_score = recall_map[selected_cid].get("recall_score", 50.0)
            priority = recall_map[selected_cid].get("priority", priority_for(recall_score))
        else:
            recall_score, priority = 50.0, "Medium"
            st.caption("No recall data yet for this concept — log a session first.")

        tone = {"High": "red", "Medium": "amber", "Low": "green"}[priority]
        html(f'<div style="margin:.5rem 0">{pill(f"Recall {recall_score:.0f}% · {priority} priority", tone, dot=True)}</div>')
        level = "Beginner" if recall_score < 40 else ("Intermediate" if recall_score < 65 else "Advanced")
        st.caption(f"{level} content will be generated.")

        html('<div style="height:.4rem"></div>')
        type_options = [FLASHCARD, SUMMARY, QUIZ, CODING_TASK]
        type_labels = {FLASHCARD: "Flashcard", SUMMARY: "Summary", QUIZ: "Quiz (3 MCQs)", CODING_TASK: "Coding task"}

        default_type_idx = 0
        if prefill and prefill.get("content_type") in type_options:
            default_type_idx = type_options.index(prefill["content_type"])

        selected_type = st.radio("Content type", options=type_options, index=default_type_idx,
                                  format_func=lambda t: type_labels[t])

        force_regen = st.checkbox("Force regenerate", value=False,
                                   help="Skip the cache and generate fresh content.")

        generate_clicked = st.button("Generate", type="primary", width="stretch",
                                      icon=":material/auto_awesome:",
                                      disabled=not (kb_ready and llm_ready))

with right:
    if generate_clicked:
        st.session_state["gen_concept_id"] = selected_cid
        st.session_state["gen_concept_name"] = selected_name
        st.session_state["gen_content_type"] = selected_type
        st.session_state["gen_recall_score"] = recall_score
        st.session_state["gen_force"] = force_regen
        st.session_state["gen_result"] = None

    if st.session_state.get("gen_concept_id") and st.session_state.get("gen_result") is None:
        spinner_msg = "Retrieving from your notes and asking the model… (can take 30–90s on CPU)"
        with st.spinner(spinner_msg):
            result = generate_content(
                user_id=user_id,
                concept_id=st.session_state["gen_concept_id"],
                concept_name=st.session_state["gen_concept_name"],
                content_type=st.session_state["gen_content_type"],
                recall_score=st.session_state["gen_recall_score"],
                force_regenerate=st.session_state["gen_force"],
            )
        st.session_state["gen_result"] = result

    result = st.session_state.get("gen_result")

    if result is None:
        empty_state("sparkles", "Nothing generated yet",
                     "Pick a concept and a content type on the left, then press Generate.")
    elif result.get("error"):
        st.error(result["error"])
    else:
        content = result["content"]
        content_type = result["content_type"]
        difficulty_level = result.get("difficulty_level", "intermediate")
        from_cache = result.get("from_cache", False)
        ic_name, tone = TYPE_META.get(content_type, ("file-text", "blue"))
        box_key = f"{selected_cid}_{content_type}"

        badge_row = st.columns([2, 2, 2])
        with badge_row[0]:
            html(pill(CONTENT_LABELS.get(content_type, content_type), tone, dot=True))
        with badge_row[1]:
            html(pill(difficulty_level.title(), "blue"))
        with badge_row[2]:
            st.caption(("From cache · < 24h old" if from_cache else "Freshly generated"))

        html('<div style="height:.6rem"></div>')

        if content_type == FLASHCARD:
            render_flashcard(content, box_key)
        elif content_type == SUMMARY:
            render_summary(content)
        elif content_type == QUIZ:
            render_quiz(content, box_key)
        elif content_type == CODING_TASK:
            render_coding_task(content, box_key)
        else:
            st.markdown(content)

        # ── feedback ──────────────────────────────────────────────────────────
        html('<div style="height:.8rem"></div>')
        section_title("Was this helpful?")
        f1, f2, f3, f4 = st.columns([1, 1, 1.3, 1.7], gap="small")
        with f1:
            if st.button("Good", key="thumbs_up", icon=":material/thumb_up:", width="stretch"):
                rate_content(user_id, selected_cid, selected_type, 1)
                flash("Thanks for the feedback!")
                st.rerun()
        with f2:
            if st.button("Bad", key="thumbs_down", icon=":material/thumb_down:", width="stretch"):
                st.session_state["show_reason_picker"] = True
        with f3:
            if st.button("Regenerate", key="regen", icon=":material/refresh:", width="stretch"):
                st.session_state["gen_force"] = True
                st.session_state["gen_result"] = None
                st.rerun()
        with f4:
            if st.button("Mark as studied", key="studied", type="primary", width="stretch",
                         icon=":material/check_circle:"):
                mark_content_as_reviewed(user_id, selected_cid, selected_type)
                flash(f"'{selected_name}' marked as studied.")
                st.rerun()

        if st.session_state.get("show_reason_picker", False):
            from backend.prompt_optimizer import save_feedback_with_reason
            reason_labels = {
                "too_easy": "Too basic / easy", "too_hard": "Too advanced / hard",
                "off_topic": "Off-topic or irrelevant", "wrong_format": "Wrong format",
                "inaccurate": "Factually wrong", "too_long": "Too long / verbose",
                "too_short": "Too brief", "generic": "Too generic",
            }
            with card("reason"):
                reason_choice = st.selectbox("What was wrong with it?", options=list(reason_labels.values()))
                reason_code = {v: k for k, v in reason_labels.items()}[reason_choice]
                if st.button("Submit feedback", key="submit_reason"):
                    save_feedback_with_reason(user_id, selected_cid, selected_type, -1, reason_code)
                    st.session_state["show_reason_picker"] = False
                    flash("Feedback saved — future generations will improve.")
                    st.rerun()

        with st.expander("Generation history for this concept"):
            history = list(get_collection("generated_content").find(
                {"user_id": user_id, "concept_id": selected_cid}, {"_id": 0},
            ).sort("generated_at", -1).limit(8))
            if not history:
                st.caption("No generation history yet.")
            else:
                for h in history:
                    ts = h.get("generated_at", "")
                    ts_s = ts.strftime("%d %b, %I:%M %p") if hasattr(ts, "strftime") else str(ts)
                    rating = h.get("rating", 0)
                    rtone = "green" if rating == 1 else ("red" if rating == -1 else "gray")
                    rlabel = "Helpful" if rating == 1 else ("Not helpful" if rating == -1 else "No rating")
                    html(f'<div class="rt-row"><div class="rt-row-n">{CONTENT_LABELS.get(h.get("content_type",""), h.get("content_type",""))}'
                         f'<span class="rt-row-s"> · {h.get("difficulty_level","")} · {ts_s}</span></div>'
                         f'<div class="rt-row-r">{pill(rlabel, rtone)}</div></div>')
