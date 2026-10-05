"""
frontend/components/practice.py
===============================
Practice widgets that SAVE THEMSELVES.  Every page that lets the student practise
(Study, My Notes) uses these, so quizzes, flashcards, coding tasks and reading are
recorded the same way everywhere and nobody has to log them by hand.

Each widget gets `log(score, event_type, source, minutes)` (see backend.study_log.make_logger)
and returns True in the run where it logged something, so the page can refresh its own state.
Logging happens at most once per set (a `...:logged` flag in session_state); call reset(prefix)
when new content replaces the old.
"""
import streamlit as st

from backend.sandbox import run_code, compare_output

GRADES = {"Knew it": 1.0, "Partly": 0.5, "Forgot": 0.0}
RATINGS = (("Got it", 0.85), ("Mostly", 0.60), ("Need to revisit", 0.30))


def reset(prefix: str):
    """Forget the widget state of earlier content."""
    for k in [k for k in st.session_state.keys() if str(k).startswith(prefix)]:
        del st.session_state[k]


def _saved(res, text="Saved automatically"):
    recall = res.get("recall") if isinstance(res, dict) else None
    st.toast(f"{text} - recall now {recall:.0f}%" if recall is not None else text)


# ------------------------------------------------------------------ reading / summary
def self_rating(key, log, prompt="Finished reading? How did it go?", event_type="reading",
                source="notes", minutes=5) -> bool:
    st.markdown(f"**{prompt}**")
    logged = False
    for col, (label, score) in zip(st.columns(len(RATINGS)), RATINGS):
        with col:
            if st.button(label, key=f"{key}:rate:{score}"):
                res = log(score, event_type, source, minutes)
                if res:
                    _saved(res)
                    logged = True
                else:
                    st.error("Could not save to the database. Is MongoDB running?")
    return logged


# ------------------------------------------------------------------------- flashcards
def flashcards(cards, key, log, source="study_flashcards", minutes=5) -> bool:
    """Open a card, check the answer, grade yourself.  When every card is graded the result is saved."""
    flag = f"{key}:logged"
    done = bool(st.session_state.get(flag))
    graded = {}
    for i, card in enumerate(cards):
        with st.expander(f"Card {i + 1}: {card['question']}"):
            st.write(card["answer"])
            g = st.radio("How did you do?", list(GRADES), index=None, horizontal=True,
                         key=f"{key}:g{i}", disabled=done)
            if g:
                graded[i] = GRADES[g]
    if done:
        st.success("Result saved automatically.")
        return False
    if not graded:
        st.caption("Try to answer each card in your head first, then open it and grade yourself. "
                   "Your result is saved automatically when you finish.")
        return False
    finish = len(graded) == len(cards)
    if not finish:
        finish = st.button("Finish with the cards I've done", key=f"{key}:finish")
    if finish:
        score = sum(graded.values()) / len(graded)
        res = log(score, "quiz", source, minutes)
        if res:
            st.session_state[flag] = True
            _saved(res, f"Saved: {score:.0%} on {len(graded)} card{'s' if len(graded) != 1 else ''}")
            return True
        st.error("Could not save to the database. Is MongoDB running?")
    return False


# ------------------------------------------------------------------------------- quiz
def quiz(questions, key, log, source="study_quiz", minutes=5) -> bool:
    result = st.session_state.get(f"{key}:result")
    for i, q in enumerate(questions):
        st.markdown(f"**{i + 1}. {q['question']}**")
        options = [f"{letter}) {text}" for letter, text in q["options"]]
        st.radio("Your answer", options, index=None, key=f"{key}:q{i}",
                 label_visibility="collapsed", disabled=result is not None)
        if result is not None:
            ok = result["right"][i]
            (st.success if ok else st.error)(
                ("Correct. " if ok else f"Not quite - the answer is {q['correct']}. ") + q["explanation"])
    if result is not None:
        st.metric("Your score", f"{sum(result['right'])} / {len(questions)}")
        st.caption("Saved automatically - your recall estimate is updated.")
        return False
    if st.button("Check answers", key=f"{key}:check"):
        right = []
        for i, q in enumerate(questions):
            choice = st.session_state.get(f"{key}:q{i}")
            right.append(bool(choice) and choice[0] == q["correct"])
        score = sum(right) / len(questions)
        res = log(score, "quiz", source, minutes)
        st.session_state[f"{key}:result"] = {"right": right, "score": score, "saved": bool(res)}
        if res:
            _saved(res, f"Saved: {sum(right)} of {len(questions)}")
        st.rerun()
    return False


# ---------------------------------------------------------------------------- coding
def coding(task: dict, key, log, source="study_coding", minutes=10) -> bool:
    """task = generate.parse_coding_task(...).  Logs 1.0 when the output matches; otherwise the student
    decides (the expected output was written by the AI and can be wrong)."""
    flag = f"{key}:logged"
    st.markdown("**Problem**")
    st.write(task["problem"])
    if task["input"]:
        st.markdown("**Input**")
        st.code(task["input"], language="python")
    if task["hint"]:
        with st.expander("Show a hint"):
            st.write(task["hint"])
    code = st.text_area("Your solution (Python)", value="# write your solution here\n", height=220,
                        key=f"{key}:code")
    run = st.button("Run code", key=f"{key}:run", type="primary")
    if run:
        st.session_state[f"{key}:run_result"] = run_code(code)
    res = st.session_state.get(f"{key}:run_result")
    logged = False
    if res:
        st.markdown("**Output**")
        if res["error"]:
            st.error(res["error"])
        if res["stdout"]:
            st.code(res["stdout"], language="text")
        elif not res["error"]:
            st.caption("(the program printed nothing)")
        done = bool(st.session_state.get(flag))
        match = bool(task["expected"] and res["success"] and compare_output(res["stdout"], task["expected"])["passed"])
        if done:
            st.success("Saved automatically.")
        elif match:
            out = log(1.0, "coding", source, minutes)
            if out:
                st.session_state[flag] = True
                st.success("Matches the expected output - saved automatically.")
                _saved(out, "Saved: solved")
                logged = True
        elif res["success"]:
            st.warning("This does not match the expected output. The expected output was written by the AI "
                       "and can be wrong - you decide:")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("My output is correct", key=f"{key}:mine"):
                    if log(0.85, "coding", source, minutes):
                        st.session_state[flag] = True
                        logged = True
                        st.rerun()
            with c2:
                if st.button("I couldn't solve it", key=f"{key}:gaveup"):
                    if log(0.30, "coding", source, minutes):
                        st.session_state[flag] = True
                        logged = True
                        st.rerun()
            if task["expected"]:
                with st.expander("Show the expected output"):
                    st.code(task["expected"], language="text")
    return logged
