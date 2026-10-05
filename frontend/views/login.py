"""Sign-in screen (shown by main.py until a learner is selected)."""
import streamlit as st

from frontend.components.theme import LOGO_MARK, html, icon, TONES

# hide the (empty) sidebar on the login screen
html("""<style>

/* Hide sidebar */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"] {
    display: none !important;
}

/* Normal page width */
.block-container {
    max-width: 980px !important;
}

/* EXACT vertical + horizontal centering */
[data-testid="stHorizontalBlock"] {
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    width: 980px !important;
    max-width: calc(100vw - 40px) !important;
    margin: 0 !important;
    transform: translate(-50%, -50%) !important;
    z-index: 10 !important;
}

</style>""")

left, right = st.columns([1.05, 1], gap="large", vertical_alignment="center")

with left:
    html(f"""
    <div style="padding:1rem 0">
      <div>{LOGO_MARK.replace('width="38" height="38"', 'width="64" height="64"')}</div>
      <h1 style="font-size:2.7rem;font-weight:800;letter-spacing:-.035em;margin:1.1rem 0 .4rem;color:#0F1F3D;line-height:1.08">
        Remember more.<br>Revise smarter.</h1>
    <!--
      <p style="color:#64748B;font-size:1.05rem;margin:0 0 1.4rem;line-height:1.55">
        Retent Engram predicts what you're about to forget and builds the right revision material — right on time.</p>
      <div class="rt-feat"><div class="rt-tile" style="background:{TONES['blue'][1]};color:{TONES['blue'][0]}">{icon('trending-up',20)}</div>
        <div><b>Predicts forgetting</b><span>Concept-wise recall scores from your own study history.</span></div></div>
      <div class="rt-feat"><div class="rt-tile" style="background:{TONES['purple'][1]};color:{TONES['purple'][0]}">{icon('sparkles',20)}</div>
        <div><b>Generates review material</b><span>Flashcards, summaries, quizzes and coding tasks from your notes.</span></div></div>
      <div class="rt-feat"><div class="rt-tile" style="background:{TONES['amber'][1]};color:{TONES['amber'][0]}">{icon('flame',20)}</div>
        <div><b>Keeps you consistent</b><span>A daily queue, goals and streaks so nothing slips.</span></div></div>
    -->
    </div>""")

with right:
    with st.container(key="card_login"):
        html('<div style="font-size:1.35rem;font-weight:800;color:#0F1F3D;letter-spacing:-.02em">Welcome</div>'
             '<div style="color:#64748B;font-size:.92rem;margin:.25rem 0 1rem">Sign in with your name and learner ID. '
             'A new ID creates a new profile.</div>')
        with st.form("login_form", border=False):
            name = st.text_input("Your name", placeholder="e.g. name")
            uid = st.text_input("Learner ID", placeholder="e.g. user_id", help="No spaces. Used to keep your data separate.")
            go = st.form_submit_button("Continue", type="primary", width="stretch")

    if go:
        name, uid = name.strip(), uid.strip()
        if not name or not uid:
            st.warning("Please enter both your name and a learner ID.")
        elif " " in uid:
            st.warning("Learner ID must not contain spaces.")
        else:
            try:
                from backend.db import db, get_or_create_user
                db.command("ping")
                get_or_create_user(uid, name)
            except Exception as e:
                st.error("Couldn't reach MongoDB. Start it and try again.")
                st.caption(f"Details: {str(e)[:120]}")
            else:
                st.session_state.user_id = uid.lower()
                st.session_state.user_name = name
                st.rerun()
