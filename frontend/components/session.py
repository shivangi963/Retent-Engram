"""
frontend/components/session.py
==============================
Who is signed in?  Every page asks here.

If nobody is signed in (for example you opened a page file directly with
`streamlit run frontend/pages/log_session.py`, or the browser tab was refreshed and the
session was lost) the page shows a small sign-in box instead of a dead end.
"""
import streamlit as st

USER_KEYS = ("user_id", "username", "user")          # keys your main file may have used to store the login


def current_user():
    for k in USER_KEYS:
        if st.session_state.get(k):
            return str(st.session_state[k])
    return None


def _sign_in():
    name = (st.session_state.get("_signin_id") or "").strip().lower()
    if name:
        st.session_state["user_id"] = name


def require_user() -> str:
    """The signed-in user id, or a sign-in box (and the page stops here)."""
    uid = current_user()
    if uid:
        return uid
    st.info("Sign in to continue.")
    st.text_input("Your user ID", key="_signin_id", placeholder="e.g. user1")
    clicked = st.button("Continue", key="_signin_go", type="primary", on_click=_sign_in)
    if clicked and not (st.session_state.get("_signin_id") or "").strip():
        st.warning("Type your user ID first.")
    st.stop()
