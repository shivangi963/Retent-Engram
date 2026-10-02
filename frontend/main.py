"""
frontend/main.py — application shell
====================================
Run with:  streamlit run frontend/main.py

This file is only the *router*: page config, theme, login gate, sidebar and
navigation. The actual screens live in frontend/views/.
"""
import os
import sys

import streamlit as st

# make `backend`, `rag` and `frontend` importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from frontend.components.theme import inject_css, LOGO_MARK, html
from frontend.components.data import show_flash

ASSETS = os.path.join(os.path.dirname(__file__), "assets")

st.set_page_config(
    page_title="Retent Engram",
    page_icon=os.path.join(ASSETS, "favicon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ── session defaults ─────────────────────────────────────────────────────────
st.session_state.setdefault("user_id", "")
st.session_state.setdefault("user_name", "")


def sign_out():
    for k in list(st.session_state.keys()):
        del st.session_state[k]


# ── not signed in → login screen only ────────────────────────────────────────
if not st.session_state.user_id:
    login = st.Page("views/login.py", title="Sign in", url_path="login", default=True)
    st.navigation([login], position="hidden").run()
    st.stop()

# ── signed in → full app ─────────────────────────────────────────────────────
P = {
    "dashboard": st.Page("views/dashboard.py",   title="Dashboard",   icon=":material/space_dashboard:", url_path="dashboard", default=True),
    "today":     st.Page("views/today.py",       title="Today",       icon=":material/today:",           url_path="today"),
    "study":     st.Page("views/study.py",       title="Study",       icon=":material/auto_stories:",    url_path="study"),
    "log":       st.Page("views/log_session.py", title="Log Session", icon=":material/edit_note:",       url_path="log-session"),
    "history":   st.Page("views/history.py",     title="History",     icon=":material/insights:",        url_path="history"),
    "notes":     st.Page("views/notes.py",       title="My Notes",    icon=":material/upload_file:",     url_path="notes"),
    "settings":  st.Page("views/settings.py",    title="Settings",    icon=":material/settings:",        url_path="settings"),
}
SECTIONS = [
    ("Learn",  ["dashboard", "today", "study"]),
    ("Track",  ["log", "history"]),
    ("Manage", ["notes", "settings"]),
]

nav = st.navigation(list(P.values()), position="hidden")

name = st.session_state.user_name or st.session_state.user_id
with st.sidebar:
    html(f"""
      <div class="rt-brand">{LOGO_MARK}
        <div><div class="rt-brand-n">Retent Engram</div>
             <div class="rt-brand-s">Cognitive recall assistor</div></div></div>
      <div class="rt-user"><div class="rt-avatar">{name[:1].upper()}</div>
        <div><div class="rt-user-n">{name}</div>
             <div class="rt-user-i">@{st.session_state.user_id}</div></div></div>""")
    for label, keys in SECTIONS:
        html(f'<div class="rt-navlabel">{label}</div>')
        for k in keys:
            st.page_link(P[k])
    html('<div style="height:.6rem"></div>')
    st.button("Sign out", icon=":material/logout:", on_click=sign_out, key="signout")

show_flash()
nav.run()
