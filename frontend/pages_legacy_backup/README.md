These files are the pre-redesign pages, kept only for reference.
The live app no longer uses Streamlit's automatic `pages/` auto-discovery —
frontend/main.py now builds the sidebar itself with st.navigation() over the
screens in frontend/views/. This folder is intentionally NOT named `pages`
so Streamlit doesn't pick it up a second time.
