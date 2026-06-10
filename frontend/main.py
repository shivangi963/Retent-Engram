import streamlit as st

st.set_page_config(
    page_title="Retent Engram",
    layout="centered"
)

st.title("Retent Engram")
st.subheader("Personalised Cognitive Recall Assistor")

st.markdown("""
Welcome! Use the sidebar to navigate:

- **Log Event** — Record a study session
- **Dashboard** — View your recall scores *(coming in Phase 3)*
- **Review Queue** — Today's review list *(coming in Phase 5)*
""")

st.info("Start by logging a study event from the sidebar.")