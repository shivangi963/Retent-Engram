import streamlit as st
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# =============================================================================
# PAGE CONFIG — must be the FIRST streamlit call in the file
# =============================================================================

st.set_page_config(
    page_title="Retent Engram",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)


# =============================================================================
# SYSTEM HEALTH CHECK FUNCTIONS
# =============================================================================

def check_mongodb() -> tuple:
    """
    Checks if MongoDB is reachable.

    Returns:
        (bool, str): (is_running, status_message)
    """
    try:
        from backend.db import db
        db.command("ping")   # lightweight ping command
        return True, "Connected"
    except Exception as e:
        return False, f"Cannot connect: {str(e)[:60]}"


def check_ollama() -> tuple:
    """Checks if Ollama server is running."""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            models = r.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            has_mistral = any("mistral" in n for n in model_names)
            if has_mistral:
                return True, "Running + Mistral loaded"
            return True, "Running (mistral not pulled yet — run: ollama pull mistral)"
        return False, "Running but returned error"
    except Exception:
        return False, "Not running (run: ollama serve)"


def check_faiss_index() -> tuple:
    """Checks if the FAISS knowledge base has been built."""
    index_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "faiss_index", "index.faiss"
    )
    if os.path.exists(index_path):
        size_kb = os.path.getsize(index_path) / 1024
        return True, f"Built ({size_kb:.0f} KB)"
    return False, "Not built (run: python rag/ingest.py)"


def check_ml_model() -> tuple:
    """Checks if the ML model has been trained."""
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "recall_model.pkl"
    )
    metadata_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "model_metadata.json"
    )
    if os.path.exists(model_path):
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path) as f:
                meta = json.load(f)
            model_name = meta.get("active_model", "unknown")
            auc = meta.get(model_name, {}).get("auc", None)
            if auc:
                return True, f"{model_name} (AUC: {auc:.3f})"
        return True, "Trained"
    return False, "Not trained (run: python scripts/run_training.py)"


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    """
    <div style='text-align: center; padding: 20px 0 10px 0;'>
        <h1 style='font-size: 3em;'>🧠 Retent Engram</h1>
        <p style='font-size: 1.2em; color: #6C63FF;'>
            Personalised Cognitive Recall Assistor
        </p>
        <p style='color: #888; font-size: 0.95em;'>
            Track your study events · Predict forgetting · Generate revision content
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# =============================================================================
# USER LOGIN
# =============================================================================

# Initialize session state if not set
if "user_id" not in st.session_state:
    st.session_state.user_id   = ""
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

st.subheader("👤 Login / Set User")

col1, col2 = st.columns(2)
with col1:
    name_input = st.text_input(
        "Your name",
        value=st.session_state.user_name,
        placeholder="e.g. Shivangi"
    )
with col2:
    uid_input = st.text_input(
        "User ID (no spaces)",
        value=st.session_state.user_id,
        placeholder="e.g. shivangi_01"
    )

if st.button("✅ Set User", type="primary"):
    if not name_input.strip() or not uid_input.strip():
        st.warning("Please enter both your name and user ID.")
    elif " " in uid_input.strip():
        st.warning("User ID must not contain spaces.")
    else:
        # Save to session state (persists across all pages)
        st.session_state.user_id   = uid_input.strip().lower()
        st.session_state.user_name = name_input.strip()

        # Auto-create user in MongoDB if new
        try:
            from backend.db import get_or_create_user
            get_or_create_user(uid_input.strip(), name_input.strip())
        except Exception:
            pass  # MongoDB might not be running — don't crash home page

        st.success(f"Welcome, **{name_input}**! Use the sidebar to navigate.")

# Show who is logged in
if st.session_state.user_id:
    st.info(
        f"Logged in as **{st.session_state.user_name}** "
        f"(`{st.session_state.user_id}`)"
    )

st.divider()


# =============================================================================
# SYSTEM HEALTH PANEL
# =============================================================================

st.subheader("🔧 System Health")
st.caption("All four components must be ready for full functionality.")

h1, h2, h3, h4 = st.columns(4)

with h1:
    ok, msg = check_mongodb()
    if ok:
        st.success(f"**MongoDB**\n{msg}")
    else:
        st.error(f"**MongoDB**\n{msg}")

with h2:
    ok, msg = check_ollama()
    if ok:
        st.success(f"**Ollama**\n{msg}")
    else:
        st.warning(f"**Ollama**\n{msg}")

with h3:
    ok, msg = check_faiss_index()
    if ok:
        st.success(f"**Knowledge Base**\n{msg}")
    else:
        st.warning(f"**Knowledge Base**\n{msg}")

with h4:
    ok, msg = check_ml_model()
    if ok:
        st.success(f"**ML Model**\n{msg}")
    else:
        st.warning(f"**ML Model**\n{msg}")

st.divider()


# =============================================================================
# NAVIGATION GUIDE
# =============================================================================

st.subheader("📍 Navigation Guide")

nav_data = [
    ("📝", "1 Log Event",    "Record a study session — reading, quiz, or coding practice"),
    ("📊", "2 Dashboard",   "View recall scores, priority charts, and forgetting curves"),
    ("📖", "3 Review",      "Generate AI flashcards, summaries, quizzes, and coding tasks"),
    ("📋", "4 Queue",       "Today's review queue sorted by urgency — what to study now"),
    ("📅", "5 History",     "Full event timeline, score trends, and CSV export"),
    ("⚙️",  "6 Settings",   "Change display name, reset data, manage daily goals"),
    ("ℹ️",  "7 About",      "Project info, architecture, tech stack, and team details"),
]

for emoji, page, desc in nav_data:
    c1, c2 = st.columns([1, 4])
    with c1:
        st.markdown(f"### {emoji} **{page}**")
    with c2:
        st.markdown(f"<br><span style='color:#aaa'>{desc}</span>", unsafe_allow_html=True)

st.divider()


# =============================================================================
# QUICK STATS (if logged in and MongoDB running)
# =============================================================================

if st.session_state.user_id:
    st.subheader(f"📈 Quick Stats — {st.session_state.user_name}")

    try:
        from backend.db import get_total_events_count, get_recall_scores
        from backend.scheduler import get_streak

        uid    = st.session_state.user_id
        total  = get_total_events_count(uid)
        scores = get_recall_scores(uid)
        streak = get_streak(uid)

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.metric("Sessions Logged", total)
        with s2:
            st.metric("Concepts Tracked", len(scores))
        with s3:
            needs = sum(1 for s in scores if s.get("recall_score", 100) < 65)
            st.metric("Need Review", needs)
        with s4:
            st.metric("🔥 Streak", f"{streak} days")

    except Exception:
        st.caption("Log in and connect MongoDB to see your stats.")


# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.markdown(
    """
    <div style='text-align:center; color:#555; font-size:0.85em;'>
        Retent Engram · VTU Major Project 2026–2027 ·
        Bangalore College of Engineering & Technology ·
        Shivangi Shukla & S. Harini
    </div>
    """,
    unsafe_allow_html=True
)