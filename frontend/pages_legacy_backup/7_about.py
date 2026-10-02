"""
frontend/pages/7_about.py
===========================
PHASE 7 — About Page

PURPOSE
-------
Professional project presentation page for VTU submission and demo.

SECTIONS:
  1. Project overview and description
  2. Architecture diagram (embedded SVG)
  3. Tech stack table
  4. Team and institution details
  5. Future work
"""

import streamlit as st
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# =============================================================================
# PAGE SETUP
# =============================================================================

st.set_page_config(
    page_title="About — Retent Engram",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About Retent Engram")

# =============================================================================
# PROJECT OVERVIEW
# =============================================================================

st.markdown("""
## Retent Engram: Personalised Cognitive Recall Assistor

**Retent Engram** is an AI-powered study companion that predicts concept-wise
forgetting probability for each learner and recommends timely, personalised
revision interventions — before recall failure occurs.

The system addresses a core limitation of existing educational tools:
they record learning activity but do not convert it into a predictive
retention pipeline with automatic content generation.

### Core Idea

> **"Study smarter, not harder — review the right concept at the right time
> with the right material."**

Instead of fixed study schedules that treat all students the same,
Retent Engram personalises the revision experience by:
- Tracking how each student performs on each concept over time
- Predicting when they are likely to forget using ML models
- Generating flashcards, summaries, quizzes, and coding tasks on demand
- Building an urgency queue so students always know what to study today
""")

st.divider()


# =============================================================================
# ARCHITECTURE DIAGRAM (SVG inline)
# =============================================================================

st.subheader("🏗️ System Architecture")

st.markdown("""
```
┌──────────────────────────────────────────────────────────────────────┐
│                         STUDENT (Browser)                            │
│          Reads / Solves / Takes Quiz / Reviews Content               │
└──────────────────────┬───────────────────────────────────────────────┘
                       │ HTTP
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                                │
│  main.py │ 1_log_event │ 2_dashboard │ 3_review │ 4_queue           │
│           5_history   │ 6_settings  │ 7_about                       │
└───────┬──────────────────────────────────────┬────────────────────────┘
        │                                      │
        ▼                                      ▼
┌───────────────────┐               ┌──────────────────────────────────┐
│   MongoDB         │               │        BACKEND PYTHON MODULES    │
│   ─────────────   │               │  ─────────────────────────────── │
│   events          │◄──────────────│  backend/db.py                   │
│   users           │               │  backend/ml/features.py          │
│   concepts        │               │  backend/ml/scorer.py            │
│   recall_scores   │◄──────────────│  backend/ml/pipeline.py (Phase4) │
│   generated_      │               │  backend/scheduler.py  (Phase5)  │
│   content         │               └──────────────────────────────────┘
└───────────────────┘                           │
                                                ▼
                              ┌─────────────────────────────────┐
                              │         ML PIPELINE             │
                              │  ─────────────────────────────  │
                              │  XGBoost / Logistic Regression  │
                              │  (trained on event history)     │
                              │  models/recall_model.pkl        │
                              └─────────────────────────────────┘
                                                │
                                                ▼
                              ┌─────────────────────────────────┐
                              │         RAG PIPELINE            │
                              │  ─────────────────────────────  │
                              │  FAISS Vector Index             │
                              │  sentence-transformers          │
                              │  Mistral 7B via Ollama          │
                              │  rag/ingest.py + generate.py   │
                              └─────────────────────────────────┘
```
""")

st.divider()


# =============================================================================
# TECH STACK TABLE
# =============================================================================

st.subheader("🛠️ Technology Stack")

import pandas as pd

tech_data = [
    ("Frontend",           "Streamlit 1.32",            "Lightweight, Python-native, fast to build"),
    ("Backend Logic",      "Python 3.11 modules",        "No server overhead for MVP"),
    ("Database",           "MongoDB",                    "Flexible schema, excellent for event logs"),
    ("ML — Baseline",      "Logistic Regression",        "Interpretable baseline model"),
    ("ML — Main",          "XGBoost 2.0",                "Non-linear tabular prediction"),
    ("Survival Model",     "lifelines",                  "Time-to-forget modeling"),
    ("Experiment Tracking","MLflow",                     "Track model runs, metrics, and artifacts"),
    ("Vector Store",       "FAISS (faiss-cpu)",          "Local vector similarity search, no cloud"),
    ("Embeddings",         "sentence-transformers",      "all-MiniLM-L6-v2, 384-dim, runs on CPU"),
    ("LLM",                "Mistral 7B Q4 via Ollama",   "Local inference, no API cost"),
    ("Data Processing",    "Pandas, NumPy, SciPy",       "Feature engineering and data wrangling"),
    ("Visualisation",      "Plotly 5.19",                "Interactive charts in Streamlit"),
    ("Versioning",         "Git + GitHub",               "Source control and collaboration"),
    ("Environment",        "Python venv",                "Isolated dependencies"),
]

tech_df = pd.DataFrame(tech_data, columns=["Layer", "Technology", "Reason"])
st.dataframe(tech_df, use_container_width=True, hide_index=True)

st.divider()


# =============================================================================
# PHASE SUMMARY TABLE
# =============================================================================

st.subheader("📅 Development Phases")

phases_data = [
    ("Phase 0", "Environment Setup",         "MongoDB connection, empty Streamlit app"),
    ("Phase 1", "Event Logger",              "Study session logging with MongoDB storage"),
    ("Phase 2", "Feature Extraction",        "Ebbinghaus decay formula, recall scoring"),
    ("Phase 3", "Dashboard",                 "Charts, priority table, forgetting curve"),
    ("Phase 4", "ML Model",                  "XGBoost + Logistic Regression, MLflow tracking"),
    ("Phase 5", "Review Scheduler",          "Urgency scoring, snooze, streaks, daily queue"),
    ("Phase 6", "RAG + LLM Generation",      "FAISS retrieval + Mistral content generation"),
    ("Phase 7", "Polish + Final Submission", "UI theme, history page, settings, about, README"),
]

phases_df = pd.DataFrame(phases_data, columns=["Phase", "Name", "Deliverable"])
st.dataframe(phases_df, use_container_width=True, hide_index=True)

st.divider()


# =============================================================================
# TEAM AND INSTITUTION
# =============================================================================

st.subheader("👩‍💻 Team")

t1, t2 = st.columns(2)
with t1:
    st.markdown("""
    **Submitted By:**
    - Shivangi Shukla (1BC23CS056)
    - S. Harini (1BC23CS049)

    **Under the Guidance of:**
    Mrs. Swathi Priya, M.E
    """)

with t2:
    st.markdown("""
    **Institution:**
    Bangalore College of Engineering & Technology
    Chandapura, Bengaluru — 560099

    Affiliated to Visvesvaraya Technological University, Belagavi
    Department of Computer Science and Engineering
    6th Semester, 2026–2027
    """)

st.divider()


# =============================================================================
# FUTURE WORK
# =============================================================================

st.subheader("🚀 Future Work")

st.markdown("""
The following enhancements are planned for future iterations:

- **Multi-user support** — Separate dashboards for multiple students with instructor view
- **PDF ingestion** — Auto-parse VTU notes PDFs into the FAISS knowledge base
- **Concept dependency graph** — Flag prerequisite concepts when a concept is urgent
- **Spaced content repetition** — Re-show flashcards after 2 days if student got wrong
- **Code execution sandbox** — Run coding tasks directly inside the app
- **Mobile-friendly layout** — Responsive design using Streamlit's column system
- **Export to PDF** — Download a personalised weak-concept report
- **Voice input** — Log a study event by speaking instead of typing
- **Adaptive prompting** — Use thumbs-down ratings to improve future LLM prompts
- **LSTM sequence model** — Capture temporal learning patterns for better predictions
""")

st.divider()


# =============================================================================
# REFERENCES
# =============================================================================

with st.expander("📚 References"):
    st.markdown("""
1. Balazevic, I., Allen, C., and Hospedales, T. M., *"TuckER: Tensor Factorization for
   Knowledge Graph Completion,"* AMLW, 2019.
2. Personal Knowledge Decay Predictor (PKDP) — project planning document.
3. Ebbinghaus, H., *"Memory: A Contribution to Experimental Psychology,"* 1885.
4. Chen, T. and Guestrin, C., *"XGBoost: A Scalable Tree Boosting System,"* KDD, 2016.
5. Reimers, N. and Gurevych, I., *"Sentence-BERT: Sentence Embeddings using Siamese
   BERT-Networks,"* EMNLP, 2019.
6. Johnson, J., Douze, M., and Jégou, H., *"Billion-scale similarity search with GPUs,"*
   IEEE Big Data, 2021.
7. Jiang, A. Q. et al., *"Mistral 7B,"* arXiv:2310.06825, 2023.
    """)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(
    """
    <div style='text-align:center; color:#555; font-size:0.85em; margin-top:40px;'>
        Retent Engram · VTU Major Project · 2026–2027 ·
        Built with ❤️ using Python, Streamlit, MongoDB, and Mistral
    </div>
    """,
    unsafe_allow_html=True
)