#  Retent Engram

**A personal project I built to stop forgetting what I study.**

Ever revised for an exam, felt confident, then blanked on the same topic two weeks later?
That kept happening to me. So I built a system that tracks what I study, predicts when I'll
forget it, and generates personalised revision content right before I do.

---

## What It Does

Instead of generic flashcard apps or fixed revision schedules, this app is personalized
to *my* forgetting patterns:

- **Logs study sessions** — every time I read, attempt a quiz, or do a coding problem,
  I log the concept, how long it took, and how well I did
- **Predicts recall** — uses the Ebbinghaus forgetting curve and an XGBoost model
  trained on my own event history to estimate how much I remember *right now*
- **Builds a daily queue** — ranks concepts by urgency (low recall + long gap + high difficulty)
  and shows me exactly what to study today
- **Generates study material** — uses Mistral 7B running locally to generate flashcards,
  summaries, MCQ quizzes, and coding tasks tailored to my current recall level
- **Tracks streaks and goals** — keeps me honest about daily review habits

---

## Tech Stack

| What | How |
|---|---|
| Frontend | Streamlit |
| Database | MongoDB |
| ML model | XGBoost + Logistic Regression (scikit-learn) |
| Experiment tracking | MLflow |
| Vector search | FAISS |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Local LLM | Mistral 7B via Ollama |
| Data processing | Pandas, NumPy, SciPy |
| Visualisation | Plotly |

Everything runs locally — no cloud APIs, no subscriptions, no data leaving my machine.

---

## How It Works (the interesting parts)

### Recall Scoring
Each concept gets a recall score (0–100%) computed from:
- Time since last review
- Number of total reviews
- Recent performance scores
- Success streak (last 3 sessions)

These feed into a modified Ebbinghaus decay formula first, then into a trained
XGBoost classifier once enough events are logged. The model predicts the probability
of recall at the next review and converts it to a score.

### Urgency Scoring
The daily queue uses a weighted urgency formula:

```
urgency = (0.5 × low_recall) + (0.3 × long_gap) + (0.2 × high_difficulty)
```

This means a hard concept I haven't touched in a week with low recall shoots to the
top of the queue — exactly what I need reviewed most urgently.

### RAG Pipeline
When I need to study a concept, the app:
1. Searches a local FAISS vector index built from my own notes
2. Retrieves the 3–4 most relevant text chunks
3. Sends them as context to Mistral 7B with a difficulty-adapted prompt
4. Returns a flashcard, summary, quiz, or coding task grounded in my notes

This prevents hallucination — the LLM can only use what's in my notes.

---

## Running It Locally

**Requirements:** Python 3.11+, MongoDB, Ollama

```bash
# Clone and set up
git clone https://github.com/shivangi963/Retent-Engram.git
cd Retent-Engram
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env: MONGO_URI=mongodb://localhost:27017

# Build the knowledge base (indexes concept notes for RAG)
python rag/ingest.py

# Train the ML model
python scripts/run_training.py

# Run the app
streamlit run frontend/main.py
```

For the LLM to work, install [Ollama](https://ollama.com) separately and run:
```bash
ollama pull mistral
ollama serve
```


## What I Learned Building This

- **MongoDB aggregation pipelines** — writing `$group`, `$sort`, `$match` to derive
  stats from raw event logs without pulling everything into Python first
- **FAISS vector search** — building and querying a local vector index, understanding
  why normalized embeddings let you use inner product instead of cosine similarity
- **Prompt engineering** — how much prompt structure matters for consistent LLM output;
  specifying exact output format (QUESTION:/ANSWER:) made parsing reliable
- **XGBoost with MLflow** — tracking experiments properly so I could actually compare
  runs instead of guessing which hyperparameters worked
- **Streamlit session state** — managing user login across multiple pages without
  a backend auth system
- **Ebbinghaus memory science** — the math behind spaced repetition and why half-life
  based models outperform fixed-interval schedules

---

*Built this over several months while studying for exams. The irony of using it to
study the same CS topics it's built on (DBMS, OS, DSA) was not lost on me.*