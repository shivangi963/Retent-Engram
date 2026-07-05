
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import requests
import numpy as np
from datetime import datetime, timezone

from backend.db import get_collection
from rag.ingest import load_index_and_model, is_index_available


# =============================================================================
# CONSTANTS
# =============================================================================

# Ollama server URL — runs locally
OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "mistral"   # ollama pull mistral

# Difficulty thresholds (must match scorer.py)
BEGINNER_THRESHOLD    = 40.0
INTERMEDIATE_THRESHOLD = 65.0

# How many chunks to retrieve from FAISS
TOP_K_CHUNKS = 4

# How many chunks to include in the prompt context
# (we retrieve 4 but might use only top 3 to keep prompt shorter)
CONTEXT_CHUNKS_TO_USE = 3

# Generation timeout in seconds
# Mistral 7B on CPU/laptop can be slow — 120 seconds is safe
GENERATION_TIMEOUT = 120

# Content types
FLASHCARD   = "flashcard"
SUMMARY     = "summary"
QUIZ        = "quiz"
CODING_TASK = "coding_task"

VALID_CONTENT_TYPES = [FLASHCARD, SUMMARY, QUIZ, CODING_TASK]


# =============================================================================
# STEP 1 — DETERMINE DIFFICULTY LEVEL
# =============================================================================

def get_difficulty_level(recall_score: float) -> str:
    """
    Returns content difficulty based on student's recall score.

    LOGIC:
      If student barely remembers → generate BEGINNER content
        (basic definitions, simple questions, no traps)
      If student partially remembers → generate INTERMEDIATE content
        (application questions, comparisons, moderate depth)
      If student remembers well → generate ADVANCED content
        (edge cases, tricky variations, deep understanding required)

    WHY ADAPT DIFFICULTY?
      Generating the same content regardless of recall wastes the student's time:
      - Too easy for a student who remembers 80% → boring, no learning
      - Too hard for a student who remembers 20% → overwhelming, discouraging
      Adaptive difficulty maximizes learning efficiency.

    Args:
        recall_score: float 0–100

    Returns:
        str: "beginner", "intermediate", or "advanced"
    """
    if recall_score < BEGINNER_THRESHOLD:
        return "beginner"
    elif recall_score < INTERMEDIATE_THRESHOLD:
        return "intermediate"
    else:
        return "advanced"


# =============================================================================
# STEP 2 — RETRIEVE RELEVANT CHUNKS FROM FAISS
# =============================================================================

def retrieve_chunks(concept_id: str, query_text: str = None,
                    top_k: int = TOP_K_CHUNKS) -> list:
    """
    Queries the FAISS index to find the most relevant text chunks
    for a given concept and optional query.

    HOW IT WORKS:
      1. Build a query string (concept name or custom query)
      2. Embed the query using sentence-transformers (same model as ingest.py)
      3. Search FAISS for top-k most similar chunk vectors
      4. Return those chunks' text from metadata

    QUERY STRATEGY:
      If query_text is provided, use it directly.
      Otherwise, use the concept_id as the query. FAISS will find
      the chunks most relevant to the concept name itself.

    FILTERING BY CONCEPT:
      We return ALL top-k results but mark their concept_id.
      The caller can optionally filter to only same-concept chunks.
      However, cross-concept retrieval can sometimes help:
      Example: "OS deadlock" query might retrieve from "process_mgmt" notes
               which describes the same topic from a different angle.

    SIMILARITY SCORE:
      FAISS returns inner product scores (= cosine similarity for normalized vectors).
      Score range: -1 to +1
      Score > 0.5: very relevant
      Score > 0.3: somewhat relevant
      Score < 0.2: probably not relevant

    Args:
        concept_id: which concept to retrieve context for
        query_text: optional custom query (if None, uses concept_id)
        top_k:      how many chunks to retrieve

    Returns:
        list of dicts: [{"text": ..., "concept_id": ..., "similarity": ...}]
        Returns empty list if index not available.
    """
    if not is_index_available():
        print("⚠️  FAISS index not found. Run: python rag/ingest.py first.")
        return []

    # Load index, metadata, and model (cached after first call)
    index, metadata, model = load_index_and_model()

    if index is None:
        return []

    # Build query string
    if query_text is None:
        # Default: search for the concept name
        # This finds the most central/definitional chunks
        query_text = concept_id.replace("_", " ")

    # Embed the query (same process as chunked texts in ingest.py)
    query_vector = model.encode(
        [query_text],
        normalize_embeddings=True
    ).astype("float32")

    # Search FAISS index
    # Returns: distances shape (1, top_k), indices shape (1, top_k)
    # distances[0] = similarity scores, indices[0] = chunk indices in metadata
    distances, indices = index.search(query_vector, k=min(top_k, index.ntotal))

    # Collect results
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:   # FAISS returns -1 for empty slots
            continue
        chunk = metadata[idx]
        results.append({
            "text":        chunk["text"],
            "concept_id":  chunk["concept_id"],
            "chunk_id":    chunk["chunk_id"],
            "similarity":  float(dist),
            "chunk_index": chunk["chunk_index"]
        })

    return results


def build_context_string(chunks: list, max_chunks: int = CONTEXT_CHUNKS_TO_USE) -> str:
    """
    Combines retrieved chunks into a single context string for the LLM prompt.

    FORMAT:
      [Source: os, similarity: 0.82]
      Deadlock occurs when processes wait indefinitely...

      [Source: os, similarity: 0.71]
      Four necessary conditions for deadlock...

    WHY INCLUDE SIMILARITY SCORE?
      For debugging. When content seems off-topic, you can check
      if similarity was low (< 0.3 = retrieval probably failed).

    Args:
        chunks:     list of chunk dicts from retrieve_chunks()
        max_chunks: how many chunks to include in context

    Returns:
        str: formatted context string
    """
    if not chunks:
        return "No context available."

    context_parts = []
    for chunk in chunks[:max_chunks]:
        part = (
            f"[Source: {chunk['concept_id']}, "
            f"relevance: {chunk['similarity']:.2f}]\n"
            f"{chunk['text']}"
        )
        context_parts.append(part)

    return "\n\n".join(context_parts)


# =============================================================================
# STEP 3 — BUILD PROMPTS FOR EACH CONTENT TYPE
# =============================================================================

def build_flashcard_prompt(concept_name: str, context: str,
                           difficulty: str) -> str:
    """
    Builds a prompt to generate ONE high-quality flashcard.

    PROMPT DESIGN PRINCIPLES:
      - Tell the LLM exactly what format to use (QUESTION: / ANSWER:)
      - Specify difficulty explicitly
      - Tell it to use ONLY the provided context (prevents hallucination)
      - Ask for concise output (prevents rambling)

    EXAMPLE OUTPUT FROM MISTRAL:
      QUESTION: What are the four necessary conditions for deadlock?
      ANSWER: Mutual Exclusion, Hold and Wait, No Preemption, and Circular Wait.
              All four must hold simultaneously for deadlock to occur.

    Args:
        concept_name: display name like "Operating Systems"
        context:      retrieved chunks as string
        difficulty:   "beginner", "intermediate", or "advanced"

    Returns:
        str: complete prompt to send to Mistral
    """
    difficulty_instructions = {
        "beginner": (
            "Create a BASIC flashcard testing fundamental definitions. "
            "Question should ask for a simple definition or key term. "
            "Answer should be 1-2 clear sentences."
        ),
        "intermediate": (
            "Create an INTERMEDIATE flashcard testing understanding. "
            "Question should ask how something works or compare two concepts. "
            "Answer should explain the mechanism clearly in 2-3 sentences."
        ),
        "advanced": (
            "Create an ADVANCED flashcard testing deep understanding. "
            "Question should involve a tricky scenario, edge case, or 'why' question. "
            "Answer should demonstrate expert-level insight in 2-4 sentences."
        )
    }

    instruction = difficulty_instructions.get(difficulty, difficulty_instructions["intermediate"])

    prompt = f"""You are a computer science tutor creating study flashcards.

CONTEXT (use ONLY this information):
{context}

TASK: Create ONE flashcard about {concept_name}.
{instruction}

Rules:
- Use ONLY information from the context above
- Do NOT add information not present in the context
- Keep the answer concise and accurate
- Format EXACTLY as shown below

FLASHCARD:
QUESTION: [your question here]
ANSWER: [your answer here]"""

    return prompt


def build_summary_prompt(concept_name: str, context: str,
                         difficulty: str) -> str:
    """
    Builds a prompt to generate a 3-5 bullet point summary.

    EXAMPLE OUTPUT:
      SUMMARY: Operating Systems — Key Points

      • Process scheduling algorithms (FCFS, SJF, Round Robin) determine CPU allocation
      • Deadlock requires all four Coffman conditions simultaneously
      • Virtual memory allows processes to use more RAM than physically available
      • Semaphores (binary and counting) solve synchronization problems
      • Paging eliminates external fragmentation at cost of internal fragmentation
    """
    depth_instructions = {
        "beginner":     "Cover the 3 most fundamental concepts only. Use simple language.",
        "intermediate": "Cover 4-5 key concepts. Include how they relate to each other.",
        "advanced":     "Cover 5 concepts including subtle points and common misconceptions."
    }

    depth = depth_instructions.get(difficulty, depth_instructions["intermediate"])

    prompt = f"""You are a computer science tutor creating study summaries.

CONTEXT (use ONLY this information):
{context}

TASK: Write a revision summary for {concept_name}.
{depth}

Rules:
- Use ONLY information from the context above
- Each bullet point must be ONE clear, exam-relevant sentence
- Start each bullet with a key term in bold (use **)
- Format EXACTLY as shown below

SUMMARY: {concept_name} — Key Points

• **[Key Term]**: [explanation]
• **[Key Term]**: [explanation]
• **[Key Term]**: [explanation]
• **[Key Term]**: [explanation]
• **[Key Term]**: [explanation]"""

    return prompt


def build_quiz_prompt(concept_name: str, context: str,
                      difficulty: str) -> str:
    """
    Builds a prompt to generate 3 multiple choice questions.

    EXAMPLE OUTPUT:
      QUIZ: Operating Systems

      Q1: Which scheduling algorithm can cause the "convoy effect"?
      A) Round Robin
      B) First Come First Serve
      C) Shortest Job First
      D) Priority Scheduling
      CORRECT: B
      EXPLANATION: FCFS causes convoy effect because long processes
                   block short ones waiting behind them.

      Q2: ...
    """
    difficulty_instructions = {
        "beginner":     (
            "Test basic recall: definitions, names, and simple facts. "
            "Distractors should be clearly wrong to a student who knows the material."
        ),
        "intermediate": (
            "Test understanding: ask about mechanisms and comparisons. "
            "Distractors should be plausible but subtly incorrect."
        ),
        "advanced":     (
            "Test deep knowledge: edge cases, exceptions, and subtle differences. "
            "Distractors should represent common misconceptions."
        )
    }

    instruction = difficulty_instructions.get(difficulty, difficulty_instructions["intermediate"])

    prompt = f"""You are a computer science professor writing exam questions.

CONTEXT (use ONLY this information):
{context}

TASK: Write exactly 3 multiple choice questions about {concept_name}.
{instruction}

Rules:
- Use ONLY information from the context above
- Each question must have exactly 4 options (A, B, C, D)
- Include a brief explanation for why the correct answer is right
- Format EXACTLY as shown below (3 questions, same format each)

QUIZ: {concept_name}

Q1: [question text]
A) [option]
B) [option]
C) [option]
D) [option]
CORRECT: [A/B/C/D]
EXPLANATION: [one sentence why this is correct]

Q2: [question text]
A) [option]
B) [option]
C) [option]
D) [option]
CORRECT: [A/B/C/D]
EXPLANATION: [one sentence why this is correct]

Q3: [question text]
A) [option]
B) [option]
C) [option]
D) [option]
CORRECT: [A/B/C/D]
EXPLANATION: [one sentence why this is correct]"""

    return prompt


def build_coding_task_prompt(concept_name: str, context: str,
                             difficulty: str) -> str:
    """
    Builds a prompt to generate a short Python coding problem.

    EXAMPLE OUTPUT:
      CODING TASK: Operating Systems — Process Scheduling

      PROBLEM:
      Implement a simple Round Robin scheduler in Python.
      Given a list of processes with burst times and a time quantum,
      simulate the Round Robin algorithm and return the average waiting time.

      INPUT:
      processes = [("P1", 10), ("P2", 5), ("P3", 8)]
      quantum = 4

      EXPECTED OUTPUT:
      Average waiting time: 10.67

      HINT:
      Use a queue to track processes. Each iteration, run the process
      for min(remaining_time, quantum) and requeue if not finished.
    """
    difficulty_instructions = {
        "beginner": (
            "Create a SIMPLE coding task. Student should implement a basic "
            "function demonstrating one concept from the notes. "
            "Problem should be solvable in 10-15 lines of Python."
        ),
        "intermediate": (
            "Create a MODERATE coding task. Student should implement a small "
            "simulation or algorithm from the notes. "
            "Problem should be solvable in 20-30 lines of Python."
        ),
        "advanced": (
            "Create a CHALLENGING coding task with a real-world twist. "
            "Student should implement something non-trivial requiring "
            "deep understanding of the concept. "
            "Problem should demonstrate practical application."
        )
    }

    instruction = difficulty_instructions.get(difficulty, difficulty_instructions["intermediate"])

    prompt = f"""You are a computer science coding instructor.

CONTEXT (use ONLY this information):
{context}

TASK: Create a Python coding exercise about {concept_name}.
{instruction}

Rules:
- The problem must be solvable using only Python standard library
- Provide clear INPUT and EXPECTED OUTPUT
- Include a HINT to guide the student without giving away the solution
- Keep the problem focused on ONE concept from the context
- Format EXACTLY as shown below

CODING TASK: {concept_name}

PROBLEM:
[problem description]

INPUT:
[example input]

EXPECTED OUTPUT:
[what the code should print/return]

HINT:
[one helpful hint without revealing the solution]"""

    return prompt


def get_prompt_for_type(content_type: str, concept_name: str,
                        context: str, difficulty: str) -> str:
    """
    Router: returns the right prompt builder for each content type.

    Args:
        content_type: one of FLASHCARD, SUMMARY, QUIZ, CODING_TASK
        concept_name: display name of the concept
        context:      retrieved context string
        difficulty:   "beginner", "intermediate", or "advanced"

    Returns:
        str: complete prompt ready to send to Mistral
    """
    builders = {
        FLASHCARD:   build_flashcard_prompt,
        SUMMARY:     build_summary_prompt,
        QUIZ:        build_quiz_prompt,
        CODING_TASK: build_coding_task_prompt
    }

    builder = builders.get(content_type)
    if builder is None:
        raise ValueError(f"Unknown content type: {content_type}. "
                         f"Must be one of: {VALID_CONTENT_TYPES}")

    return builder(concept_name, context, difficulty)


# =============================================================================
# STEP 4 — CALL OLLAMA API
# =============================================================================

def call_ollama(prompt: str, temperature: float = 0.7) -> str:
    """
    Calls the local Ollama API to generate text using Mistral 7B.

    HOW OLLAMA API WORKS:
      POST http://localhost:11434/api/generate
      Body: { "model": "mistral", "prompt": "...", "stream": false }
      Response: { "response": "generated text", "done": true, ... }

    PARAMETERS:
      temperature: controls randomness (0.0 = deterministic, 1.0 = creative)
        - 0.7 is a good balance for educational content:
          creative enough to vary, focused enough to be accurate
        - For quizzes, use 0.5 (more consistent)
        - For coding tasks, use 0.3 (more precise)

    stream=false:
      We want the complete response at once, not streamed token by token.
      streaming would require handling partial JSON which is more complex.

    TIMEOUT:
      120 seconds. Mistral 7B on a laptop CPU takes 30-90 seconds
      to generate a full quiz or coding task.

    ERROR HANDLING:
      - ConnectionError: Ollama server not running → tell user to run 'ollama serve'
      - Timeout: generation took too long → increase GENERATION_TIMEOUT
      - HTTPError: invalid request → check prompt format

    Args:
        prompt:      complete prompt string
        temperature: float 0.0-1.0

    Returns:
        str: generated text from Mistral
        Raises RuntimeError with user-friendly message on failure
    """
    payload = {
        "model":       OLLAMA_MODEL,
        "prompt":      prompt,
        "stream":      False,   # get complete response, not streamed
        "options": {
            "temperature":   temperature,
            "num_predict":   800,    # max tokens to generate
            "top_p":         0.9,    # nucleus sampling
            "repeat_penalty": 1.1,  # discourage repetition
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=GENERATION_TIMEOUT
        )
        response.raise_for_status()   # raises HTTPError for 4xx/5xx

        data = response.json()
        return data.get("response", "").strip()

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "❌ Cannot connect to Ollama.\n"
            "Make sure Ollama is running:\n"
            "  1. Install from https://ollama.com\n"
            "  2. Run: ollama serve\n"
            "  3. In another terminal: ollama pull mistral"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"❌ Ollama timed out after {GENERATION_TIMEOUT}s.\n"
            "Mistral is slow on CPU. Try:\n"
            "  - Close other applications to free RAM\n"
            "  - Increase GENERATION_TIMEOUT constant"
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"❌ Ollama API error: {e}")


def is_ollama_running() -> bool:
    """
    Checks if the Ollama server is running and responsive.

    Returns:
        bool: True if Ollama is running at localhost:11434
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


# =============================================================================
# STEP 5 — SAVE TO MONGODB
# =============================================================================

def save_generated_content(user_id: str, concept_id: str,
                            content_type: str, content: str,
                            difficulty_level: str) -> str:
    """
    Saves generated content to the generated_content MongoDB collection.

    COLLECTION SCHEMA:
      {
        "user_id":         "shivangi_01",
        "concept_id":      "os",
        "content_type":    "flashcard",
        "content":         "QUESTION: What is deadlock?\nANSWER: ...",
        "difficulty_level": "intermediate",
        "generated_at":    datetime(...),
        "rating":          0,         ← 1=good, -1=bad, 0=not rated yet
        "was_reviewed":    false,      ← did student actually study this?
        "model_used":      "mistral"
      }

    CACHING LOGIC:
      This function saves every generated piece.
      generate_or_retrieve() (below) checks if recent content exists
      before calling generation. If content was generated < 24 hours ago
      AND the student hasn't requested regeneration, we return the cached version.

    Args:
        user_id:         student's ID
        concept_id:      concept the content is about
        content_type:    one of: flashcard, summary, quiz, coding_task
        content:         the generated text
        difficulty_level: "beginner", "intermediate", "advanced"

    Returns:
        str: MongoDB document ID of the saved document
    """
    col = get_collection("generated_content")

    doc = {
        "user_id":          user_id,
        "concept_id":       concept_id,
        "content_type":     content_type,
        "content":          content,
        "difficulty_level": difficulty_level,
        "generated_at":     datetime.now(timezone.utc),
        "rating":           0,       # 0 = not rated, 1 = good, -1 = bad
        "was_reviewed":     False,
        "model_used":       OLLAMA_MODEL
    }

    result = col.insert_one(doc)
    return str(result.inserted_id)


def get_cached_content(user_id: str, concept_id: str,
                       content_type: str,
                       max_age_hours: int = 24) -> dict | None:
    """
    Retrieves recently generated content from MongoDB if it exists.

    CACHING STRATEGY:
      If the student already has content generated in the last 24 hours
      for this user + concept + content_type combination → return it.
      No need to call Ollama again (saves time + RAM).

    WHY 24 HOURS?
      If recall score hasn't dropped significantly in 24 hours,
      the same content is still relevant. After 24 hours, we regenerate
      because the student's situation (recall level, time gap) may have changed.

    STUDENT CAN FORCE REGENERATION:
      The review page has a "🔄 Regenerate" button that bypasses the cache.
      This is handled by passing force_regenerate=True to generate_content().

    Args:
        user_id:       student's ID
        concept_id:    which concept
        content_type:  flashcard/summary/quiz/coding_task
        max_age_hours: how old cached content can be (default 24h)

    Returns:
        dict: content document (without _id) if found and fresh enough
        None: if no recent content exists
    """
    from datetime import timedelta

    col = get_collection("generated_content")
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

    doc = col.find_one(
        {
            "user_id":      user_id,
            "concept_id":   concept_id,
            "content_type": content_type,
            "generated_at": {"$gte": cutoff}   # $gte = generated after cutoff
        },
        {"_id": 0},
        sort=[("generated_at", -1)]   # most recent first
    )

    return doc


def rate_content(user_id: str, concept_id: str,
                 content_type: str, rating: int) -> bool:
    """
    Saves student's thumbs up/down rating for a piece of content.

    RATING VALUES:
      1  = thumbs up (good content, helpful)
      -1 = thumbs down (bad content, unhelpful)
      0  = not rated

    HOW RATINGS ARE USED (future):
      If a content type consistently gets thumbs down for a concept,
      future prompts can be adjusted to avoid the same mistakes.
      For Phase 6, ratings are stored and displayed but not yet used
      to modify prompts.

    Args:
        user_id:      student's ID
        concept_id:   which concept
        content_type: which content type was rated
        rating:       1, -1, or 0

    Returns:
        bool: True if successful
    """
    if rating not in [-1, 0, 1]:
        return False

    try:
        col = get_collection("generated_content")
        col.update_one(
            {
                "user_id":      user_id,
                "concept_id":   concept_id,
                "content_type": content_type
            },
            {"$set": {"rating": rating}},
            sort=[("generated_at", -1)]   # update the most recent one
        )
        return True
    except Exception as e:
        print(f"⚠️  rate_content error: {e}")
        return False


def mark_content_as_reviewed(user_id: str, concept_id: str,
                              content_type: str) -> bool:
    """
    Marks a piece of generated content as actually studied by the student.

    CALLED BY: 3_review.py when student confirms they've finished studying.

    This differs from queue's "Mark as Reviewed" (which marks the CONCEPT):
    This marks the specific CONTENT PIECE (flashcard/quiz/etc.) as reviewed.

    Useful for:
      - Tracking which content types the student engages with most
      - Future: if student reviewed the flashcard but not the quiz,
        next time prioritize the quiz
    """
    try:
        col = get_collection("generated_content")
        col.update_one(
            {
                "user_id":      user_id,
                "concept_id":   concept_id,
                "content_type": content_type
            },
            {"$set": {"was_reviewed": True}},
            sort=[("generated_at", -1)]
        )
        return True
    except Exception as e:
        print(f"⚠️  mark_content_as_reviewed error: {e}")
        return False


# =============================================================================
# MASTER FUNCTION — generate_content()
# =============================================================================

def generate_content(
    user_id: str,
    concept_id: str,
    concept_name: str,
    content_type: str,
    recall_score: float,
    force_regenerate: bool = False
) -> dict:
    """
    Main function: generates (or retrieves cached) content for a concept.

    THIS IS THE ONLY FUNCTION 3_review.py CALLS FROM THIS FILE.
    All other functions above are helpers for this one.

    FULL PIPELINE:
      1. Check cache (unless force_regenerate)
      2. Determine difficulty from recall_score
      3. Retrieve relevant chunks from FAISS
      4. Build prompt for content_type + difficulty
      5. Call Ollama API
      6. Save to MongoDB
      7. Return content dict

    RETURN FORMAT:
      {
        "content":          "QUESTION: ...\nANSWER: ...",
        "content_type":     "flashcard",
        "difficulty_level": "intermediate",
        "concept_id":       "os",
        "concept_name":     "Operating Systems",
        "recall_score":     52.1,
        "from_cache":       False,
        "error":            None     ← or error message string if failed
      }

    ERROR HANDLING:
      If ANY step fails (FAISS not ready, Ollama not running, etc.),
      we return the dict with "error" field set to a user-friendly message.
      The review page displays this error gracefully without crashing.

    Args:
        user_id:          student's ID
        concept_id:       which concept (e.g. "os")
        concept_name:     display name (e.g. "Operating Systems")
        content_type:     one of VALID_CONTENT_TYPES
        recall_score:     student's current recall % (0–100)
        force_regenerate: if True, skip cache and generate fresh

    Returns:
        dict with keys: content, content_type, difficulty_level,
                        concept_id, concept_name, from_cache, error
    """
    result = {
        "content":          None,
        "content_type":     content_type,
        "difficulty_level": None,
        "concept_id":       concept_id,
        "concept_name":     concept_name,
        "recall_score":     recall_score,
        "from_cache":       False,
        "error":            None
    }

    # ── Validate content type ─────────────────────────────────────────────────
    if content_type not in VALID_CONTENT_TYPES:
        result["error"] = f"Invalid content type '{content_type}'"
        return result

    # ── Check cache first ─────────────────────────────────────────────────────
    if not force_regenerate:
        cached = get_cached_content(user_id, concept_id, content_type)
        if cached:
            result["content"]          = cached["content"]
            result["difficulty_level"] = cached["difficulty_level"]
            result["from_cache"]       = True
            return result

    # ── Check dependencies ────────────────────────────────────────────────────
    if not is_index_available():
        result["error"] = (
            "📚 Knowledge base not built yet.\n"
            "Run: python rag/ingest.py\n"
            "Then restart the app."
        )
        return result

    if not is_ollama_running():
        result["error"] = (
            "🤖 Ollama is not running.\n"
            "Start it with: ollama serve\n"
            "Make sure you've pulled the model: ollama pull mistral"
        )
        return result

    try:
        # ── Step 1: Determine difficulty ──────────────────────────────────────
        difficulty = get_difficulty_level(recall_score)
        result["difficulty_level"] = difficulty

        # ── Step 2: Retrieve chunks from FAISS ───────────────────────────────
        chunks = retrieve_chunks(concept_id, query_text=concept_name)

        if not chunks:
            # Fallback: search with just the concept_id
            chunks = retrieve_chunks(concept_id)

        if not chunks:
            result["error"] = (
                f"No relevant content found for '{concept_name}'.\n"
                f"Make sure data/notes/{concept_id}.txt exists and "
                f"python rag/ingest.py has been run."
            )
            return result

        # ── Step 3: Build context string ──────────────────────────────────────
        context = build_context_string(chunks)

        # ── Step 4: Build prompt ──────────────────────────────────────────────
        temperature = 0.7 if content_type != CODING_TASK else 0.3
        prompt = get_prompt_for_type(content_type, concept_name, context, difficulty)

        # ── Step 5: Call Ollama ───────────────────────────────────────────────
        generated_text = call_ollama(prompt, temperature=temperature)

        if not generated_text:
            result["error"] = "Ollama returned empty response. Try again."
            return result

        # ── Step 6: Save to MongoDB ───────────────────────────────────────────
        save_generated_content(
            user_id, concept_id, content_type,
            generated_text, difficulty
        )

        result["content"] = generated_text
        return result

    except RuntimeError as e:
        # Ollama-specific errors with user-friendly messages
        result["error"] = str(e)
        return result

    except Exception as e:
        result["error"] = f"Unexpected error: {e}"
        print(f"⚠️  generate_content error: {e}")
        return result