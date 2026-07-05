"""
backend/prompt_optimizer.py
============================
IMPROVEMENT 3 — Better LLM Prompting via Feedback Loop

PURPOSE
-------
When a student gives a thumbs-down to generated content, we learn from it.
This module:
  1. Reads thumbs-down ratings from MongoDB generated_content collection
  2. Analyses WHAT went wrong (too easy, too hard, off-topic, wrong format)
  3. Builds a "feedback context" string that gets prepended to the next prompt
  4. Tracks which prompt patterns work and which don't per concept

BEFORE (without feedback loop):
  generate flashcard about OS → same prompt every time → student rates 👎
  generate flashcard about OS again → exact same prompt → likely same bad result

AFTER (with feedback loop):
  generate flashcard about OS → student rates 👎 → reason: "too basic"
  generate flashcard about OS again → prompt now says:
    "IMPORTANT: Previous flashcards were rated too basic. Generate HARDER content."
  → better result

HOW RATINGS ARE USED:
  rating =  1 (👍): content was good → no changes needed
  rating = -1 (👎): content was bad  → add feedback hint to next prompt
  rating =  0 (⬜): not rated        → ignore

WHY NOT FINE-TUNING?
  Fine-tuning Mistral on our ratings would need GPU + many examples.
  Prompt injection is simpler and works surprisingly well.
  This is called "in-context learning" — a known effective technique.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone, timedelta
from backend.db import get_collection


# =============================================================================
# FEEDBACK REASON CATEGORIES
# =============================================================================

# Maps short reason codes to prompt hints
# The UI will let the student pick a reason when they give 👎
FEEDBACK_REASONS = {
    "too_easy":      "The previous content was too basic. Generate HARDER, more advanced content.",
    "too_hard":      "The previous content was too difficult. Generate SIMPLER, more beginner-friendly content.",
    "off_topic":     "The previous content drifted off-topic. Stay STRICTLY on the requested concept using only the provided context.",
    "wrong_format":  "The previous content had formatting issues. Follow the output format EXACTLY as specified.",
    "inaccurate":    "The previous content had factual errors. Be MORE careful and only use information explicitly in the context.",
    "too_long":      "The previous content was too long and verbose. Be MORE CONCISE.",
    "too_short":     "The previous content was too brief. Provide MORE DETAIL and explanation.",
    "generic":       "The previous content was too generic. Be MORE SPECIFIC to this concept.",
}


# =============================================================================
# SAVE FEEDBACK REASON
# =============================================================================

def save_feedback_with_reason(user_id: str, concept_id: str,
                               content_type: str,
                               rating: int,
                               reason_code: str = None) -> bool:
    """
    Saves a rating AND an optional reason code to the generated_content doc.

    Called when student clicks 👍 or 👎 and optionally selects a reason.

    REASON CODES (optional, from FEEDBACK_REASONS keys):
      "too_easy", "too_hard", "off_topic", "wrong_format",
      "inaccurate", "too_long", "too_short", "generic"

    Args:
        user_id:      student's ID
        concept_id:   which concept was rated
        content_type: flashcard / summary / quiz / coding_task
        rating:       1 = good, -1 = bad
        reason_code:  optional reason key from FEEDBACK_REASONS

    Returns:
        bool: True if saved successfully
    """
    try:
        col = get_collection("generated_content")

        update_fields = {
            "rating":           rating,
            "rating_timestamp": datetime.now(timezone.utc)
        }
        if reason_code and reason_code in FEEDBACK_REASONS:
            update_fields["feedback_reason"] = reason_code

        col.update_one(
            {
                "user_id":      user_id,
                "concept_id":   concept_id,
                "content_type": content_type
            },
            {"$set": update_fields},
            sort=[("generated_at", -1)]
        )
        return True

    except Exception as e:
        print(f"⚠️  save_feedback_with_reason error: {e}")
        return False


# =============================================================================
# BUILD FEEDBACK CONTEXT FOR NEXT PROMPT
# =============================================================================

def get_feedback_context(user_id: str, concept_id: str,
                          content_type: str,
                          lookback_days: int = 14) -> str:
    """
    Builds a feedback string to prepend to the next generation prompt.

    HOW IT WORKS:
      1. Fetch last N rated generations for this user + concept + content_type
      2. Separate into positive (👍) and negative (👎) examples
      3. If more thumbs-down than thumbs-up → add corrective hints
      4. If specific reason codes were given → add targeted instructions
      5. Return a formatted string ready to paste into the prompt

    EXAMPLES OF WHAT THIS RETURNS:

    Case 1 — No ratings yet:
      "" (empty string — no feedback context needed)

    Case 2 — Multiple thumbs down, reason "too_easy":
      '''
      FEEDBACK FROM PREVIOUS GENERATIONS:
      - Previous content was rated POOR. Reason: too basic.
        Instruction: The previous content was too basic. Generate HARDER content.
      - Apply this feedback carefully in the content you generate now.
      '''

    Case 3 — Mix of ratings:
      '''
      FEEDBACK FROM PREVIOUS GENERATIONS:
      - 2 previous generations were rated GOOD. Keep that quality level.
      - 1 previous generation was rated POOR. Reason: off_topic.
        Instruction: Stay STRICTLY on the requested concept.
      '''

    Args:
        user_id:      student's ID
        concept_id:   which concept
        content_type: flashcard / summary / quiz / coding_task
        lookback_days: how far back to look for ratings

    Returns:
        str: feedback context string (empty if no useful feedback)
    """
    col = get_collection("generated_content")
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    # Fetch recent rated generations (exclude unrated ones)
    rated_docs = list(col.find(
        {
            "user_id":      user_id,
            "concept_id":   concept_id,
            "content_type": content_type,
            "rating":       {"$ne": 0},     # $ne = not equal (exclude unrated)
            "generated_at": {"$gte": cutoff}
        },
        {"rating": 1, "feedback_reason": 1, "_id": 0},
        sort=[("generated_at", -1)],
        limit=5   # look at last 5 rated generations
    ))

    if not rated_docs:
        return ""   # no ratings yet → no feedback to give

    good_count = sum(1 for d in rated_docs if d.get("rating", 0) == 1)
    bad_count  = sum(1 for d in rated_docs if d.get("rating", 0) == -1)

    # Only build feedback if there are negative ratings
    if bad_count == 0:
        return ""   # all good → don't change anything

    feedback_lines = ["FEEDBACK FROM PREVIOUS GENERATIONS (apply this to improve):"]

    if good_count > 0:
        feedback_lines.append(
            f"- {good_count} previous generation(s) were rated GOOD. "
            "Maintain that quality."
        )

    # Add specific reason-based instructions
    bad_docs = [d for d in rated_docs if d.get("rating", 0) == -1]
    reasons  = [d.get("feedback_reason") for d in bad_docs if d.get("feedback_reason")]
    reason_counts = {}
    for r in reasons:
        reason_counts[r] = reason_counts.get(r, 0) + 1

    if reason_counts:
        for reason_code, count in reason_counts.items():
            hint = FEEDBACK_REASONS.get(reason_code, "")
            if hint:
                feedback_lines.append(
                    f"- {count} generation(s) were rated POOR. Reason: {reason_code}.\n"
                    f"  Instruction: {hint}"
                )
    else:
        # No specific reason given — add a generic hint
        feedback_lines.append(
            f"- {bad_count} generation(s) were rated POOR. "
            "Try a different approach, angle, or level of detail."
        )

    feedback_lines.append(
        "Apply ALL of the above feedback when generating the content below."
    )

    return "\n".join(feedback_lines)


# =============================================================================
# INJECT FEEDBACK INTO PROMPT
# =============================================================================

def inject_feedback_into_prompt(base_prompt: str, feedback_context: str) -> str:
    """
    Inserts feedback context into an existing prompt.

    PLACEMENT STRATEGY:
      We insert feedback right after the CONTEXT section
      and before the TASK section. This ensures the model
      reads the feedback before it starts generating.

    Example prompt structure:
      You are a tutor...
      CONTEXT: [retrieved chunks]
      [FEEDBACK INSERTED HERE]
      TASK: Create a flashcard...
      Rules: ...

    If no TASK: marker found, feedback is prepended to the entire prompt.

    Args:
        base_prompt:      the original prompt from generate.py
        feedback_context: the string from get_feedback_context()

    Returns:
        str: modified prompt with feedback injected
    """
    if not feedback_context:
        return base_prompt   # nothing to inject

    # Find the TASK: line and insert before it
    if "TASK:" in base_prompt:
        parts = base_prompt.split("TASK:", 1)
        return parts[0] + f"\n{feedback_context}\n\nTASK:" + parts[1]

    # Fallback: prepend to entire prompt
    return f"{feedback_context}\n\n{base_prompt}"


# =============================================================================
# GET FEEDBACK STATS (for settings/about page display)
# =============================================================================

def get_feedback_stats(user_id: str) -> dict:
    """
    Returns a summary of how ratings have evolved over time.

    USED BY: settings page to show "Your feedback has improved generation quality"

    Returns:
        dict: {
          "total_rated":   15,
          "thumbs_up":      9,
          "thumbs_down":    6,
          "satisfaction":  60.0,   ← thumbs_up / total_rated * 100
          "most_common_issue": "too_easy",
          "concepts_rated":  ["os", "dbms"]
        }
    """
    col = get_collection("generated_content")
    docs = list(col.find(
        {"user_id": user_id, "rating": {"$ne": 0}},
        {"rating": 1, "feedback_reason": 1, "concept_id": 1, "_id": 0}
    ))

    if not docs:
        return {
            "total_rated": 0, "thumbs_up": 0,
            "thumbs_down": 0, "satisfaction": 0.0,
            "most_common_issue": None, "concepts_rated": []
        }

    good    = sum(1 for d in docs if d.get("rating") ==  1)
    bad     = sum(1 for d in docs if d.get("rating") == -1)
    total   = len(docs)
    satisfy = round(good / total * 100, 1) if total else 0

    reasons = [d["feedback_reason"] for d in docs if d.get("feedback_reason")]
    most_common = max(set(reasons), key=reasons.count) if reasons else None

    concepts = list(set(d.get("concept_id", "") for d in docs))

    return {
        "total_rated":        total,
        "thumbs_up":          good,
        "thumbs_down":        bad,
        "satisfaction":       satisfy,
        "most_common_issue":  most_common,
        "concepts_rated":     concepts
    }