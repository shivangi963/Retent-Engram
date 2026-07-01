"""
backend/scheduler.py
======================
PHASE 5 — Review Scheduler

PURPOSE
-------
This is the BRAIN of Phase 5. It takes recall scores (from Phase 4)
and answers the question every student has every day:
  "What should I study today, and in what order?"

It does this by computing an URGENCY SCORE for every concept,
then filtering + sorting to produce the daily review queue.

SIX FEATURES THIS FILE HANDLES:
  1. Urgency Score Calculator
  2. Daily Review Queue builder
  3. Review Completion Tracking (Mark as Reviewed)
  4. Snooze Feature (hide concept for 1 or 3 days)
  5. Daily Goal Management (set + track)
  6. Study Streak Tracker (consecutive days)

URGENCY FORMULA EXPLAINED
--------------------------
urgency = (
    0.5 × (1 - recall/100)          ← recall component   (50% weight)
  + 0.3 × min(hours_elapsed/168, 1) ← time component     (30% weight)
  + 0.2 × (difficulty/5)            ← difficulty factor  (20% weight)
) × 100

This gives a number from 0 to 100:
  100 = MOST URGENT (completely forgotten + week-long gap + hardest concept)
  0   = NOT URGENT  (perfectly remembered + just reviewed + easiest concept)

WHY THESE WEIGHTS?
  - Recall (50%) dominates because it's the most direct signal of need
  - Time (30%) matters because forgetting is time-dependent (Ebbinghaus)
  - Difficulty (20%) acts as a tiebreaker — hard concepts deserve more attention

WHY cap time at 168 hours (7 days)?
  After a week without review, the urgency from time is "maxed out".
  We don't want a concept unseen for 6 months to score 10x higher than
  one unseen for 1 week — both are equally "urgent from time perspective".

HOW DATA FLOWS IN PHASE 5:
  MongoDB (events + recall_scores)
      ↓
  compute_urgency_for_user()    ← THIS FILE
      ↓
  Urgency scores saved to recall_scores collection
      ↓
  build_review_queue()          ← THIS FILE
      ↓
  4_queue.py displays the filtered, sorted queue
      ↓
  Student clicks "Mark as Reviewed" or "Snooze"
      ↓
  mark_as_reviewed() / snooze_concept()  ← THIS FILE
      ↓
  MongoDB updated, queue refreshes
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone, timedelta
from backend.db import get_collection


# =============================================================================
# CONSTANTS
# =============================================================================

# Urgency formula weights (must sum to 1.0)
WEIGHT_RECALL     = 0.50   # recall component
WEIGHT_TIME       = 0.30   # time-since-last-review component
WEIGHT_DIFFICULTY = 0.20   # concept difficulty component

# Time cap for urgency calculation: 7 days = 168 hours
# Beyond this, the time component is "maxed out" at 1.0
MAX_HOURS_FOR_TIME_FACTOR = 168.0

# Only show concepts below this recall threshold in the queue
REVIEW_THRESHOLD = 65.0

# Suggested content type based on recall level
# Used in the queue to guide the student on HOW to review
CONTENT_SUGGESTIONS = {
    "High":   "📝 Flashcard + Quiz",   # recall < 40: need active recall
    "Medium": "📖 Summary + Quiz",     # recall 40–65: reinforce understanding
    "Low":    "⚡ Quick Flashcard",     # recall ≥ 65: light touch (shouldn't be in queue)
}


# =============================================================================
# HELPER — make datetime timezone-aware
# =============================================================================

def _make_aware(dt: datetime) -> datetime:
    """Converts naive datetime to UTC-aware datetime."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# =============================================================================
# FEATURE 1 — URGENCY SCORE CALCULATOR
# =============================================================================

def compute_urgency_score(
    recall_score: float,
    hours_since_last: float,
    difficulty: int
) -> float:
    """
    Computes a single urgency number for one concept.

    FORMULA:
      urgency = (
          W_recall     × (1 - recall/100)
        + W_time       × min(hours/MAX_HOURS, 1.0)
        + W_difficulty × (difficulty/5)
      ) × 100

    COMPONENT BREAKDOWN:

    Recall component: (1 - recall/100)
      recall=0   → component=1.0 (completely forgotten = maximum urgency)
      recall=50  → component=0.5 (half forgotten = medium urgency)
      recall=100 → component=0.0 (perfectly remembered = no urgency)

    Time component: min(hours/168, 1.0)
      hours=0   → component=0.0 (just reviewed = no time urgency)
      hours=84  → component=0.5 (3.5 days ago = medium time urgency)
      hours=168 → component=1.0 (7+ days ago = maximum time urgency)
      hours=336 → component=1.0 (still capped at 1.0)

    Difficulty component: difficulty/5
      difficulty=1 → component=0.2 (easiest)
      difficulty=3 → component=0.6 (medium)
      difficulty=5 → component=1.0 (hardest)

    EXAMPLE:
      recall=30%, hours=48, difficulty=4
      = 0.5×(1-0.30) + 0.3×(48/168) + 0.2×(4/5) × 100
      = 0.5×0.70 + 0.3×0.286 + 0.2×0.80 × 100
      = 0.350 + 0.086 + 0.160 × 100
      = 59.6  → High urgency

    Args:
        recall_score:    float 0–100 from Phase 4
        hours_since_last: float hours since last review
        difficulty:      int 1–5 from concepts.json

    Returns:
        float: urgency score 0–100 (higher = review sooner)
    """
    # Recall component: lower recall = higher urgency
    recall_component = 1.0 - (recall_score / 100.0)

    # Time component: longer gap = higher urgency, capped at 1 week
    time_component = min(hours_since_last / MAX_HOURS_FOR_TIME_FACTOR, 1.0)

    # Difficulty component: harder concepts = more urgent when forgotten
    difficulty_component = difficulty / 5.0

    # Weighted sum scaled to 0–100
    urgency = (
        WEIGHT_RECALL     * recall_component +
        WEIGHT_TIME       * time_component +
        WEIGHT_DIFFICULTY * difficulty_component
    ) * 100.0

    return round(urgency, 2)


def get_urgency_level(urgency_score: float) -> str:
    """
    Converts a numeric urgency score into a human-readable level.

    LEVELS:
      ≥ 70 → Critical  (review immediately)
      ≥ 50 → High      (review today)
      ≥ 30 → Medium    (review soon)
      < 30 → Low       (can wait)

    Returns:
        str: "Critical", "High", "Medium", or "Low"
    """
    if urgency_score >= 70:
        return "Critical"
    elif urgency_score >= 50:
        return "High"
    elif urgency_score >= 30:
        return "Medium"
    else:
        return "Low"


def get_urgency_emoji(level: str) -> str:
    """Returns an emoji for each urgency level for display in the queue."""
    return {
        "Critical": "🔴",
        "High":     "🟠",
        "Medium":   "🟡",
        "Low":      "🟢"
    }.get(level, "⚪")


# =============================================================================
# FEATURE 1b — COMPUTE URGENCY FOR ALL CONCEPTS OF ONE USER
# =============================================================================

def compute_urgency_for_user(user_id: str, concepts_lookup: dict) -> list:
    """
    Computes urgency scores for ALL concepts of one user and saves them
    to the recall_scores collection in MongoDB.

    HOW IT WORKS:
      1. Fetch all recall_score documents for this user
         (these were saved by pipeline.py in Phase 4)
      2. For each recall_score document:
         a. Get recall_score and features.hours_since_last
         b. Look up difficulty from concepts_lookup dict
         c. Call compute_urgency_score()
         d. Save urgency_score back to the same MongoDB document
      3. Return list of enriched results for the queue to display

    CONCEPTS_LOOKUP:
      A dict from concepts.json: { "os": {"difficulty": 4, ...}, ... }
      Passed in from the caller (4_queue.py) which already loaded concepts.json

    WHAT GETS ADDED TO MongoDB recall_scores documents:
      {
        ...existing fields...,
        "urgency_score":       59.6,         ← new
        "urgency_level":       "High",        ← new
      }

    Args:
        user_id:         student's ID string
        concepts_lookup: dict { concept_id: concept_dict_from_json }

    Returns:
        list of enriched result dicts sorted by urgency descending
    """
    col = get_collection("recall_scores")

    # Fetch all recall score docs for this user
    all_scores = list(col.find({"user_id": user_id}, {"_id": 0}))

    if not all_scores:
        return []

    results = []

    for score_doc in all_scores:
        concept_id   = score_doc.get("concept_id", "")
        recall_score = score_doc.get("recall_score", 50.0)
        features     = score_doc.get("features", {})

        # Get hours since last review from stored features
        hours_since_last = features.get("hours_since_last", 24.0)

        # Get difficulty from concepts.json lookup (default 3 if not found)
        concept_info = concepts_lookup.get(concept_id, {})
        difficulty   = concept_info.get("difficulty", 3)

        # Compute urgency
        urgency_score = compute_urgency_score(recall_score, hours_since_last, difficulty)
        urgency_level = get_urgency_level(urgency_score)

        # Save urgency back to MongoDB (update the existing recall_score document)
        col.update_one(
            {"user_id": user_id, "concept_id": concept_id},
            {"$set": {
                "urgency_score": urgency_score,
                "urgency_level": urgency_level
            }}
        )

        # Add to results list for display
        results.append({
            "concept_id":     concept_id,
            "concept_name":   concept_info.get("name", concept_id),
            "subject":        concept_info.get("subject", ""),
            "difficulty":     difficulty,
            "recall_score":   recall_score,
            "priority":       score_doc.get("priority", "Medium"),
            "urgency_score":  urgency_score,
            "urgency_level":  urgency_level,
            "hours_since":    hours_since_last,
            "snoozed_until":  score_doc.get("snoozed_until", None),
            "last_reviewed_today": score_doc.get("last_reviewed_today", False),
            "features":       features
        })

    # Sort by urgency descending: most urgent first
    results.sort(key=lambda x: x["urgency_score"], reverse=True)
    return results


# =============================================================================
# FEATURE 2 — BUILD DAILY REVIEW QUEUE
# =============================================================================

def build_review_queue(user_id: str, enriched_results: list) -> list:
    """
    Filters the urgency-sorted results to produce TODAY'S review queue.

    FILTERING RULES (applied in order):
      1. Remove concepts with recall >= REVIEW_THRESHOLD (65%)
         → Student remembers them well enough, no need to review
      2. Remove concepts that are currently SNOOZED
         → Student explicitly asked to hide them for now
      3. Remove concepts already marked as reviewed TODAY
         → Once reviewed today, concept exits the queue
         → It re-enters tomorrow if recall drops again

    WHAT REMAINS = the queue: concepts needing review right now.

    ENRICHED WITH:
      - suggested_content: what study method to use (flashcard, quiz, etc.)
      - estimated_time_min: how long this review might take

    Args:
        user_id:          student's ID
        enriched_results: list from compute_urgency_for_user()

    Returns:
        list of queue items, sorted by urgency descending
    """
    now = datetime.now(timezone.utc)
    queue = []

    for item in enriched_results:
        recall        = item["recall_score"]
        snoozed_until = item.get("snoozed_until")
        reviewed_today = item.get("last_reviewed_today", False)

        # Rule 1: Skip if recall is above threshold
        if recall >= REVIEW_THRESHOLD:
            continue

        # Rule 2: Skip if currently snoozed
        if snoozed_until is not None:
            snoozed_dt = _make_aware(snoozed_until)
            if now < snoozed_dt:
                continue   # still within snooze window

        # Rule 3: Skip if already reviewed today
        if reviewed_today:
            continue

        # All filters passed → add to queue with extra display info
        priority = item.get("priority", "Medium")
        features = item.get("features", {})

        queue_item = {
            **item,   # spread all existing fields

            # Suggested content type based on priority
            "suggested_content": CONTENT_SUGGESTIONS.get(priority, "📖 Review"),

            # Estimated time: use avg_response_time from Phase 2 features
            # If no history, default to 20 minutes
            "estimated_time_min": round(features.get("avg_response_time", 20.0))
        }
        queue.append(queue_item)

    return queue


# =============================================================================
# FEATURE 3 — MARK AS REVIEWED
# =============================================================================

def mark_as_reviewed(user_id: str, concept_id: str, score: float = 0.75) -> bool:
    """
    Called when student clicks "Mark as Reviewed" on a queue item.

    WHAT HAPPENS:
      1. Logs a new "review" event in the events collection
         (this is real event data that improves the ML model in Phase 4)
      2. Sets last_reviewed_today = True on the recall_score document
         (concept disappears from today's queue)
      3. Updates the user's streak and last_active_date

    WHY LOG A NEW EVENT?
      Every "Mark as Reviewed" action is genuine study evidence.
      Phase 4's XGBoost model was trained on events data.
      Adding this event means the model gets better data next training run.
      The score defaults to 0.75 (good) — student can override it.

    WHY last_reviewed_today?
      When pipeline.py recomputes recall scores (on dashboard load),
      it recalculates hours_since_last based on the new event.
      The concept's urgency drops immediately.
      But we ALSO set last_reviewed_today = True so it doesn't
      reappear in today's queue even if urgency is still high.
      (It resets to False at midnight via reset_daily_flags below.)

    Args:
        user_id:    student's ID
        concept_id: which concept was reviewed
        score:      self-reported performance score 0.0–1.0 (default 0.75)

    Returns:
        bool: True if successful, False if error occurred
    """
    try:
        now = datetime.now(timezone.utc)

        # ── Log new event in events collection ───────────────────────────────
        events_col = get_collection("events")
        new_event = {
            "user_id":            user_id,
            "concept_id":         concept_id,
            "event_type":         "review",          # special event type for Phase 5
            "score":              score,
            "response_time_min":  20,                # default, can be timed in future
            "hints_used":         0,
            "timestamp":          now,
            "source":             "queue_review"     # marks this as from Phase 5 queue
        }
        events_col.insert_one(new_event)

        # ── Update recall_scores: mark reviewed today ─────────────────────────
        scores_col = get_collection("recall_scores")
        scores_col.update_one(
            {"user_id": user_id, "concept_id": concept_id},
            {"$set": {
                "last_reviewed_today": True,
                "last_queue_review":   now
            }}
        )

        # ── Update user streak ────────────────────────────────────────────────
        update_streak(user_id)

        return True

    except Exception as e:
        print(f"⚠️  mark_as_reviewed error: {e}")
        return False


# =============================================================================
# FEATURE 4 — SNOOZE FEATURE
# =============================================================================

def snooze_concept(user_id: str, concept_id: str, days: int = 1) -> bool:
    """
    Hides a concept from the review queue for a specified number of days.

    WHAT SNOOZE MEANS:
      The concept is NOT deleted. It still has a recall score.
      It's just hidden from today's queue until snoozed_until datetime passes.
      When the snooze expires, the concept reappears in the queue normally.

    USE CASE:
      Student knows they'll review "OS" in class tomorrow.
      They snooze it for 1 day. The queue shows other concepts instead.
      Next day, OS is back in the queue.

    HOW IT WORKS:
      Sets snoozed_until = now + days in recall_scores MongoDB document.
      build_review_queue() checks this field and skips snoozed concepts.

    EXAMPLE:
      snooze_concept("shivangi_01", "os", days=3)
      → recall_scores.snoozed_until = 2026-06-13T10:30:00Z
      → "OS" hidden from queue until June 13

    Args:
        user_id:    student's ID
        concept_id: which concept to snooze
        days:       how many days to snooze (1 or 3)

    Returns:
        bool: True if successful
    """
    try:
        now = datetime.now(timezone.utc)
        snooze_until = now + timedelta(days=days)

        col = get_collection("recall_scores")
        col.update_one(
            {"user_id": user_id, "concept_id": concept_id},
            {"$set": {"snoozed_until": snooze_until}},
            upsert=True
        )

        print(f"😴  Snoozed '{concept_id}' for {days} day(s) until {snooze_until.date()}")
        return True

    except Exception as e:
        print(f"⚠️  snooze_concept error: {e}")
        return False


def unsnooze_concept(user_id: str, concept_id: str) -> bool:
    """
    Removes a snooze, making the concept immediately visible in the queue.

    USE CASE:
      Student accidentally snoozed a concept and wants it back.

    Args:
        user_id:    student's ID
        concept_id: which concept to unsnooze

    Returns:
        bool: True if successful
    """
    try:
        col = get_collection("recall_scores")
        col.update_one(
            {"user_id": user_id, "concept_id": concept_id},
            {"$set": {"snoozed_until": None}}
        )
        return True
    except Exception as e:
        print(f"⚠️  unsnooze error: {e}")
        return False


def get_snoozed_concepts(user_id: str) -> list:
    """
    Returns all currently snoozed concepts for a user.

    USED BY: 4_queue.py to show a "Snoozed" section at the bottom
    so the student knows what they've hidden.

    Returns:
        list of (concept_id, snoozed_until) tuples currently active
    """
    now = datetime.now(timezone.utc)
    col = get_collection("recall_scores")

    snoozed = list(col.find(
        {
            "user_id":      user_id,
            "snoozed_until": {"$gt": now}   # $gt = greater than now = still snoozed
        },
        {"concept_id": 1, "snoozed_until": 1, "_id": 0}
    ))

    return snoozed


# =============================================================================
# FEATURE 5 — DAILY GOAL MANAGEMENT
# =============================================================================

def get_or_create_daily_goal(user_id: str) -> int:
    """
    Returns the student's daily review goal.
    Creates a default of 3 if not set yet.

    STORED IN: users collection as daily_goal field.

    Default of 3 comes from the project plan:
    "Student sets a daily review goal (e.g. review 3 concepts per day)"

    Args:
        user_id: student's ID

    Returns:
        int: daily goal (default 3)
    """
    col = get_collection("users")
    user = col.find_one({"user_id": user_id}, {"daily_goal": 1, "_id": 0})

    if user and "daily_goal" in user:
        return user["daily_goal"]

    # First time: set default goal
    col.update_one(
        {"user_id": user_id},
        {"$set": {"daily_goal": 3}},
        upsert=True
    )
    return 3


def set_daily_goal(user_id: str, goal: int) -> bool:
    """
    Updates the student's daily review goal.

    Args:
        user_id: student's ID
        goal:    new goal (1–20 concepts per day)

    Returns:
        bool: True if successful
    """
    if goal < 1 or goal > 20:
        print("⚠️  Daily goal must be between 1 and 20")
        return False

    try:
        col = get_collection("users")
        col.update_one(
            {"user_id": user_id},
            {"$set": {"daily_goal": goal}},
            upsert=True
        )
        return True
    except Exception as e:
        print(f"⚠️  set_daily_goal error: {e}")
        return False


def count_reviewed_today(user_id: str) -> int:
    """
    Counts how many concepts the student has reviewed TODAY.

    HOW IT WORKS:
      Counts recall_score documents where last_reviewed_today = True.
      This flag is set by mark_as_reviewed() and reset by reset_daily_flags().

    USED BY:
      4_queue.py progress bar: "2 of 3 concepts reviewed today"

    Args:
        user_id: student's ID

    Returns:
        int: count of concepts reviewed today
    """
    col = get_collection("recall_scores")
    return col.count_documents({
        "user_id":           user_id,
        "last_reviewed_today": True
    })


def reset_daily_flags(user_id: str):
    """
    Resets last_reviewed_today = False for all concepts.

    WHEN TO CALL:
      Called automatically when the queue page loads if it's a new day.
      We detect a "new day" by comparing the user's last_active_date
      with today's date. If different → new day → reset flags.

    WHY NEEDED:
      Without this, concepts marked as reviewed yesterday would stay
      hidden from today's queue.

    AUTOMATIC LOGIC (handled in build_and_get_queue below):
      1. On queue page load, check user's last_active_date
      2. If last_active_date is before today → call this function
      3. Update last_active_date to today

    Args:
        user_id: student's ID
    """
    col = get_collection("recall_scores")
    col.update_many(
        {"user_id": user_id},
        {"$set": {"last_reviewed_today": False}}
    )
    print(f"🔄  Daily flags reset for user '{user_id}'")


# =============================================================================
# FEATURE 6 — STUDY STREAK TRACKER
# =============================================================================

def update_streak(user_id: str) -> dict:
    """
    Updates the student's study streak after they mark a concept as reviewed.

    STREAK RULES:
      - Streak increases by 1 each day the student reviews AT LEAST ONE concept
      - If student misses a day, streak resets to 1 (today counts as day 1)
      - Streak is stored in users.streak_days

    ALGORITHM:
      1. Get user's last_active_date from MongoDB
      2. Compare with TODAY
      3. If last_active_date == yesterday → streak continues (streak_days + 1)
      4. If last_active_date == today → already updated today, no change
      5. If last_active_date is older → missed a day, streak resets to 1
      6. Save new streak_days and last_active_date

    EXAMPLE:
      Day 1: review → streak=1, last_active=Day1
      Day 2: review → streak=2, last_active=Day2
      Day 3: no review
      Day 4: review → streak=1, last_active=Day4 (reset because missed Day 3)

    EDGE CASE — same day:
      Student reviews 3 concepts on same day.
      Only the FIRST review updates the streak.
      Subsequent reviews on same day are ignored (no double-counting).

    Args:
        user_id: student's ID

    Returns:
        dict: {"streak_days": int, "last_active_date": datetime}
    """
    users_col = get_collection("users")
    now = datetime.now(timezone.utc)
    today = now.date()

    # Fetch current streak data
    user = users_col.find_one(
        {"user_id": user_id},
        {"streak_days": 1, "last_active_date": 1, "_id": 0}
    )

    current_streak = user.get("streak_days", 0) if user else 0
    last_active    = user.get("last_active_date") if user else None

    if last_active is not None:
        last_active = _make_aware(last_active)
        last_active_date = last_active.date()
    else:
        last_active_date = None

    # Determine new streak value
    if last_active_date == today:
        # Already reviewed today — no change
        new_streak = current_streak
    elif last_active_date == today - timedelta(days=1):
        # Reviewed yesterday — streak continues!
        new_streak = current_streak + 1
    else:
        # Missed one or more days — streak resets
        new_streak = 1

    # Save to MongoDB
    users_col.update_one(
        {"user_id": user_id},
        {"$set": {
            "streak_days":      new_streak,
            "last_active_date": now
        }},
        upsert=True
    )

    return {
        "streak_days":      new_streak,
        "last_active_date": now
    }


def get_streak(user_id: str) -> int:
    """
    Returns the student's current streak in days.

    ALSO CHECKS if the streak has been broken:
      If last_active_date was before yesterday, the streak is 0.
      We don't reset it in MongoDB here (update_streak handles that),
      but we return 0 so the display shows the correct value.

    Args:
        user_id: student's ID

    Returns:
        int: current streak (0 if broken)
    """
    users_col = get_collection("users")
    user = users_col.find_one(
        {"user_id": user_id},
        {"streak_days": 1, "last_active_date": 1, "_id": 0}
    )

    if not user:
        return 0

    streak       = user.get("streak_days", 0)
    last_active  = user.get("last_active_date")

    if last_active is None:
        return 0

    last_active = _make_aware(last_active)
    yesterday   = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    today       = datetime.now(timezone.utc).date()

    # If last active was before yesterday → streak is broken
    if last_active.date() < yesterday:
        return 0

    return streak


# =============================================================================
# ESTIMATED REVIEW TIME
# =============================================================================

def compute_total_estimated_time(queue: list) -> int:
    """
    Sums up estimated review times for all items in the queue.

    DISPLAYED AS:
      "Today's review will take ~25 minutes"
      at the top of the queue page.

    Args:
        queue: list of queue items from build_review_queue()

    Returns:
        int: total estimated minutes
    """
    total = sum(item.get("estimated_time_min", 20) for item in queue)
    return total


# =============================================================================
# DAY BOUNDARY CHECK
# =============================================================================

def check_and_reset_new_day(user_id: str) -> bool:
    """
    Checks if it's a new day since the user last used the app.
    If yes, resets daily flags and returns True.

    CALLED BY: 4_queue.py on every page load (before building the queue).

    HOW IT WORKS:
      1. Read user's last_active_date from MongoDB
      2. Compare .date() (just the calendar date, not time) with today
      3. If different → new day → call reset_daily_flags()

    NOTE: We do NOT update last_active_date here.
      last_active_date is updated by update_streak() when the student
      actually reviews something. If we updated it here, a student
      who just opened the queue without reviewing would count as "active".

    Args:
        user_id: student's ID

    Returns:
        bool: True if it was a new day (flags were reset), False otherwise
    """
    users_col = get_collection("users")
    user = users_col.find_one(
        {"user_id": user_id},
        {"last_active_date": 1, "_id": 0}
    )

    today = datetime.now(timezone.utc).date()

    if not user or not user.get("last_active_date"):
        # First time using the app — reset to be safe
        reset_daily_flags(user_id)
        return True

    last_active = _make_aware(user["last_active_date"])
    if last_active.date() < today:
        reset_daily_flags(user_id)
        return True

    return False


# =============================================================================
# MASTER FUNCTION — called by 4_queue.py
# =============================================================================

def build_and_get_queue(user_id: str, concepts_lookup: dict) -> dict:
    """
    One-stop function that does everything needed to display the queue page.

    CALLED BY: frontend/pages/4_queue.py on every page load.

    STEPS:
      1. Check if it's a new day → reset daily flags if needed
      2. Compute urgency scores for all concepts
      3. Build the filtered review queue
      4. Get daily goal + progress
      5. Get streak
      6. Compute total estimated time

    RETURN VALUE — everything the queue page needs:
      {
        "queue":            [list of queue items sorted by urgency],
        "all_concepts":     [list of ALL concepts with urgency, for reference],
        "daily_goal":       3,
        "reviewed_today":   1,
        "streak_days":      5,
        "is_new_day":       False,
        "total_time_min":   45,
        "snoozed_concepts": [list of currently snoozed items]
      }

    Args:
        user_id:         student's ID
        concepts_lookup: dict { concept_id: concept_dict } from concepts.json

    Returns:
        dict with all queue page data
    """
    # Step 1: Check day boundary
    is_new_day = check_and_reset_new_day(user_id)

    # Step 2: Compute urgency for all concepts
    all_concepts = compute_urgency_for_user(user_id, concepts_lookup)

    # Step 3: Build filtered queue
    queue = build_review_queue(user_id, all_concepts)

    # Step 4: Daily goal progress
    daily_goal      = get_or_create_daily_goal(user_id)
    reviewed_today  = count_reviewed_today(user_id)

    # Step 5: Streak
    streak_days = get_streak(user_id)

    # Step 6: Total estimated time
    total_time = compute_total_estimated_time(queue)

    # Step 7: Snoozed concepts
    snoozed = get_snoozed_concepts(user_id)

    return {
        "queue":            queue,
        "all_concepts":     all_concepts,
        "daily_goal":       daily_goal,
        "reviewed_today":   reviewed_today,
        "streak_days":      streak_days,
        "is_new_day":       is_new_day,
        "total_time_min":   total_time,
        "snoozed_concepts": snoozed
    }

def get_dependent_concepts(concept_id: str, concepts_lookup: dict) -> list:
    """
    Returns concepts that depend on this one (have it as a prerequisite).
    If OS is urgent, its dependents should also be flagged.
    """
    dependents = []
    for cid, info in concepts_lookup.items():
        if concept_id in info.get("prerequisites", []):
            dependents.append(cid)
    return dependents



def predict_forgetting_date(recall_score: float, stability_hours: float) -> str:
    """
    Predicts when recall will drop below 40% (High priority threshold).
    Returns a human-readable string like "Tomorrow" or "In 3 days".
    """
    import math
    if stability_hours <= 0 or recall_score <= 0:
        return "Already forgotten"

    # Solve for t: 40 = recall × e^(-t/stability)
    # → t = -stability × ln(40/recall)
    if recall_score <= 40:
        return "Already at High priority"

    t_hours = -stability_hours * math.log(40.0 / recall_score)
    if t_hours < 0:
        return "Already forgotten"
    elif t_hours < 24:
        return f"In ~{int(t_hours)}h"
    elif t_hours < 48:
        return "Tomorrow"
    else:
        return f"In ~{int(t_hours // 24)} days"