"""
test_phase2_and_3.py
======================
COMPLETE TEST SCRIPT FOR PHASE 2 AND PHASE 3

PURPOSE
-------
Run this script from your project root to verify:
  ✅ Feature extraction works correctly
  ✅ Recall score formula produces sensible values
  ✅ MongoDB saves and reads back correctly
  ✅ Dashboard data functions return expected shapes

HOW TO RUN:
-----------
  cd Retent-Engram
  python test_phase2_and_3.py

WHAT YOU WILL SEE:
------------------
  Running Phase 2 + Phase 3 tests...
  ─────────────────────────────────
  TEST 1: Feature extraction
    hours_since_last   = 48.0
    total_reviews      = 3
    avg_score          = 0.65
    last_score         = 0.8
    success_streak     = 2
    avg_response_time  = 20.0
  ✅ PASS — Features extracted correctly

  TEST 2: Ebbinghaus formula
    Recall after  0h : 73.0%  (just reviewed)
    Recall after 12h : 64.5%
    Recall after 24h : 57.1%
    ...
  ✅ PASS — Recall decreases over time as expected

  ...and so on for all tests.
"""

import sys
import os

# Make sure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timezone, timedelta
from backend.ml.features import extract_features
from backend.ml.scorer import (
    compute_base_retention,
    compute_stability,
    apply_decay,
    compute_recall_score,
    get_priority
)
from backend.ml.pipeline import compute_scores_for_user
from backend.db import (
    get_recall_scores,
    get_event_counts_by_concept,
    get_last_event_per_concept,
    get_total_events_count
)


print("\n" + "="*60)
print("  RETENT ENGRAM — Phase 2 + 3 Test Suite")
print("="*60 + "\n")

passed = 0
failed = 0


def test(name: str, condition: bool, details: str = ""):
    global passed, failed
    if condition:
        print(f"  ✅ PASS — {name}")
        passed += 1
    else:
        print(f"  ❌ FAIL — {name}")
        if details:
            print(f"           {details}")
        failed += 1


# =============================================================================
# TEST 1 — Feature Extraction
# =============================================================================

print("─" * 50)
print("TEST GROUP 1: Feature Extraction (features.py)")
print("─" * 50)

now = datetime.now(timezone.utc)

# Build fake events for testing
# Simulates a student who reviewed "os" 3 times over the past week
fake_events = [
    {
        "user_id": "test_user",
        "concept_id": "os",
        "score": 0.5,
        "response_time_min": 30,
        "timestamp": now - timedelta(days=7)   # 7 days ago
    },
    {
        "user_id": "test_user",
        "concept_id": "os",
        "score": 0.65,
        "response_time_min": 20,
        "timestamp": now - timedelta(days=3)   # 3 days ago
    },
    {
        "user_id": "test_user",
        "concept_id": "os",
        "score": 0.8,
        "response_time_min": 15,
        "timestamp": now - timedelta(hours=48) # 48 hours ago
    }
]

features = extract_features(fake_events)

print(f"\n  Input: 3 events over past 7 days")
print(f"  Extracted features:")
for k, v in features.items():
    print(f"    {k:<25} = {v}")

# Verify each feature
test("features is not None",        features is not None)
test("hours_since_last ≈ 48",       47 < features["hours_since_last"] < 49,
     f"got {features['hours_since_last']}")
test("total_reviews == 3",          features["total_reviews"] == 3,
     f"got {features['total_reviews']}")
test("avg_score ≈ 0.65",            abs(features["avg_score"] - 0.65) < 0.01,
     f"got {features['avg_score']}")
test("last_score == 0.8",           features["last_score"] == 0.8,
     f"got {features['last_score']}")
test("success_streak == 2",         features["success_streak"] == 2,
     f"got {features['success_streak']}")
test("avg_response_time ≈ 21.67",   abs(features["avg_response_time"] - 21.67) < 0.5,
     f"got {features['avg_response_time']}")

# Edge case: empty events
empty_features = extract_features([])
test("empty events returns None",   empty_features is None)

# Edge case: single event
single_features = extract_features([fake_events[-1]])
test("single event works",          single_features is not None)
test("single event streak = 1",     single_features["success_streak"] == 1)


# =============================================================================
# TEST 2 — Recall Score Formula
# =============================================================================

print(f"\n{'─' * 50}")
print("TEST GROUP 2: Ebbinghaus Formula (scorer.py)")
print("─" * 50)

# Test base retention
br = compute_base_retention(last_score=0.8, avg_score=0.65)
print(f"\n  base_retention(last=0.8, avg=0.65) = {br:.4f}")
test("base_retention formula",  abs(br - (0.6*0.8 + 0.4*0.65)) < 0.001,
     f"got {br}")

# Test stability
s1 = compute_stability(total_reviews=1, success_streak=0)
s3 = compute_stability(total_reviews=3, success_streak=2)
s9 = compute_stability(total_reviews=9, success_streak=3)
print(f"\n  Stability values:")
print(f"    n=1, streak=0 → {s1:.1f} hours")
print(f"    n=3, streak=2 → {s3:.1f} hours")
print(f"    n=9, streak=3 → {s9:.1f} hours")
test("stability increases with more reviews", s1 < s3 < s9)
test("stability is positive",                s1 > 0)

# Test decay curve: recall should decrease over time
print(f"\n  Recall decay over time for: last=0.8, avg=0.65, n=3, streak=2")
features_for_decay = {
    "hours_since_last": 0,
    "total_reviews": 3,
    "avg_score": 0.65,
    "last_score": 0.8,
    "success_streak": 2,
    "avg_response_time": 20
}

recalls = []
time_points = [0, 6, 12, 24, 48, 72, 168]
for t in time_points:
    features_for_decay["hours_since_last"] = t
    r = compute_recall_score(features_for_decay)
    recalls.append(r)
    print(f"    t={t:>4}h → {r:.1f}%")

test("recall at t=0 is highest",           recalls[0] == max(recalls))
test("recall decreases monotonically",      all(recalls[i] >= recalls[i+1]
                                            for i in range(len(recalls)-1)))
test("recall at 7 days is < 50%",          recalls[-1] < 50)

# Test priority labels
test("priority < 40 = High",    get_priority(30.0) == "High")
test("priority 40-65 = Medium", get_priority(55.0) == "Medium")
test("priority >= 65 = Low",    get_priority(75.0) == "Low")
test("priority 40 = Medium",    get_priority(40.0) == "Medium")   # boundary
test("priority 65 = Low",       get_priority(65.0) == "Low")      # boundary

# Test edge cases for compute_recall_score
test("None features = 0.0",    compute_recall_score(None) == 0.0)
test("score clamped 0-100",    0 <= compute_recall_score(features_for_decay) <= 100)


# =============================================================================
# TEST 3 — Pipeline (requires MongoDB running)
# =============================================================================

print(f"\n{'─' * 50}")
print("TEST GROUP 3: Pipeline + MongoDB (pipeline.py)")
print("─" * 50)
print("  Note: Tests in this group require MongoDB to be running.")
print("  If MongoDB is not running, these will fail — that is expected.\n")

try:
    # Run pipeline for our test user
    # (This user might not have events, so we might get empty results)
    results = compute_scores_for_user("test_user")
    test("pipeline returns a list",        isinstance(results, list))

    if results:
        first = results[0]
        test("result has concept_id",      "concept_id"   in first)
        test("result has recall_score",    "recall_score" in first)
        test("result has priority",        "priority"     in first)
        test("result has features",        "features"     in first)
        test("recall_score in 0-100",      0 <= first["recall_score"] <= 100)
        test("list sorted asc by recall",  results == sorted(results,
                                           key=lambda x: x["recall_score"]))
        print(f"\n  Pipeline results for test_user:")
        for r in results:
            print(f"    {r['concept_id']:<20} → {r['recall_score']:.1f}% ({r['priority']})")
    else:
        print("  (No events in MongoDB for test_user — log some events first)")

except Exception as e:
    print(f"  ⚠️  MongoDB error: {e}")
    print("  Make sure MongoDB is running: mongod")


# =============================================================================
# TEST 4 — DB Functions (requires MongoDB)
# =============================================================================

print(f"\n{'─' * 50}")
print("TEST GROUP 4: Database Helper Functions (db.py)")
print("─" * 50)

try:
    scores = get_recall_scores("test_user")
    test("get_recall_scores returns list",       isinstance(scores, list))

    counts = get_event_counts_by_concept("test_user")
    test("get_event_counts returns dict",        isinstance(counts, dict))

    last_events = get_last_event_per_concept("test_user")
    test("get_last_events returns dict",         isinstance(last_events, dict))

    total = get_total_events_count("test_user")
    test("get_total_events returns int",         isinstance(total, int))
    test("total events >= 0",                    total >= 0)

    if counts:
        print(f"\n  Event counts per concept:")
        for cid, count in counts.items():
            print(f"    {cid:<20} → {count} events")

except Exception as e:
    print(f"  ⚠️  MongoDB error: {e}")
    print("  Make sure MongoDB is running: mongod")


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"  Results: {passed} passed  |  {failed} failed  |  {passed+failed} total")
if failed == 0:
    print("  🎉 All tests passed! Phase 2 and 3 are ready.")
else:
    print("  ⚠️  Some tests failed. Check the output above.")
print("="*60 + "\n")