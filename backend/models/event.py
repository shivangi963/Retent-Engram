from datetime import datetime

def create_event(user_id: str, concept_id: str, event_type: str,
                 score: float, response_time_min: int, hints_used: int) -> dict:
    """
    Builds a clean event dict ready to insert into MongoDB.
    response_time is stored in MINUTES (not seconds) from Phase 1 onward.
    """
    return {
        "user_id": user_id.strip().lower(),
        "concept_id": concept_id.strip().lower(),
        "event_type": event_type,
        "score": round(float(score), 2),
        "response_time_min": int(response_time_min),
        "hints_used": int(hints_used),
        "timestamp": datetime.utcnow()
    }