"""Streamlit session-state initialization and privacy-preserving reset."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

SESSION_DEFAULTS: dict[str, object] = {
    "wizard_step": 1,
    "session_id": "",
    "session_timestamp_utc": "",
    "participant_id": "",
    "session_label": "",
    "hand_side": "",
    "selected_finger": "index",
    "mirrored": False,
    "notes": "",
    "analyses": [],
    "uploaded_extension": [],
    "uploaded_flexion": [],
    "extension_aggregation": None,
    "flexion_aggregation": None,
    "tam_results": None,
    "tam_error": None,
    "demo_loaded": False,
}


def initialize_session_state(state: object) -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in state:  # type: ignore[operator]
            if key == "session_id":
                value = str(uuid4())
            elif key == "session_timestamp_utc":
                value = datetime.now(UTC).isoformat()
            elif key == "participant_id":
                value = f"CASE-{str(uuid4())[:8].upper()}"
            state[key] = value  # type: ignore[index]
    if not state["participant_id"]:  # type: ignore[index]
        state["participant_id"] = f"CASE-{str(uuid4())[:8].upper()}"  # type: ignore[index]


def clear_session_state(state: object) -> None:
    """Remove patient-specific in-memory data and return to the welcome screen."""
    for key in list(state.keys()):  # type: ignore[attr-defined]
        del state[key]  # type: ignore[index]
    initialize_session_state(state)


def notes_may_contain_identifiers(notes: str) -> bool:
    """Deliberately simple privacy reminder, not medical-data classification."""
    lowered = notes.lower()
    identifier_terms = ("email", "phone", "address", "dob", "date of birth", "@")
    digit_count = sum(character.isdigit() for character in notes)
    return any(term in lowered for term in identifier_terms) or digit_count >= 8
