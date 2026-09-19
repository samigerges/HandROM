from __future__ import annotations

from handrom.session_manager import (
    clear_session_state,
    initialize_session_state,
    notes_may_contain_identifiers,
)


def test_initialize_and_clear_session() -> None:
    state: dict[str, object] = {}
    initialize_session_state(state)
    old_session_id = state["session_id"]
    state["analyses"] = ["patient-specific"]
    state["wizard_step"] = 5
    clear_session_state(state)
    assert state["analyses"] == []
    assert state["wizard_step"] == 1
    assert state["session_id"] != old_session_id


def test_simple_notes_privacy_warning() -> None:
    assert notes_may_contain_identifiers("email: person@example.test")
    assert notes_may_contain_identifiers("phone 01234567890")
    assert not notes_may_contain_identifiers("repeat measurement in indoor lighting")
