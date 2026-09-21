from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"
POSE_ASSET_DIR = APP_PATH.parent / "assets" / "pose_examples"


def test_each_supported_finger_has_its_own_avatar_guide() -> None:
    guide_filenames = (
        "index_extension_avatar_hand.png",
        "index_flexion_avatar_hand.png",
        "middle_extension_avatar_hand.png",
        "middle_flexion_avatar_hand.png",
        "ring_extension_avatar_hand.png",
        "ring_flexion_avatar_hand.png",
        "little_extension_avatar_hand.png",
        "little_flexion_avatar_hand.png",
    )
    for filename in guide_filenames:
        guide_path = POSE_ASSET_DIR / filename
        assert guide_path.is_file()
        assert guide_path.stat().st_size > 100_000


def test_index_guide_uses_two_separate_pose_assets() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    assert '("Maximum active extension", "index_extension_avatar_hand.png")' in source
    assert '("Maximum active flexion", "index_flexion_avatar_hand.png")' in source


def test_middle_guide_uses_two_separate_pose_assets() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    assert '("Maximum active extension", "middle_extension_avatar_hand.png")' in source
    assert '("Maximum active flexion", "middle_flexion_avatar_hand.png")' in source


def test_ring_and_little_guides_use_two_separate_pose_assets() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    for finger in ("ring", "little"):
        assert f'("Maximum active extension", "{finger}_extension_avatar_hand.png")' in source
        assert f'("Maximum active flexion", "{finger}_flexion_avatar_hand.png")' in source


def test_start_measurement_does_not_require_acknowledgment() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()
    assert not app.exception
    assert not any(expander.label == "Privacy and limitations" for expander in app.expander)
    assert not any(
        checkbox.label == "I understand this is an experimental estimate."
        for checkbox in app.checkbox
    )
    start = next(button for button in app.button if button.label == "Start measurement")
    assert not start.disabled
    start.click().run()
    assert any(title.value == "New measurement" for title in app.title)


def test_upload_screen_has_simplified_settings_and_per_finger_capture_groups() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()
    app.session_state["wizard_step"] = 2
    app.run()
    assert not any(field.label == "Case ID" for field in app.text_input)
    assert not any(checkbox.label == "Mirrored photos" for checkbox in app.checkbox)
    assert not any(
        checkbox.label == "I confirm the index images meet the side-view acceptance checklist."
        for checkbox in app.checkbox
    )
    assert app.session_state["participant_id"].startswith("CASE-")
    assert app.session_state["mirrored"] is False
    captions = {caption.value for caption in app.caption}
    assert any("صوّر إصبعًا واحدًا" in caption for caption in captions)
    assert any(
        "Match the avatar's extension and flexion positions with the index finger"
        in caption
        for caption in captions
    )
    uploader_labels = {uploader.label for uploader in app.file_uploader}
    assert len(uploader_labels) == 2
    assert "Add 1–3 index extension photos" in uploader_labels
    assert "Add 1–3 index flexion photos" in uploader_labels
    assert not any(title.value == "Session setup" for title in app.title)

    finger_selector = next(selectbox for selectbox in app.selectbox if selectbox.label == "Finger to measure")
    finger_selector.set_value("middle").run()
    captions = {caption.value for caption in app.caption}
    assert any(
        "Match the avatar's extension and flexion positions with the middle finger"
        in caption
        for caption in captions
    )
    assert not any(
        "Match the avatar's extension and flexion positions with the index finger"
        in caption
        for caption in captions
    )
    uploader_labels = {uploader.label for uploader in app.file_uploader}
    assert uploader_labels == {
        "Add 1–3 middle extension photos",
        "Add 1–3 middle flexion photos",
    }

    for finger in ("ring", "little", "index"):
        finger_selector = next(
            selectbox
            for selectbox in app.selectbox
            if selectbox.label == "Finger to measure"
        )
        finger_selector.set_value(finger).run()
        assert any(
            markdown.value == f"### {finger.title()} finger avatar pose guide"
            for markdown in app.markdown
        )
        captions = {caption.value for caption in app.caption}
        assert any(
            f"Match the avatar's extension and flexion positions with the {finger} finger"
            in caption
            for caption in captions
        )

    hand_selector = next(radio for radio in app.radio if radio.label == "Hand")
    hand_selector.set_value("Left").run()
    hand_selector = next(radio for radio in app.radio if radio.label == "Hand")
    assert hand_selector.value == "Left"


def test_missing_pose_validation() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()
    app.session_state["wizard_step"] = 2
    app.session_state["participant_id"] = "P-TEST"
    app.run()
    next(button for button in app.button if button.label == "Analyze Images").click().run()
    assert any("Index maximum-extension" in error.value for error in app.error)


def test_failed_result_displays_the_tam_rejection_reason() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()
    app.session_state["wizard_step"] = 3
    app.session_state["analyses"] = []
    app.session_state["tam_results"] = None
    app.session_state["tam_error"] = "TAM requires non-Low extension and flexion quality."
    app.run()
    assert any(
        "Reason: TAM requires non-Low extension and flexion quality." in error.value
        for error in app.error
    )


def test_demo_results_layout_and_actions(monkeypatch) -> None:
    monkeypatch.setenv("HANDROM_DEMO", "1")
    app = AppTest.from_file(str(APP_PATH)).run()
    app.session_state["wizard_step"] = 2
    app.session_state["participant_id"] = "DEMO-TEST"
    app.run()
    next(
        button for button in app.button if button.label == "Load synthetic demo data"
    ).click().run()
    assert any(title.value == "Results" for title in app.title)
    assert app.dataframe
    assert list(app.dataframe[0].value.columns) == [
        "Total flexion",
        "Total extension deficit",
        "Estimated TAM",
    ]
    assert len(app.metric) == 12
    assert {button.label for button in app.download_button} == {"PDF report"}
    assert {button.label for button in app.button} == {"Edit photos", "Clear session"}
