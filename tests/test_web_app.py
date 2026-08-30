from pathlib import Path
from io import BytesIO
import os

import pytest

import app as app_module
from app import app
from voice_to_text.core import Segment, TranscriptionResult


def test_local_web_interface_loads() -> None:
    client = app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    assert b"Private Conversation Transcriber" in response.data
    assert b"Keep the app's local copy" in response.data
    assert b"127.0.0.1" not in response.data


def test_gui_downloads_do_not_require_console_progress_streams() -> None:
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"


def test_transcription_requires_audio() -> None:
    client = app.test_client()
    response = client.post("/api/transcribe", data={})

    assert response.status_code == 400
    assert response.get_json()["error"] == "No audio recording was supplied"


def test_cross_site_local_action_is_blocked() -> None:
    client = app.test_client()
    response = client.post(
        "/api/token/forget",
        headers={"Origin": "https://unrelated.example"},
    )

    assert response.status_code == 403


def test_gated_model_error_has_actionable_instructions() -> None:
    message = app_module._friendly_error(RuntimeError("401: Cannot access gated repo"))

    assert "accept its access" in message
    assert "Create a new read token" in message
    assert "Do not paste the token into chat" in message


def test_invalid_new_token_is_rejected_before_audio_is_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RejectingApi:
        def whoami(self, token: str) -> None:
            raise RuntimeError("Invalid user token")

    monkeypatch.setattr(app_module, "HfApi", RejectingApi)
    monkeypatch.setattr(app_module, "RECORDINGS_DIR", tmp_path)
    client = app.test_client()
    response = client.post(
        "/api/transcribe",
        data={
            "audio": (BytesIO(b"private audio"), "conversation.wav"),
            "diarize": "true",
            "token": "invalid-token",
            "remember_token": "false",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 401
    assert "Create a new read token" in response.get_json()["error"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(("keep_audio", "should_exist"), [("false", False), ("true", True)])
def test_audio_copy_retention_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep_audio: str, should_exist: bool
) -> None:
    source = tmp_path / "conversation.wav"
    source.write_bytes(b"temporary recording")
    output_dir = tmp_path / "outputs"

    class FakeTranscriber:
        def __init__(self, **_kwargs) -> None:
            self.progress_callback = None

        def transcribe(self, audio_path, _language, _prompt) -> TranscriptionResult:
            return TranscriptionResult(
                source=Path(audio_path),
                model="test",
                device="cpu",
                language="en",
                language_probability=1.0,
                duration=1.0,
                segments=(Segment(0.0, 1.0, "hello"),),
            )

    def fake_write_outputs(_result, destination):
        destination.mkdir(parents=True, exist_ok=True)
        transcript = destination / "conversation.transcript.txt"
        transcript.write_text("hello\n", encoding="utf-8")
        return {"txt": transcript}

    monkeypatch.setattr(app_module, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(app_module, "write_outputs", fake_write_outputs)
    monkeypatch.setattr(app_module, "TRANSCRIPTS_DIR", output_dir)
    app_module.TRANSCRIBERS.clear()
    job_id = f"retention-{keep_audio}"
    app_module.JOBS[job_id] = {}
    settings = {
        "engine": "whisper",
        "model": "small",
        "device": "cpu",
        "language": "en",
        "prompt": "",
        "diarize": "false",
        "remember_token": "false",
        "keep_audio": keep_audio,
        "speaker1": "Speaker 1",
        "speaker2": "Speaker 2",
        "token": "",
    }

    app_module._run_job(job_id, source, settings)

    assert source.exists() is should_exist
    assert app_module.JOBS[job_id]["status"] == "done"
