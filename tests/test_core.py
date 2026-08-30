from pathlib import Path
from types import SimpleNamespace

from voice_to_text.core import Transcriber


def test_whisper_uses_hotwords_and_bounds_timestamps_to_audio() -> None:
    calls = {}

    class FakeModel:
        def transcribe(self, _source: str, **kwargs):
            calls.update(kwargs)
            segments = (
                SimpleNamespace(start=0.1, end=2.0, text=" valid speech "),
                SimpleNamespace(start=4.0, end=29.0, text=" hallucinated "),
            )
            info = SimpleNamespace(duration=3.0, language="hi", language_probability=0.9)
            return segments, info

    transcriber = Transcriber(model_size="medium", device="cpu")
    transcriber._model = FakeModel()
    result = transcriber._transcribe_whisper(Path("short.mp3"), None, "Mohit, family name")

    assert calls["hotwords"] == "Mohit, family name"
    assert "initial_prompt" not in calls
    assert calls["multilingual"] is True
    assert calls["word_timestamps"] is True
    assert result.text == "valid speech"
    assert result.segments[0].end <= result.duration


def test_selected_language_disables_per_segment_redetection() -> None:
    calls = {}

    class FakeModel:
        def transcribe(self, _source: str, **kwargs):
            calls.update(kwargs)
            return (), SimpleNamespace(duration=1.0, language="hi", language_probability=1.0)

    transcriber = Transcriber(model_size="medium", device="cpu")
    transcriber._model = FakeModel()
    transcriber._transcribe_whisper(Path("short.mp3"), "hi", None)

    assert calls["language"] == "hi"
    assert calls["multilingual"] is False
