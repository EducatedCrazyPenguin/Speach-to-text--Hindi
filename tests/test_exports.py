import json
from pathlib import Path

from voice_to_text.core import Segment, TranscriptionResult
from voice_to_text.exports import timestamp, write_outputs


def test_timestamp_formats() -> None:
    assert timestamp(65.432) == "00:01:05.432"
    assert timestamp(3661.007, srt=True) == "01:01:01,007"


def test_writes_utf8_transcript_subtitles_and_json(tmp_path: Path) -> None:
    source = tmp_path / "बातचीत.wav"
    source.touch()
    result = TranscriptionResult(
        source=source,
        model="small",
        device="cpu",
        language="hi",
        language_probability=0.98,
        duration=4.0,
        segments=(
            Segment(0.0, 1.5, "नमस्ते", "Mohit"),
            Segment(2.0, 3.5, "Hello", "Wife"),
        ),
    )

    outputs = write_outputs(result, tmp_path / "out")

    text = outputs["txt"].read_text(encoding="utf-8")
    assert "Mohit: नमस्ते" in text
    assert "Duration: 00:00:04.000" in text
    assert "00:00:00,000 --> 00:00:01,500" in outputs["srt"].read_text(encoding="utf-8")
    data = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert data["text"] == "नमस्ते Hello"
    assert data["segments"][1]["start"] == 2.0
    assert data["segments"][1]["speaker"] == "Wife"
