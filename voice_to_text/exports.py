from __future__ import annotations

import json
from pathlib import Path

from .core import Segment, TranscriptionResult


def timestamp(seconds: float, *, srt: bool = False) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    separator = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _text_document(result: TranscriptionResult) -> str:
    confidence = f"{result.language_probability:.0%}" if result.language_probability else "n/a"
    lines = [
        f"Source: {result.source.name}",
        f"Language: {result.language} (confidence: {confidence})",
        f"Model: {result.model}",
        f"Device: {result.device}",
        f"Duration: {timestamp(result.duration)}",
        "",
    ]
    for segment in result.segments:
        speaker = f"{segment.speaker}: " if segment.speaker else ""
        lines.append(f"[{timestamp(segment.start)}] {speaker}{segment.text}")
    return "\n".join(lines).rstrip() + "\n"


def _srt_document(segments: tuple[Segment, ...]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        speaker = f"{segment.speaker}: " if segment.speaker else ""
        blocks.append(
            f"{index}\n{timestamp(segment.start, srt=True)} --> "
            f"{timestamp(segment.end, srt=True)}\n{speaker}{segment.text}"
        )
    return "\n\n".join(blocks).rstrip() + "\n"


def _json_document(result: TranscriptionResult) -> str:
    payload = {
        "source": str(result.source),
        "model": result.model,
        "device": result.device,
        "language": result.language,
        "language_probability": result.language_probability,
        "duration_seconds": result.duration,
        "text": result.text,
        "segments": [
            {"start": item.start, "end": item.end, "text": item.text, "speaker": item.speaker}
            for item in result.segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _markdown_document(result: TranscriptionResult) -> str:
    confidence = f"{result.language_probability:.0%}" if result.language_probability else "n/a"
    lines = [
        f"# Transcript: {result.source.name}",
        "",
        f"- Language: `{result.language}` (confidence: {confidence})",
        f"- Model: `{result.model}`",
        f"- Device: `{result.device}`",
        f"- Duration: `{timestamp(result.duration)}`",
        "",
        "## Conversation",
        "",
    ]
    for segment in result.segments:
        speaker = f" **{segment.speaker}:**" if segment.speaker else ""
        lines.append(f"- `{timestamp(segment.start)}`{speaker} {segment.text}")
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(result: TranscriptionResult, output_dir: str | Path) -> dict[str, Path]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{result.source.stem}.transcript"
    paths = {
        "txt": destination / f"{stem}.txt",
        "srt": destination / f"{stem}.srt",
        "json": destination / f"{stem}.json",
        "md": destination / f"{stem}.md",
    }
    paths["txt"].write_text(_text_document(result), encoding="utf-8")
    paths["srt"].write_text(_srt_document(result.segments), encoding="utf-8")
    paths["json"].write_text(_json_document(result), encoding="utf-8")
    paths["md"].write_text(_markdown_document(result), encoding="utf-8")
    return paths
