from __future__ import annotations

import json
from pathlib import Path

from .core import Segment, TranscriptionResult, WordTiming


def timestamp(seconds: float, *, srt: bool = False) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    separator = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _text_document(
    result: TranscriptionResult,
    segments: tuple[Segment, ...] | None = None,
) -> str:
    confidence = f"{result.language_probability:.0%}" if result.language_probability else "n/a"
    lines = [
        f"Source: {result.source.name}",
        f"Language: {result.language} (confidence: {confidence})",
        f"Model: {result.model}",
        f"Device: {result.device}",
        f"Duration: {timestamp(result.duration)}",
        "",
    ]
    for segment in segments if segments is not None else result.segments:
        speaker = f"{segment.speaker}: " if segment.speaker else ""
        overlap = "[overlap] " if segment.overlap else ""
        uncertain = "[uncertain] " if segment.uncertain else ""
        lines.append(f"[{timestamp(segment.start)}] {speaker}{overlap}{uncertain}{segment.text}")
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


def result_to_dict(result: TranscriptionResult) -> dict:
    def segment_payload(item: Segment) -> dict:
        return {
            "start": item.start,
            "end": item.end,
            "text": item.text,
            "speaker": item.speaker,
            "confidence": item.confidence,
            "overlap": item.overlap,
            "uncertain": item.uncertain,
            "words": [
                {
                    "start": word.start,
                    "end": word.end,
                    "text": word.text,
                    "confidence": word.confidence,
                    "speaker": word.speaker,
                    "overlap": word.overlap,
                    "origin": word.origin,
                    "alternatives": list(word.alternatives),
                    "uncertain": word.uncertain,
                    "repetition_removed": word.repetition_removed,
                }
                for word in item.words
            ],
        }

    return {
        "source": str(result.source),
        "model": result.model,
        "device": result.device,
        "language": result.language,
        "language_probability": result.language_probability,
        "duration_seconds": result.duration,
        "text": result.text,
        "readable_text": result.readable_text,
        "diagnostics": dict(result.diagnostics),
        "provenance": list(result.provenance),
        "segments": [segment_payload(item) for item in result.segments],
        "raw_segments": [segment_payload(item) for item in result.raw_segments],
    }


def result_from_dict(payload: dict) -> TranscriptionResult:
    segments = []
    for item in payload.get("segments", []):
        words = tuple(
            WordTiming(
                start=float(word["start"]),
                end=float(word["end"]),
                text=str(word["text"]),
                confidence=word.get("confidence"),
                speaker=word.get("speaker"),
                overlap=bool(word.get("overlap", False)),
                origin=str(word.get("origin", "primary")),
                alternatives=tuple(str(value) for value in word.get("alternatives", [])),
                uncertain=bool(word.get("uncertain", False)),
                repetition_removed=bool(word.get("repetition_removed", False)),
            )
            for word in item.get("words", [])
        )
        segments.append(
            Segment(
                start=float(item["start"]),
                end=float(item["end"]),
                text=str(item["text"]),
                speaker=item.get("speaker"),
                confidence=item.get("confidence"),
                overlap=bool(item.get("overlap", False)),
                words=words,
                uncertain=bool(item.get("uncertain", False)),
            )
        )
    raw_segments = []
    for item in payload.get("raw_segments", []):
        raw_segments.append(
            Segment(
                start=float(item["start"]),
                end=float(item["end"]),
                text=str(item["text"]),
                speaker=item.get("speaker"),
                confidence=item.get("confidence"),
                overlap=bool(item.get("overlap", False)),
                words=tuple(
                    WordTiming(
                        start=float(word["start"]),
                        end=float(word["end"]),
                        text=str(word["text"]),
                        confidence=word.get("confidence"),
                        speaker=word.get("speaker"),
                        overlap=bool(word.get("overlap", False)),
                        origin=str(word.get("origin", "primary")),
                        alternatives=tuple(str(value) for value in word.get("alternatives", [])),
                        uncertain=bool(word.get("uncertain", False)),
                        repetition_removed=bool(word.get("repetition_removed", False)),
                    )
                    for word in item.get("words", [])
                ),
                uncertain=bool(item.get("uncertain", False)),
            )
        )
    return TranscriptionResult(
        source=Path(payload["source"]),
        model=str(payload.get("model", "unknown")),
        device=str(payload.get("device", "unknown")),
        language=str(payload.get("language", "unknown")),
        language_probability=float(payload.get("language_probability", 0.0)),
        duration=float(payload.get("duration_seconds", 0.0)),
        segments=tuple(segments),
        diagnostics=payload.get("diagnostics", {}),
        provenance=tuple(payload.get("provenance", [])),
        readable_text=payload.get("readable_text"),
        raw_segments=tuple(raw_segments),
    )


def _json_document(result: TranscriptionResult) -> str:
    payload = result_to_dict(result)
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
        overlap = " `[overlap]`" if segment.overlap else ""
        lines.append(f"- `{timestamp(segment.start)}`{speaker}{overlap} {segment.text}")
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(result: TranscriptionResult, output_dir: str | Path) -> dict[str, Path]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{result.source.stem}.transcript"
    paths = {
        "txt": destination / f"{stem}.txt",
        "raw": destination / f"{result.source.stem}.raw.txt",
        "verbatim": destination / f"{result.source.stem}.verbatim.txt",
        "readable": destination / f"{result.source.stem}.readable.txt",
        "srt": destination / f"{stem}.srt",
        "json": destination / f"{stem}.json",
        "md": destination / f"{stem}.md",
    }
    paths["txt"].write_text(_text_document(result), encoding="utf-8")
    paths["verbatim"].write_text(_text_document(result), encoding="utf-8")
    raw_segments = result.raw_segments or result.segments
    paths["raw"].write_text(_text_document(result, raw_segments), encoding="utf-8")
    readable = result.readable_text or result.text
    paths["readable"].write_text(
        f"Source: {result.source.name}\nType: readable standard-Hindi copy\n\n{readable.strip()}\n",
        encoding="utf-8",
    )
    paths["srt"].write_text(_srt_document(result.segments), encoding="utf-8")
    paths["json"].write_text(_json_document(result), encoding="utf-8")
    paths["md"].write_text(_markdown_document(result), encoding="utf-8")
    return paths
