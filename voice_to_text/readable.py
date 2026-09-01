from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterable, Sequence

from .core import Segment, TranscriptionResult
from .metrics import normalize_text, numbers_match_exactly


READABLE_MODEL = "Qwen/Qwen3.5-4B"
READABLE_RUNTIME = Path(__file__).resolve().parents[1] / ".cache" / "readable-transformers"


def _tokens(text: str) -> Counter[str]:
    return Counter(normalize_text(text, canonicalize_numbers=False).split())


def validate_readable_copy(source: str, rewritten: str, glossary: Iterable[str] = ()) -> bool:
    if not rewritten.strip() or not numbers_match_exactly(source, rewritten):
        return False
    source_tokens = _tokens(source)
    rewritten_tokens = _tokens(rewritten)
    retained = sum(min(count, rewritten_tokens[token]) for token, count in source_tokens.items())
    introduced = sum(max(0, count - source_tokens[token]) for token, count in rewritten_tokens.items())
    source_count = max(1, sum(source_tokens.values()))
    if retained / source_count < 0.85 or introduced / source_count > 0.15:
        return False
    source_normalized = normalize_text(source, canonicalize_numbers=False)
    rewritten_normalized = normalize_text(rewritten, canonicalize_numbers=False)
    for item in glossary:
        normalized = normalize_text(item, canonicalize_numbers=False)
        if normalized and (normalized in source_normalized) != (normalized in rewritten_normalized):
            return False
    return True


def validate_structured_turns(
    segments: Sequence[Segment],
    rewritten: Sequence[dict],
    glossary: Iterable[str] = (),
) -> tuple[str | None, ...] | None:
    if len(rewritten) != len(segments):
        return None
    output: list[str | None] = []
    for index, (segment, item) in enumerate(zip(segments, rewritten, strict=True), 1):
        if int(item.get("id", -1)) != index:
            return None
        text = str(item.get("text", "")).strip()
        output.append(text if validate_readable_copy(segment.text, text, glossary) else None)
    return tuple(output)


def _safe_fallback(result: TranscriptionResult) -> str:
    lines = []
    for segment in result.segments:
        prefix = f"{segment.speaker}: " if segment.speaker else ""
        uncertainty = "[uncertain] " if segment.uncertain else ""
        text = segment.text.strip()
        if text and text[-1] not in ".!?।":
            text += "।"
        lines.append(prefix + uncertainty + text)
    return "\n".join(lines).strip()


def _run_readable_worker(segments: Sequence[Segment]) -> list[dict]:
    if not READABLE_RUNTIME.is_dir():
        raise RuntimeError("The isolated Qwen3.5 runtime is not installed")
    request = {
        "model": READABLE_MODEL,
        "turns": [
            {"id": index, "speaker": segment.speaker or "", "text": segment.text}
            for index, segment in enumerate(segments, 1)
        ],
    }
    READABLE_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="readable-", dir=READABLE_RUNTIME.parent) as directory:
        root = Path(directory)
        input_path, output_path = root / "input.json", root / "output.json"
        input_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "voice_to_text.readable_worker",
                str(input_path),
                str(output_path),
                str(READABLE_RUNTIME),
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        if completed.returncode != 0 or not output_path.is_file():
            message = completed.stderr.strip()[-500:] or "Qwen3.5 readable worker failed"
            raise RuntimeError(message)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return list(payload.get("turns", []))


def generate_readable_copy(
    result: TranscriptionResult,
    glossary: Iterable[str] = (),
    *,
    use_local_model: bool = True,
    allow_download: bool = False,
) -> tuple[str, str]:
    """Create a separately validated readable copy; never change verbatim text."""
    fallback = _safe_fallback(result)
    if not use_local_model:
        return fallback, "punctuation-only safeguard"
    try:
        rewritten = _run_readable_worker(result.segments)
        validated = validate_structured_turns(result.segments, rewritten, glossary)
        if validated is None:
            return fallback, "punctuation-only safeguard (Qwen3.5 validation rejected)"
        lines = []
        rejected = 0
        for segment, text in zip(result.segments, validated, strict=True):
            prefix = f"{segment.speaker}: " if segment.speaker else ""
            if text is None:
                rejected += 1
                fallback_text = segment.text.rstrip(".!?।") + "।"
                lines.append(prefix + "[uncertain] " + fallback_text)
            else:
                lines.append(prefix + text)
        method = READABLE_MODEL if not rejected else f"{READABLE_MODEL}; {rejected} rejected turn(s)"
        return "\n".join(lines), method
    except Exception as exc:
        return fallback, f"punctuation-only safeguard ({type(exc).__name__})"
