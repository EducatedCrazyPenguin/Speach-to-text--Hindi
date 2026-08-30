from __future__ import annotations

from dataclasses import replace
import math
import re
import unicodedata
from typing import Iterable, Sequence

import numpy as np

from .core import Segment, WordTiming


_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\S+")


def normalized_token(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    return "".join(ch for ch in text if ch.isalnum() or "\u0900" <= ch <= "\u097f")


def proportional_word_timings(
    text: str,
    start: float,
    end: float,
    confidence: float | None = None,
) -> tuple[WordTiming, ...]:
    """Create monotonic fallback timings while retaining the original word text."""
    matches = list(_TOKEN_RE.finditer(text.strip()))
    if not matches or end <= start:
        return ()
    weights = [max(1, len(match.group(0))) for match in matches]
    total = float(sum(weights))
    cursor = start
    words: list[WordTiming] = []
    for index, (match, weight) in enumerate(zip(matches, weights, strict=True)):
        word_end = end if index == len(matches) - 1 else cursor + (end - start) * weight / total
        words.append(WordTiming(cursor, max(cursor + 0.001, word_end), match.group(0), confidence))
        cursor = word_end
    return tuple(words)


def speech_weighted_word_timings(
    text: str,
    waveform: np.ndarray,
    start: float,
    confidence: float | None = None,
    sample_rate: int = 16_000,
) -> tuple[WordTiming, ...]:
    """Distribute fallback word times over active speech, skipping silent gaps.

    This remains a fallback rather than forced alignment, but it is substantially
    safer for speaker attribution than spreading a Qwen chunk uniformly across
    telephone hold noise and pauses.
    """
    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    end = start + audio.size / sample_rate
    matches = list(_TOKEN_RE.finditer(text.strip()))
    frame = max(1, int(0.020 * sample_rate))
    hop = max(1, int(0.010 * sample_rate))
    if not matches or audio.size < frame * 2:
        return proportional_word_timings(text, start, end, confidence)

    frame_count = 1 + (audio.size - frame) // hop
    indices = np.arange(frame, dtype=np.int64)[None, :] + hop * np.arange(frame_count)[:, None]
    rms = np.sqrt(np.mean(np.square(audio[indices], dtype=np.float32), axis=1) + 1e-10)
    levels = 20.0 * np.log10(rms + 1e-10)
    floor = float(np.percentile(levels, 20))
    speech = float(np.percentile(levels, 90))
    if speech - floor < 6.0:
        return proportional_word_timings(text, start, end, confidence)
    active = levels >= floor + min(12.0, max(4.0, 0.28 * (speech - floor)))
    # Add 30 ms of context to avoid clipping low-energy consonants.
    active = np.convolve(active.astype(np.int8), np.ones(7, dtype=np.int8), mode="same") > 0
    active_ratio = float(np.mean(active))
    if active_ratio < 0.10 or active_ratio > 0.96:
        return proportional_word_timings(text, start, end, confidence)

    intervals: list[tuple[float, float]] = []
    interval_start: int | None = None
    for index, is_active in enumerate(active):
        if is_active and interval_start is None:
            interval_start = index * hop
        if interval_start is not None and (not is_active or index == len(active) - 1):
            final_index = index if not is_active else index + 1
            interval_end = min(audio.size, (final_index - 1) * hop + frame)
            intervals.append((interval_start / sample_rate, interval_end / sample_rate))
            interval_start = None
    total_active = sum(right - left for left, right in intervals)
    if total_active <= 0:
        return proportional_word_timings(text, start, end, confidence)

    def active_offset_to_time(offset: float, *, boundary_to_next: bool = False) -> float:
        remaining = min(total_active, max(0.0, offset))
        for index, (left, right) in enumerate(intervals):
            duration = right - left
            if remaining < duration or (remaining == duration and not boundary_to_next):
                return start + left + remaining
            remaining -= duration
            if remaining <= 1e-9 and boundary_to_next and index + 1 < len(intervals):
                return start + intervals[index + 1][0]
        return start + intervals[-1][1]

    weights = [max(1, len(match.group(0))) for match in matches]
    total_weight = float(sum(weights))
    cumulative_weights = np.cumsum([0, *weights], dtype=np.float64)
    offsets = [total_active * float(value) / total_weight for value in cumulative_weights]
    active_cursor = 0.0
    pause_boundaries: list[float] = []
    for index, (left, right) in enumerate(intervals[:-1]):
        active_cursor += right - left
        if intervals[index + 1][0] - right >= 0.25:
            pause_boundaries.append(active_cursor)
    for index in range(1, len(offsets) - 1):
        if not pause_boundaries:
            break
        nearest = min(pause_boundaries, key=lambda boundary: abs(boundary - offsets[index]))
        if (
            abs(nearest - offsets[index]) <= 0.45
            and offsets[index - 1] + 0.02 < nearest < offsets[index + 1] - 0.02
        ):
            offsets[index] = nearest

    words: list[WordTiming] = []
    for index, (match, weight) in enumerate(zip(matches, weights, strict=True)):
        word_start = active_offset_to_time(offsets[index], boundary_to_next=True)
        word_end = active_offset_to_time(offsets[index + 1])
        if index == len(matches) - 1:
            word_end = start + intervals[-1][1]
        words.append(WordTiming(word_start, max(word_start + 0.001, word_end), match.group(0), confidence))
    return tuple(words)


def token_timings_to_words(
    *,
    text: str,
    start: float,
    end: float,
    tokens: Sequence[str] | None,
    timestamps: Sequence[float] | None,
    logprobs: Sequence[float] | None = None,
) -> tuple[WordTiming, ...]:
    """Convert ONNX token timestamps to word timings with a safe proportional fallback.

    NeMo exports expose BPE-token rather than word timestamps. We use those timestamps
    to create word boundaries only when their shape and range are valid; malformed or
    zero-duration output is deliberately ignored.
    """
    fallback = proportional_word_timings(text, start, end)
    if not tokens or not timestamps or len(tokens) != len(timestamps):
        return fallback
    numeric = [float(value) for value in timestamps]
    if any(not math.isfinite(value) for value in numeric) or any(
        later < earlier for earlier, later in zip(numeric, numeric[1:])
    ):
        return fallback

    duration = end - start
    relative = max(numeric, default=0.0) <= duration + 1.0
    absolute = [value + start if relative else value for value in numeric]
    absolute = [min(end, max(start, value)) for value in absolute]

    pieces: list[tuple[str, float, float | None]] = []
    current_text = ""
    current_start = absolute[0]
    current_probs: list[float] = []
    for index, token in enumerate(tokens):
        token = str(token).replace("\u2581", " ")
        begins_word = bool(token[:1].isspace())
        if begins_word and current_text.strip():
            confidence = sum(current_probs) / len(current_probs) if current_probs else None
            pieces.append((current_text.strip(), current_start, confidence))
            current_text = ""
            current_probs = []
            current_start = absolute[index]
        current_text += token
        if logprobs and index < len(logprobs):
            current_probs.append(float(math.exp(min(0.0, float(logprobs[index])))))
    if current_text.strip():
        confidence = sum(current_probs) / len(current_probs) if current_probs else None
        pieces.append((current_text.strip(), current_start, confidence))

    recognized_words = [item.group(0) for item in _TOKEN_RE.finditer(text.strip())]
    if len(pieces) != len(recognized_words):
        return fallback
    words: list[WordTiming] = []
    for index, ((_, word_start, confidence), recognized) in enumerate(
        zip(pieces, recognized_words, strict=True)
    ):
        word_end = pieces[index + 1][1] if index + 1 < len(pieces) else end
        if word_end <= word_start:
            return fallback
        words.append(WordTiming(word_start, word_end, recognized, confidence))
    return tuple(words)


def validate_word_timings(
    words: Iterable[WordTiming], duration: float
) -> tuple[WordTiming, ...]:
    valid: list[WordTiming] = []
    last_start = 0.0
    for word in words:
        if not word.text.strip():
            continue
        start = min(duration, max(last_start, float(word.start)))
        end = min(duration, max(start, float(word.end)))
        if not math.isfinite(start) or not math.isfinite(end) or end - start < 0.001:
            continue
        valid.append(replace(word, start=start, end=end))
        last_start = start
    return tuple(valid)


def deduplicate_overlapping_segments(
    segments: Sequence[Segment], max_overlap_words: int = 12
) -> tuple[Segment, ...]:
    """Remove repeated suffix/prefix words created by overlapping ASR chunks."""
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    cleaned: list[Segment] = []
    previous_tokens: list[str] = []
    for segment in ordered:
        words = list(segment.words or proportional_word_timings(segment.text, segment.start, segment.end))
        tokens = [normalized_token(word.text) for word in words]
        duplicate = 0
        if cleaned and segment.start < cleaned[-1].end + 0.05:
            limit = min(max_overlap_words, len(previous_tokens), len(tokens))
            for size in range(limit, 0, -1):
                if previous_tokens[-size:] == tokens[:size] and all(tokens[:size]):
                    duplicate = size
                    break
        words = words[duplicate:]
        if not words:
            continue
        text = " ".join(word.text.strip() for word in words).strip()
        cleaned.append(
            replace(segment, start=words[0].start, text=text, words=tuple(words))
        )
        previous_tokens.extend(normalized_token(word.text) for word in words)
    return tuple(cleaned)


def flatten_words(segments: Iterable[Segment]) -> tuple[WordTiming, ...]:
    words: list[WordTiming] = []
    for segment in segments:
        words.extend(segment.words or proportional_word_timings(segment.text, segment.start, segment.end))
    return tuple(words)


def words_to_segments(words: Iterable[WordTiming], max_gap: float = 0.8) -> tuple[Segment, ...]:
    """Group words into readable speaker turns after diarization."""
    grouped: list[list[WordTiming]] = []
    for word in sorted(words, key=lambda item: (item.start, item.end)):
        if (
            not grouped
            or grouped[-1][-1].speaker != word.speaker
            or grouped[-1][-1].overlap != word.overlap
            or word.start - grouped[-1][-1].end > max_gap
        ):
            grouped.append([word])
        else:
            grouped[-1].append(word)
    return tuple(
        Segment(
            group[0].start,
            group[-1].end,
            _SPACE_RE.sub(" ", " ".join(word.text for word in group)).strip(),
            speaker=group[0].speaker,
            confidence=(
                sum(word.confidence for word in group if word.confidence is not None)
                / len([word for word in group if word.confidence is not None])
                if any(word.confidence is not None for word in group)
                else None
            ),
            overlap=group[0].overlap,
            words=tuple(group),
        )
        for group in grouped
        if group
    )
