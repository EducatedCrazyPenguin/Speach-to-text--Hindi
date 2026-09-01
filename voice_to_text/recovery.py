from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math
from typing import Iterable, Sequence

import numpy as np

from .alignment import flatten_words, normalized_token, words_to_segments
from .core import ProgressCallback, Segment, TranscriptionResult, WordTiming, _notify
from .forced_alignment import align_hindi_candidates
from .models import RetryTranscript, transcribe_retry_windows


@dataclass(frozen=True)
class RecoverySpan:
    start: float
    end: float
    reasons: tuple[str, ...]


def _tokens(text: str) -> list[str]:
    return [token for token in (normalized_token(item) for item in text.split()) if token]


def _near_repeat(left: str, right: str) -> bool:
    if left == right:
        return True
    shortest = min(len(left), len(right))
    return shortest >= 2 and (left.startswith(right) or right.startswith(left))


def repetition_count(text: str) -> int:
    tokens = _tokens(text)
    repeats = sum(_near_repeat(left, right) for left, right in zip(tokens, tokens[1:]))
    for size in (2, 3):
        for index in range(0, len(tokens) - 2 * size + 1):
            repeats += tokens[index : index + size] == tokens[index + size : index + 2 * size]
    return int(repeats)


def merge_recovery_spans(
    spans: Iterable[RecoverySpan],
    duration: float,
    *,
    nearby_seconds: float = 2.0,
    context_seconds: float = 4.0,
    minimum_seconds: float = 6.0,
    maximum_seconds: float = 18.0,
) -> tuple[RecoverySpan, ...]:
    ordered = sorted(spans, key=lambda item: (item.start, item.end))
    merged: list[RecoverySpan] = []
    for span in ordered:
        if merged and span.start - merged[-1].end <= nearby_seconds:
            previous = merged[-1]
            merged[-1] = RecoverySpan(
                previous.start,
                max(previous.end, span.end),
                tuple(sorted(set(previous.reasons + span.reasons))),
            )
        else:
            merged.append(span)
    padded: list[RecoverySpan] = []
    for span in merged:
        center = (span.start + span.end) / 2
        target_length = min(
            maximum_seconds,
            max(minimum_seconds, span.end - span.start + 2 * context_seconds),
        )
        start = max(0.0, center - target_length / 2)
        end = min(duration, start + target_length)
        start = max(0.0, end - target_length)
        padded.append(RecoverySpan(start, end, span.reasons))
    # Padding can make adjacent regions overlap; merge those once more without
    # allowing a retry window to exceed the stated 18-second limit.
    output: list[RecoverySpan] = []
    for span in padded:
        if output and span.start <= output[-1].end and span.end - output[-1].start <= maximum_seconds:
            previous = output[-1]
            output[-1] = RecoverySpan(
                previous.start,
                max(previous.end, span.end),
                tuple(sorted(set(previous.reasons + span.reasons))),
            )
        else:
            if output and span.start < output[-1].end:
                span = RecoverySpan(output[-1].end, span.end, span.reasons)
            if span.end - span.start >= 0.5:
                output.append(span)
    return tuple(output)


def detect_recovery_spans(result: TranscriptionResult) -> tuple[RecoverySpan, ...]:
    flagged: list[RecoverySpan] = []
    for segment in result.segments:
        words = tuple(segment.words)
        reasons: list[str] = []
        if repetition_count(segment.text):
            reasons.append("repetition")
        scores = [word.confidence for word in words if word.confidence is not None]
        if len(words) >= 2 and scores and sum(scores) / len(scores) < 0.18:
            reasons.append("low_alignment")
        if len(words) and (segment.end - segment.start) / len(words) > 1.15:
            reasons.append("silence_to_word_ratio")
        if len(words) <= 2 and segment.end - segment.start < 0.8:
            reasons.append("fragment")
        if reasons:
            flagged.append(RecoverySpan(segment.start, segment.end, tuple(reasons)))
    return merge_recovery_spans(flagged, result.duration)


def _crop_words(segments: Sequence[Segment], start: float, end: float) -> tuple[WordTiming, ...]:
    return tuple(
        word
        for word in flatten_words(segments)
        if start <= (word.start + word.end) / 2 <= end
    )


def _context_for_span(result: TranscriptionResult, span: RecoverySpan) -> str:
    before = [
        segment.text for segment in result.segments if segment.end <= span.start
    ][-2:]
    after = [
        segment.text for segment in result.segments if segment.start >= span.end
    ][:2]
    context = "Previous: " + " ".join(before) + "\nFollowing: " + " ".join(after)
    return context[-600:]


def context_copy_rejected(candidate: str, context: str, primary: str) -> bool:
    candidate_tokens = _tokens(candidate)
    context_tokens = set(_tokens(context))
    primary_tokens = set(_tokens(primary))
    copied = [token for token in candidate_tokens if token in context_tokens and token not in primary_tokens]
    return len(copied) >= 4 and len(copied) / max(1, len(candidate_tokens)) > 0.25


def _agreement(text: str, all_texts: Sequence[str]) -> float:
    current = Counter(_tokens(text))
    if not current or len(all_texts) <= 1:
        return 0.5
    values = []
    for other in all_texts:
        if other == text:
            continue
        comparison = Counter(_tokens(other))
        intersection = sum((current & comparison).values())
        union = sum((current | comparison).values())
        values.append(intersection / max(1, union))
    return sum(values) / len(values) if values else 0.5


def candidate_score(
    words: Sequence[WordTiming],
    text: str,
    alternatives: Sequence[str],
    duration: float,
) -> float:
    scores = [word.confidence for word in words if word.confidence is not None]
    acoustic = sum(scores) / len(scores) if scores else 0.0
    coverage = min(1.0, len(words) / max(1.0, duration * 1.3))
    repeat_penalty = min(1.0, repetition_count(text) / max(1, len(_tokens(text))))
    plausibility = max(0.0, 1.0 - 2.5 * repeat_penalty)
    agreement = _agreement(text, alternatives)
    return 0.55 * acoustic + 0.20 * agreement + 0.15 * plausibility + 0.10 * coverage


def _shifted_windows(span: RecoverySpan, duration: float) -> tuple[tuple[float, float], ...]:
    length = span.end - span.start
    windows = []
    for shift in (-1.5, 1.5):
        start = min(max(0.0, span.start + shift), max(0.0, duration - length))
        windows.append((start, min(duration, start + length)))
    return tuple(windows)


def recover_transcript(
    result: TranscriptionResult,
    waveform: np.ndarray,
    *,
    device: str = "auto",
    glossary: Sequence[str] = (),
    allow_downloads: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> TranscriptionResult:
    """Retry suspicious spans and accept changes only with stronger evidence."""
    spans = detect_recovery_spans(result)
    if not spans:
        diagnostics = dict(result.diagnostics)
        diagnostics.update({"enhanced_recovery": True, "recovery_spans": [], "recovery_changes": 0})
        return replace(result, diagnostics=diagnostics)

    qwen_requests: list[tuple[int, int, str]] = []
    qwen_owners: list[int] = []
    for index, span in enumerate(spans):
        context = _context_for_span(result, span)
        vocabulary = ", ".join(glossary)[:300]
        prompt = f"Vocabulary: {vocabulary}\n{context}" if vocabulary else context
        for start, end in _shifted_windows(span, result.duration):
            qwen_requests.append((int(start * 16_000), int(end * 16_000), prompt))
            qwen_owners.append(index)
    _notify(progress_callback, "Re-decoding suspicious spans with shifted Qwen windows...", 0.10)
    retry_errors: dict[str, str] = {}
    try:
        qwen = transcribe_retry_windows(
            "qwen3",
            waveform,
            qwen_requests,
            device=device,
            allow_downloads=allow_downloads,
            progress=lambda message, value: _notify(
                progress_callback, message, 0.10 + 0.32 * max(0.0, min(1.0, value or 0.0))
            ),
        )
    except Exception as exc:
        qwen = ()
        retry_errors["qwen3"] = f"{type(exc).__name__}: {exc}"

    flagged_seconds = sum(span.end - span.start for span in spans)
    full_vaani = flagged_seconds / max(0.001, result.duration) > 0.35
    vaa_spans = (RecoverySpan(0.0, result.duration, ("full_retry",)),) if full_vaani else spans
    vaa_requests = [
        (int(span.start * 16_000), int(span.end * 16_000), "") for span in vaa_spans
    ]
    _notify(progress_callback, "Re-decoding suspicious spans with Vaani Whisper...", 0.44)
    try:
        vaani = transcribe_retry_windows(
            "vaani-whisper",
            waveform,
            vaa_requests,
            device=device,
            allow_downloads=allow_downloads,
            progress=lambda message, value: _notify(
                progress_callback, message, 0.44 + 0.26 * max(0.0, min(1.0, value or 0.0))
            ),
        )
    except Exception as exc:
        vaani = ()
        retry_errors["vaani-whisper"] = f"{type(exc).__name__}: {exc}"

    retry_by_span: list[list[RetryTranscript]] = [[] for _ in spans]
    for owner, retry in zip(qwen_owners, qwen, strict=False):
        retry_by_span[owner].append(retry)
    if full_vaani and vaani:
        for retries, span in zip(retry_by_span, spans, strict=True):
            retries.append(vaani[0])
    else:
        for retries, retry in zip(retry_by_span, vaani, strict=False):
            retries.append(retry)

    unique_retries: list[RetryTranscript] = []
    seen_retry_ids: set[int] = set()
    for retries in retry_by_span:
        for retry in retries:
            if retry.text and id(retry) not in seen_retry_ids:
                seen_retry_ids.add(id(retry))
                unique_retries.append(retry)
    align_inputs = [(retry.start, retry.end, retry.text) for retry in unique_retries]
    _notify(progress_callback, "Scoring retry candidates against the audio...", 0.72)
    aligned_retries = align_hindi_candidates(align_inputs, waveform, device)
    aligned_by_retry_id = {
        id(retry): aligned
        for retry, aligned in zip(unique_retries, aligned_retries, strict=True)
    }
    selected_words = list(flatten_words(result.segments))
    diagnostics_rows = []
    changes = 0
    for span, retries in zip(spans, retry_by_span, strict=True):
        primary_words = _crop_words(result.segments, span.start, span.end)
        primary_text = " ".join(word.text for word in primary_words)
        candidates: list[tuple[str, str, tuple[WordTiming, ...], float]] = []
        candidate_texts = [primary_text] + [retry.text for retry in retries if retry.text]
        candidates.append(
            (
                "primary",
                primary_text,
                primary_words,
                candidate_score(primary_words, primary_text, candidate_texts, span.end - span.start),
            )
        )
        context = _context_for_span(result, span)
        for retry in retries:
            if not retry.text:
                continue
            aligned = aligned_by_retry_id[id(retry)]
            words = _crop_words((aligned,), span.start, span.end)
            text = " ".join(word.text for word in words)
            if not text or context_copy_rejected(text, context, primary_text):
                continue
            score = candidate_score(words, text, candidate_texts, span.end - span.start)
            candidates.append((retry.model, text, words, score))
        candidates.sort(key=lambda item: item[3], reverse=True)
        best = candidates[0]
        primary = next(item for item in candidates if item[0] == "primary")
        primary_repeats = repetition_count(primary[1])
        retry_removes_repeat = [
            item for item in candidates
            if item[0] != "primary" and repetition_count(item[1]) < primary_repeats
        ]
        repeat_evidence = (
            "repetition" in span.reasons
            and primary_repeats > 0
            and len(retry_removes_repeat) >= 2
            and repetition_count(best[1]) < primary_repeats
        )
        cross_model_evidence = (
            "repetition" not in span.reasons
            and any("vaani" in item[0].casefold() for item in candidates if item[0] != "primary")
            and best[3] >= primary[3] + 0.08
        )
        accepted = (
            best[0] != "primary"
            and best[3] >= primary[3] + 0.03
            and (repeat_evidence or cross_model_evidence)
        )
        chosen = best if accepted else primary
        removed_repeat = accepted and repetition_count(chosen[1]) < repetition_count(primary[1])
        alternative_texts = tuple(dict.fromkeys(item[1] for item in candidates if item[1] != chosen[1]))[:3]
        replacement = tuple(
            replace(
                word,
                origin=chosen[0],
                alternatives=alternative_texts,
                uncertain=not accepted,
                repetition_removed=removed_repeat and word_index == 0,
            )
            for word_index, word in enumerate(chosen[2])
        )
        selected_words = [
            word
            for word in selected_words
            if not (span.start <= (word.start + word.end) / 2 <= span.end)
        ]
        selected_words.extend(replacement)
        if accepted:
            changes += 1
        diagnostics_rows.append(
            {
                "start": round(span.start, 3),
                "end": round(span.end, 3),
                "reasons": list(span.reasons),
                "selected_model": chosen[0],
                "accepted_retry": accepted,
                "repetition_removed": removed_repeat,
                "primary_score": round(primary[3], 4),
                "selected_score": round(chosen[3], 4),
                "alternatives": [
                    {"model": model, "text": text, "score": round(score, 4)}
                    for model, text, _, score in candidates
                ],
            }
        )
    selected_words.sort(key=lambda word: (word.start, word.end))
    segments = tuple(
        replace(
            segment,
            uncertain=any(word.uncertain for word in segment.words),
        )
        for segment in words_to_segments(selected_words, max_gap=0.8)
    )
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "enhanced_recovery": True,
            "recovery_spans": diagnostics_rows,
            "recovery_flagged_seconds": round(flagged_seconds, 3),
            "recovery_flagged_percent": round(100 * flagged_seconds / max(0.001, result.duration), 2),
            "recovery_full_vaani_pass": full_vaani,
            "recovery_changes": changes,
            "recovery_retry_errors": retry_errors,
        }
    )
    return replace(
        result,
        segments=segments,
        diagnostics=diagnostics,
        provenance=result.provenance + ("evidence-grounded targeted ASR recovery",),
    )
