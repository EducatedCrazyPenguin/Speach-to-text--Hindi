from __future__ import annotations

from dataclasses import replace
import gc
import re
from typing import Sequence

import numpy as np

from .alignment import proportional_word_timings
from .core import ProgressCallback, Segment, TranscriptionResult, WordTiming, _notify


_NON_MMS = re.compile(r"[^a-z' ]")
_SPACES = re.compile(r" +")


def normalize_for_mms(text: str, romanizer) -> str:
    """Romanize one word while preserving its one-to-one source-word mapping."""
    romanized = romanizer.romanize_string(text, lcode="hin")
    romanized = str(romanized).lower().replace("’", "'")
    romanized = _SPACES.sub(" ", _NON_MMS.sub(" ", romanized)).strip()
    # A source token must remain one alignment unit. Uroman occasionally emits
    # spaces for compounds; removing those is safer than changing word count.
    return romanized.replace(" ", "") or "*"


def _span_score(spans: Sequence[object]) -> float | None:
    length = sum(len(span) for span in spans)
    if not length:
        return None
    return float(sum(float(span.score) * len(span) for span in spans) / length)


def _align_segment(
    segment: Segment,
    waveform: np.ndarray,
    model,
    tokenizer,
    aligner,
    romanizer,
    torch,
    device,
    *,
    sample_rate: int = 16_000,
    context_seconds: float = 0.45,
) -> tuple[tuple[WordTiming, ...], int, list[float]]:
    source_words = tuple(
        segment.words
        or proportional_word_timings(segment.text, segment.start, segment.end, segment.confidence)
    )
    if not source_words:
        return (), 0, []

    left = max(0.0, min(segment.start, source_words[0].start) - context_seconds)
    right = min(
        waveform.size / sample_rate,
        max(segment.end, source_words[-1].end) + context_seconds,
    )
    start_sample = int(left * sample_rate)
    end_sample = int(right * sample_rate)
    audio = torch.from_numpy(
        np.asarray(waveform[start_sample:end_sample], dtype=np.float32)
    ).unsqueeze(0)
    normalized = [normalize_for_mms(word.text, romanizer) for word in source_words]
    # Star tokens absorb speech before/after this chunk's recognized words. This
    # matters because adjacent ASR chunks deliberately overlap by 1.5 seconds.
    transcript = ["*", *normalized, "*"]
    with torch.inference_mode():
        emission, _ = model(audio.to(device))
        spans = aligner(emission[0], tokenizer(transcript))
    if len(spans) != len(transcript):
        raise RuntimeError("MMS returned a different number of aligned words")

    seconds_per_frame = audio.size(1) / emission.size(1) / sample_rate
    aligned: list[WordTiming] = []
    scores: list[float] = []
    aligned_count = 0
    for source_word, word_spans, normalized_word in zip(
        source_words, spans[1:-1], normalized, strict=True
    ):
        if not word_spans:
            aligned.append(source_word)
            continue
        start = left + float(word_spans[0].start) * seconds_per_frame
        end = left + float(word_spans[-1].end) * seconds_per_frame
        score = _span_score(word_spans)
        if end <= start or start < left or end > right + 0.05:
            aligned.append(source_word)
            continue
        aligned.append(replace(source_word, start=start, end=end, confidence=score))
        if normalized_word != "*":
            aligned_count += 1
            if score is not None:
                scores.append(score)
    del emission, audio
    return tuple(aligned), aligned_count, scores


def force_align_hindi_words(
    result: TranscriptionResult,
    waveform: np.ndarray,
    device: str = "auto",
    progress_callback: ProgressCallback | None = None,
) -> TranscriptionResult:
    """Replace approximate ASR word times with multilingual acoustic alignment."""
    try:
        import torch
        import uroman as ur
        from torchaudio.pipelines import MMS_FA
    except ImportError as exc:
        raise RuntimeError(
            "Hindi forced alignment is not installed. Run setup-accuracy.ps1 first."
        ) from exc

    selected = torch.device(
        "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
    )
    _notify(progress_callback, f"Loading Hindi word aligner on {selected.type}...", 0.02)
    model = MMS_FA.get_model().to(selected).eval()
    tokenizer = MMS_FA.get_tokenizer()
    aligner = MMS_FA.get_aligner()
    romanizer = ur.Uroman()

    aligned_segments: list[Segment] = []
    aligned_count = 0
    total_count = 0
    scores: list[float] = []
    failures = 0
    for index, segment in enumerate(result.segments, 1):
        source_words = tuple(
            segment.words
            or proportional_word_timings(segment.text, segment.start, segment.end, segment.confidence)
        )
        total_count += len(source_words)
        try:
            words, count, word_scores = _align_segment(
                segment,
                waveform,
                model,
                tokenizer,
                aligner,
                romanizer,
                torch,
                selected,
            )
            aligned_count += count
            scores.extend(word_scores)
            if words:
                aligned_segments.append(
                    replace(
                        segment,
                        start=words[0].start,
                        end=words[-1].end,
                        words=words,
                    )
                )
            else:
                aligned_segments.append(segment)
        except (RuntimeError, ValueError, KeyError):
            # A hallucinated/very long chunk can be impossible to align. Keep its
            # validated fallback timings instead of losing the transcription.
            failures += 1
            aligned_segments.append(segment)
        _notify(
            progress_callback,
            f"Aligning Hindi words {index}/{len(result.segments)}",
            index / max(1, len(result.segments)),
        )

    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "word_alignment_method": "torchaudio MMS_FA + uroman",
            "aligned_words": aligned_count,
            "alignment_total_words": total_count,
            "aligned_word_percent": round(100.0 * aligned_count / max(1, total_count), 2),
            "alignment_mean_score": round(sum(scores) / len(scores), 4) if scores else None,
            "alignment_failed_segments": failures,
            "alignment_device": selected.type,
        }
    )
    del model, tokenizer, aligner, romanizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return replace(
        result,
        segments=tuple(aligned_segments),
        diagnostics=diagnostics,
        provenance=result.provenance
        + ("Hindi word timestamps forced-aligned with TorchAudio MMS_FA",),
    )


def score_hindi_candidates(
    candidates: Sequence[tuple[float, float, str]],
    waveform: np.ndarray,
    device: str = "auto",
) -> tuple[float, ...]:
    """Return comparable MMS acoustic scores for candidate retry transcripts."""
    if not candidates:
        return ()
    try:
        import torch
        import uroman as ur
        from torchaudio.pipelines import MMS_FA
    except ImportError:
        return tuple(0.0 for _ in candidates)
    selected = torch.device("cuda" if device != "cpu" and torch.cuda.is_available() else "cpu")
    model = MMS_FA.get_model().to(selected).eval()
    tokenizer = MMS_FA.get_tokenizer()
    aligner = MMS_FA.get_aligner()
    romanizer = ur.Uroman()
    scores: list[float] = []
    for start, end, text in candidates:
        segment = Segment(
            start,
            end,
            text,
            words=proportional_word_timings(text, start, end),
        )
        try:
            _, count, values = _align_segment(
                segment,
                waveform,
                model,
                tokenizer,
                aligner,
                romanizer,
                torch,
                selected,
                context_seconds=0.15,
            )
            coverage = count / max(1, len(segment.words))
            scores.append((sum(values) / len(values) if values else 0.0) * coverage)
        except (RuntimeError, ValueError, KeyError):
            scores.append(0.0)
    del model, tokenizer, aligner, romanizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tuple(scores)


def align_hindi_candidates(
    candidates: Sequence[tuple[float, float, str]],
    waveform: np.ndarray,
    device: str = "auto",
) -> tuple[Segment, ...]:
    """Force-align several retry candidates with one aligner model load."""
    if not candidates:
        return ()
    try:
        import torch
        import uroman as ur
        from torchaudio.pipelines import MMS_FA
    except ImportError:
        return tuple(
            Segment(start, end, text, words=proportional_word_timings(text, start, end))
            for start, end, text in candidates
        )
    selected = torch.device("cuda" if device != "cpu" and torch.cuda.is_available() else "cpu")
    model = MMS_FA.get_model().to(selected).eval()
    tokenizer = MMS_FA.get_tokenizer()
    aligner = MMS_FA.get_aligner()
    romanizer = ur.Uroman()
    output: list[Segment] = []
    for start, end, text in candidates:
        original = Segment(
            start,
            end,
            text,
            words=proportional_word_timings(text, start, end),
        )
        try:
            words, _, _ = _align_segment(
                original,
                waveform,
                model,
                tokenizer,
                aligner,
                romanizer,
                torch,
                selected,
                context_seconds=0.15,
            )
            output.append(
                replace(
                    original,
                    start=words[0].start if words else start,
                    end=words[-1].end if words else end,
                    words=words,
                )
            )
        except (RuntimeError, ValueError, KeyError):
            output.append(original)
    del model, tokenizer, aligner, romanizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tuple(output)
