from __future__ import annotations

import os
import gc
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

from .core import ProgressCallback, Segment, Transcriber, TranscriptionResult, _notify


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


def _overlap(segment: Segment, turn: SpeakerTurn) -> float:
    return max(0.0, min(segment.end, turn.end) - max(segment.start, turn.start))


def assign_speakers(
    segments: Sequence[Segment],
    turns: Iterable[SpeakerTurn],
    speaker_names: Sequence[str] = ("Speaker 1", "Speaker 2"),
) -> tuple[Segment, ...]:
    """Assign each transcript segment to the voice with the most temporal overlap."""
    turn_list = list(turns)
    if not turn_list:
        return tuple(segments)

    first_seen: dict[str, float] = {}
    for turn in turn_list:
        first_seen[turn.speaker] = min(first_seen.get(turn.speaker, turn.start), turn.start)
    ordered_speakers = sorted(first_seen, key=first_seen.get)
    labels = {
        speaker: speaker_names[index] if index < len(speaker_names) else f"Speaker {index + 1}"
        for index, speaker in enumerate(ordered_speakers)
    }

    assigned = []
    for segment in segments:
        scores: dict[str, float] = defaultdict(float)
        for turn in turn_list:
            scores[turn.speaker] += _overlap(segment, turn)
        if scores and max(scores.values()) > 0:
            speaker = max(scores, key=scores.get)
        else:
            midpoint = (segment.start + segment.end) / 2
            nearest = min(
                turn_list,
                key=lambda turn: abs(midpoint - ((turn.start + turn.end) / 2)),
            )
            speaker = nearest.speaker
        assigned.append(replace(segment, speaker=labels[speaker]))
    return tuple(assigned)


def merge_speaker_turns(
    turns: Iterable[SpeakerTurn],
    max_duration: float = 15.0,
    max_gap: float = 0.45,
    minimum_duration: float = 0.20,
) -> tuple[SpeakerTurn, ...]:
    """Merge nearby exclusive turns from one voice into clean ASR regions."""
    merged: list[SpeakerTurn] = []
    for turn in sorted(turns, key=lambda item: (item.start, item.end)):
        if turn.end - turn.start < minimum_duration:
            continue
        if (
            merged
            and merged[-1].speaker == turn.speaker
            and turn.start - merged[-1].end <= max_gap
            and turn.end - merged[-1].start <= max_duration
        ):
            previous = merged[-1]
            merged[-1] = SpeakerTurn(previous.start, turn.end, previous.speaker)
        else:
            merged.append(turn)
    return tuple(merged)


def _friendly_speaker_names(
    turns: Iterable[SpeakerTurn], speaker_names: Sequence[str]
) -> tuple[SpeakerTurn, ...]:
    turn_list = tuple(turns)
    first_seen: dict[str, float] = {}
    for turn in turn_list:
        first_seen[turn.speaker] = min(first_seen.get(turn.speaker, turn.start), turn.start)
    labels = {
        speaker: speaker_names[index] if index < len(speaker_names) else f"Speaker {index + 1}"
        for index, speaker in enumerate(sorted(first_seen, key=first_seen.get))
    }
    return tuple(replace(turn, speaker=labels[turn.speaker]) for turn in turn_list)


def _load_diarization_pipeline(hf_token: str, device: str, progress_callback):
    """Load cached/downloaded Community-1 and select the requested device."""
    cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    try:
        import torch
        from huggingface_hub import snapshot_download
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Speaker identification is not installed. Run setup-diarization.ps1 first."
        ) from exc

    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
    _notify(
        progress_callback,
        "Loading the two-speaker model (the first run may download it)...",
        0.02,
    )
    download_options = (
        {"token": hf_token.strip()} if hf_token.strip() else {"local_files_only": True}
    )
    try:
        model_directory = snapshot_download(
            "pyannote/speaker-diarization-community-1", **download_options
        )
    except Exception as exc:
        if not hf_token.strip():
            raise RuntimeError(
                "The speaker model is not cached yet. Paste a Hugging Face read token once to download it."
            ) from exc
        raise
    pipeline = Pipeline.from_pretrained(model_directory)
    selected_device = "cpu"
    if device != "cpu" and torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        selected_device = "cuda"
    return pipeline, torch, selected_device


def _extract_turns(output) -> tuple[SpeakerTurn, ...]:
    """Prefer Community-1's non-overlapping timeline made for transcription."""
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = output.speaker_diarization
    return tuple(
        SpeakerTurn(float(turn.start), float(turn.end), str(speaker))
        for turn, speaker in annotation
    )


def diarize_then_transcribe_two_speakers(
    transcriber: Transcriber,
    source: str | Path,
    language: str | None,
    hf_token: str,
    speaker_names: Sequence[str] = ("Speaker 1", "Speaker 2"),
    device: str = "auto",
    progress_callback: ProgressCallback | None = None,
) -> TranscriptionResult:
    """Diarize first, then ASR each exclusive speaker turn independently."""
    source_path = Path(source).expanduser().resolve()
    waveform = Transcriber._decode_audio(source_path)
    pipeline, torch, selected_device = _load_diarization_pipeline(
        hf_token, device, progress_callback
    )
    _notify(
        progress_callback,
        f"Separating two voices on {selected_device}; long calls can take a few minutes...",
        0.08,
    )
    audio = {
        "waveform": torch.from_numpy(waveform).unsqueeze(0),
        "sample_rate": 16_000,
    }
    output = pipeline(audio, num_speakers=2)
    turns = _friendly_speaker_names(
        merge_speaker_turns(_extract_turns(output)), speaker_names
    )

    # The ASR model needs most of the GPU memory, so release pyannote first.
    del output, pipeline, audio
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load ASR before publishing the 55% milestone so its own first-load message
    # cannot make the web progress bar jump backwards to 2%.
    transcriber._load_indicconformer(language)
    _notify(progress_callback, f"Transcribing {len(turns)} clean speaker turns...", 0.55)
    regions = tuple(Segment(turn.start, turn.end, "", turn.speaker) for turn in turns)
    return transcriber.transcribe_indicconformer_regions(
        source_path, language, regions, waveform=waveform
    )


def diarize_two_speakers(
    result: TranscriptionResult,
    hf_token: str,
    speaker_names: Sequence[str] = ("Speaker 1", "Speaker 2"),
    device: str = "auto",
    progress_callback: ProgressCallback | None = None,
) -> TranscriptionResult:
    """Run the local pyannote community model and label exactly two voices."""
    pipeline, torch, selected_device = _load_diarization_pipeline(
        hf_token, device, progress_callback
    )

    _notify(progress_callback, f"Separating two voices on {selected_device}...", 0.1)
    waveform = Transcriber._decode_audio(result.source)
    audio = {
        "waveform": torch.from_numpy(waveform).unsqueeze(0),
        "sample_rate": 16_000,
    }
    output = pipeline(audio, num_speakers=2)
    turns = _extract_turns(output)
    _notify(progress_callback, "Assigning words to speakers...", 0.98)
    labelled = assign_speakers(result.segments, turns, speaker_names)
    _notify(progress_callback, "Speaker identification complete", 1.0)
    return replace(result, segments=labelled)


def speaker_model_is_cached() -> bool:
    """Return whether Community-1 can be loaded without another token/network request."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            "pyannote/speaker-diarization-community-1",
            local_files_only=True,
        )
        return True
    except Exception:
        return False
