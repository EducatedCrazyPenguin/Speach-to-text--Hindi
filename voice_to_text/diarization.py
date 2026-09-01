from __future__ import annotations

import gc
import math
import os
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .alignment import flatten_words, words_to_segments
from .core import ProgressCallback, Segment, Transcriber, TranscriptionResult, WordTiming, _notify
from .profiles import list_profiles, load_profile, match_speaker_profiles


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


def _choose_pitch_label(
    pitch_hz: float | None,
    references: dict[str, float],
) -> str | None:
    """Choose a speaker only when two call-level pitch centers are distinct."""
    if pitch_hz is None or not math.isfinite(pitch_hz) or pitch_hz <= 0 or len(references) < 2:
        return None
    candidates = sorted(
        (
            (abs(math.log2(pitch_hz / reference)), label)
            for label, reference in references.items()
            if math.isfinite(reference) and reference > 0
        )
    )
    if len(candidates) < 2:
        return None
    best_distance, best_label = candidates[0]
    second_distance, second_label = candidates[1]
    separation = abs(math.log2(references[best_label] / references[second_label]))
    if separation < 0.25 or best_distance > 0.22 or second_distance - best_distance < 0.15:
        return None
    return best_label


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


def assign_word_speakers(
    segments: Sequence[Segment],
    exclusive_turns: Iterable[SpeakerTurn],
    raw_turns: Iterable[SpeakerTurn] = (),
    speaker_names: Sequence[str] = ("Speaker 1", "Speaker 2"),
    label_overrides: dict[str, str] | None = None,
) -> tuple[Segment, ...]:
    """Assign speakers after ASR so recognition keeps full conversational context."""
    turns = tuple(exclusive_turns)
    raw = tuple(raw_turns)
    if not turns:
        return tuple(segments)
    first_seen: dict[str, float] = {}
    for turn in turns:
        first_seen[turn.speaker] = min(first_seen.get(turn.speaker, turn.start), turn.start)
    labels = {
        speaker: speaker_names[index] if index < len(speaker_names) else f"Speaker {index + 1}"
        for index, speaker in enumerate(sorted(first_seen, key=first_seen.get))
    }
    labels.update(label_overrides or {})

    assigned: list[WordTiming] = []
    for word in flatten_words(segments):
        scores: dict[str, float] = defaultdict(float)
        for turn in turns:
            scores[turn.speaker] += max(0.0, min(word.end, turn.end) - max(word.start, turn.start))
        if scores and max(scores.values()) > 0:
            cluster = max(scores, key=scores.get)
        else:
            midpoint = (word.start + word.end) / 2
            cluster = min(
                turns, key=lambda turn: abs(midpoint - (turn.start + turn.end) / 2)
            ).speaker
        raw_scores: dict[str, float] = defaultdict(float)
        for turn in raw:
            raw_scores[turn.speaker] += max(0.0, min(word.end, turn.end) - max(word.start, turn.start))
        overlap_duration = sum(sorted(raw_scores.values(), reverse=True)[1:]) if len(raw_scores) > 1 else 0.0
        is_overlap = overlap_duration >= 0.5 * max(0.001, word.end - word.start)
        assigned.append(replace(word, speaker=labels.get(cluster, cluster), overlap=is_overlap))

    # Suppress isolated label flips shorter than 300 ms between the same speaker.
    for index in range(1, len(assigned) - 1):
        word = assigned[index]
        if (
            word.end - word.start < 0.30
            and assigned[index - 1].speaker == assigned[index + 1].speaker
            and word.speaker != assigned[index - 1].speaker
        ):
            assigned[index] = replace(word, speaker=assigned[index - 1].speaker)
    return words_to_segments(assigned, max_gap=0.35)


def _friendly_cluster_labels(
    turns: Sequence[SpeakerTurn],
    speaker_names: Sequence[str],
    label_overrides: dict[str, str] | None = None,
    profile_embeddings: dict[str, np.ndarray] | None = None,
) -> dict[str, str]:
    first_seen: dict[str, float] = {}
    for turn in turns:
        first_seen[turn.speaker] = min(first_seen.get(turn.speaker, turn.start), turn.start)
    labels = {
        speaker: speaker_names[index] if index < len(speaker_names) else f"Speaker {index + 1}"
        for index, speaker in enumerate(sorted(first_seen, key=first_seen.get))
    }
    labels.update(label_overrides or {})
    return labels


def refine_speaker_words_with_embeddings(
    segments: Sequence[Segment],
    waveform,
    pipeline,
    cluster_names: Sequence[str],
    cluster_embeddings,
    turns: Sequence[SpeakerTurn],
    speaker_names: Sequence[str],
    label_overrides: dict[str, str] | None = None,
    profile_embeddings: dict[str, np.ndarray] | None = None,
    *,
    pause_seconds: float = 0.50,
    maximum_group_seconds: float = 6.0,
) -> tuple[tuple[Segment, ...], dict[str, object]]:
    """Recheck short utterances against call-level speaker centroids.

    Frame-level diarization can move a boundary by a second on telephone audio.
    Forced-aligned words expose real pauses, so we embed each pause-delimited
    utterance and compare it with the two centroids produced for the whole call.
    """
    import torch

    words = list(flatten_words(segments))
    centroids = np.asarray(cluster_embeddings) if cluster_embeddings is not None else np.empty((0, 0))
    if not words or len(cluster_names) != len(centroids) or len(centroids) < 2:
        return tuple(segments), {"speaker_embedding_groups": 0, "speaker_embedding_overrides": 0}
    friendly = _friendly_cluster_labels(turns, speaker_names, label_overrides)
    named_centroids = [friendly.get(str(name), str(name)) for name in cluster_names]
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(centroid_norms, 1e-8)
    profile_names: list[str] = []
    profile_vectors: list[np.ndarray] = []
    for name, profile in (profile_embeddings or {}).items():
        vector = np.asarray(profile, dtype=np.float32).reshape(-1)
        if vector.size != centroids.shape[1]:
            continue
        vector /= max(float(np.linalg.norm(vector)), 1e-8)
        profile_names.append(name)
        profile_vectors.append(vector)
    profiles = np.vstack(profile_vectors) if profile_vectors else np.empty((0, centroids.shape[1]))
    single_profile_anchor: int | None = None
    if len(profiles) == 1 and len(centroids) >= 2:
        anchor_scores = centroids @ profiles[0]
        anchor_order = np.argsort(anchor_scores)[::-1]
        anchor_best, anchor_second = int(anchor_order[0]), int(anchor_order[1])
        if (
            float(anchor_scores[anchor_best]) >= 0.30
            and float(anchor_scores[anchor_best] - anchor_scores[anchor_second]) >= 0.08
        ):
            single_profile_anchor = anchor_best

    groups: list[list[WordTiming]] = []
    for word in words:
        if (
            not groups
            or word.speaker != groups[-1][-1].speaker
            or word.start - groups[-1][-1].end >= pause_seconds
            or word.end - groups[-1][0].start > maximum_group_seconds
        ):
            groups.append([word])
        else:
            groups[-1].append(word)

    # Batch variable-duration crops with a matching sample mask so zero padding
    # does not alter the embedding.
    crops = []
    lengths = []
    sample_rate = 16_000
    for group in groups:
        left = max(0, int((group[0].start - 0.04) * sample_rate))
        right = min(len(waveform), int((group[-1].end + 0.04) * sample_rate))
        crop = np.asarray(waveform[left:right], dtype=np.float32)
        valid_length = crop.size
        if crop.size < pipeline._embedding.min_num_samples:
            crop = np.pad(crop, (0, pipeline._embedding.min_num_samples - crop.size))
        crops.append(crop)
        lengths.append(valid_length)

    vectors = []
    batch_size = min(16, int(getattr(pipeline, "embedding_batch_size", 16)))
    for start in range(0, len(crops), batch_size):
        batch = crops[start : start + batch_size]
        longest = max(len(crop) for crop in batch)
        waveforms = np.zeros((len(batch), 1, longest), dtype=np.float32)
        masks = np.zeros((len(batch), longest), dtype=np.float32)
        for index, crop in enumerate(batch):
            waveforms[index, 0, : len(crop)] = crop
            masks[index, : lengths[start + index]] = 1.0
        vectors.append(
            pipeline._embedding(
                torch.from_numpy(waveforms), masks=torch.from_numpy(masks)
            )
        )
    utterance_embeddings = np.vstack(vectors)

    # Speaker embeddings are unreliable for sub-second speech. Pitch remains
    # measurable in many short greetings, so learn reference centers from
    # longer turns and use them only when the two voices are well separated.
    group_pitches: list[float | None] = [None] * len(groups)
    try:
        import librosa

        for index, (crop, valid_length) in enumerate(zip(crops, lengths, strict=True)):
            if valid_length < int(0.25 * sample_rate):
                continue
            f0, voiced, _ = librosa.pyin(
                crop[:valid_length], fmin=65, fmax=350, sr=sample_rate
            )
            values = f0[np.asarray(voiced, dtype=bool) & np.isfinite(f0)]
            if values.size >= 3:
                group_pitches[index] = float(np.median(values))
    except (ImportError, ValueError):
        pass
    pitch_samples: dict[str, list[float]] = {}
    for group, pitch in zip(groups, group_pitches, strict=True):
        if pitch is not None and group[-1].end - group[0].start >= 1.5:
            pitch_samples.setdefault(str(group[0].speaker), []).append(pitch)
    pitch_references = {
        label: float(np.median(values))
        for label, values in pitch_samples.items()
        if values
    }

    refined: list[WordTiming] = []
    overrides = 0
    confident_groups = 0
    margins: list[float] = []
    decisions: list[dict[str, object]] = []
    for group, embedding, group_pitch in zip(
        groups, utterance_embeddings, group_pitches, strict=True
    ):
        embedding = np.asarray(embedding, dtype=np.float32)
        embedding /= max(float(np.linalg.norm(embedding)), 1e-8)
        similarities = centroids @ embedding
        order = np.argsort(similarities)[::-1]
        best, second = int(order[0]), int(order[1])
        margin = float(similarities[best] - similarities[second])
        duration = group[-1].end - group[0].start
        required_margin = 0.10 if duration < 0.70 else 0.055
        confident = float(similarities[best]) >= 0.25 and margin >= required_margin
        selected_name = named_centroids[best] if confident else group[0].speaker
        decision_source = "call centroid" if confident else "timeline"
        best_profile_score: float | None = None
        if len(profiles) == 1:
            best_profile_score = float(profiles[0] @ embedding)
            if (
                single_profile_anchor is not None
                and best == single_profile_anchor
                and best_profile_score >= 0.42
            ):
                confident = True
                selected_name = profile_names[0]
                decision_source = "enrolled profile"
        if len(profiles) >= 2:
            profile_scores = profiles @ embedding
            profile_order = np.argsort(profile_scores)[::-1]
            profile_best, profile_second = int(profile_order[0]), int(profile_order[1])
            profile_margin = float(profile_scores[profile_best] - profile_scores[profile_second])
            if float(profile_scores[profile_best]) >= 0.50 and profile_margin >= 0.08:
                confident = True
                selected_name = profile_names[profile_best]
                decision_source = "enrolled profile"
        pitch_label = None
        if not confident and duration <= 1.2:
            pitch_label = _choose_pitch_label(group_pitch, pitch_references)
            if pitch_label is not None:
                confident = True
                selected_name = pitch_label
                decision_source = "call pitch"
        if confident:
            confident_groups += 1
            margins.append(margin)
        changed = sum(1 for word in group if confident and word.speaker != selected_name)
        overrides += changed
        refined.extend(
            replace(word, speaker=selected_name) if confident else word for word in group
        )
        decisions.append(
            {
                "start": round(group[0].start, 3),
                "end": round(group[-1].end, 3),
                "selected": selected_name,
                "best_candidate": named_centroids[best],
                "previous": group[0].speaker,
                "similarity": round(float(similarities[best]), 4),
                "margin": round(margin, 4),
                "profile_similarity": (
                    round(best_profile_score, 4) if best_profile_score is not None else None
                ),
                "pitch_hz": round(group_pitch, 1) if group_pitch is not None else None,
                "pitch_candidate": pitch_label,
                "confident": confident,
                "source": decision_source,
            }
        )
    return words_to_segments(refined, max_gap=0.35), {
        "speaker_embedding_groups": len(groups),
        "speaker_embedding_confident_groups": confident_groups,
        "speaker_embedding_overrides": overrides,
        "speaker_embedding_mean_margin": round(sum(margins) / len(margins), 4) if margins else None,
        "single_profile_anchor": (
            named_centroids[single_profile_anchor] if single_profile_anchor is not None else None
        ),
        "speaker_pitch_references_hz": {
            label: round(value, 1) for label, value in pitch_references.items()
        },
        "speaker_embedding_decisions": decisions,
    }


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


def smooth_speaker_turns(
    turns: Iterable[SpeakerTurn],
    short_flip_seconds: float = 0.30,
    matching_gap_seconds: float = 0.35,
) -> tuple[SpeakerTurn, ...]:
    """Remove very short A-B-A flips and join nearby matching turns."""
    ordered = [turn for turn in sorted(turns, key=lambda item: (item.start, item.end)) if turn.end > turn.start]
    changed = True
    while changed and len(ordered) >= 3:
        changed = False
        for index in range(1, len(ordered) - 1):
            previous, current, following = ordered[index - 1 : index + 2]
            if (
                current.end - current.start < short_flip_seconds
                and previous.speaker == following.speaker
            ):
                ordered[index - 1 : index + 2] = [
                    SpeakerTurn(previous.start, following.end, previous.speaker)
                ]
                changed = True
                break

    merged: list[SpeakerTurn] = []
    for turn in ordered:
        if (
            merged
            and merged[-1].speaker == turn.speaker
            and turn.start - merged[-1].end <= matching_gap_seconds
        ):
            previous = merged[-1]
            merged[-1] = SpeakerTurn(previous.start, max(previous.end, turn.end), previous.speaker)
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


def _extract_raw_turns(output) -> tuple[SpeakerTurn, ...]:
    annotation = getattr(output, "speaker_diarization", None)
    if annotation is None:
        return _extract_turns(output)
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
    turns = smooth_speaker_turns(_extract_turns(output))
    raw_turns = _extract_raw_turns(output)
    annotation = getattr(output, "speaker_diarization", None)
    labels = list(annotation.labels()) if annotation is not None and hasattr(annotation, "labels") else []
    profile_matches = match_speaker_profiles(labels, getattr(output, "speaker_embeddings", None))
    label_overrides = {label: name for label, (name, _score) in profile_matches.items()}
    enrolled_profiles = {}
    enrolled_canonical_names: set[str] = set()
    for profile_name in list_profiles():
        canonical_name = profile_name.casefold()
        if canonical_name in enrolled_canonical_names:
            continue
        try:
            enrolled_profiles[profile_name] = load_profile(profile_name)
            enrolled_canonical_names.add(canonical_name)
        except (OSError, ValueError, KeyError):
            continue
    _notify(progress_callback, "Assigning words to speakers...", 0.98)
    labelled = assign_word_speakers(
        result.segments,
        turns,
        raw_turns,
        speaker_names,
        label_overrides,
    )
    labelled, embedding_diagnostics = refine_speaker_words_with_embeddings(
        labelled,
        waveform,
        pipeline,
        labels,
        getattr(output, "speaker_embeddings", None),
        turns,
        speaker_names,
        label_overrides,
        enrolled_profiles,
    )
    _notify(progress_callback, "Speaker identification complete", 1.0)
    diagnostics = dict(result.diagnostics)
    diagnostics["speaker_profile_matches"] = {
        label: {"name": name, "cosine_similarity": score}
        for label, (name, score) in profile_matches.items()
    }
    diagnostics.update(embedding_diagnostics)
    diagnostics["overlap_seconds"] = sum(
        max(0.0, min(a.end, b.end) - max(a.start, b.start))
        for index, a in enumerate(raw_turns)
        for b in raw_turns[index + 1 :]
        if a.speaker != b.speaker
    )
    turn_boundaries = tuple(
        boundary
        for turn in turns
        for boundary in (turn.start, turn.end)
        if 0.0 < boundary < result.duration
    )
    diagnostics["speaker_boundary_ambiguous_words"] = sum(
        1
        for word in flatten_words(result.segments)
        if any(abs((word.start + word.end) / 2 - boundary) <= 0.25 for boundary in turn_boundaries)
    )
    diagnostics["speaker_turn_smoothing"] = {
        "short_flip_seconds": 0.30,
        "matching_gap_seconds": 0.35,
    }
    del output, pipeline, audio
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return replace(result, segments=labelled, diagnostics=diagnostics)


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
