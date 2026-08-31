from __future__ import annotations

import base64
import gc
import json
import re
import struct
import zlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .secrets import (
    forget_token,
    load_secret_bytes,
    load_token,
    save_secret_bytes,
    save_token,
)


PROFILE_INDEX_TARGET = "PrivateConversationTranscriber/SpeakerProfiles"
PROFILE_TARGET_PREFIX = "PrivateConversationTranscriber/Speaker/"
_SAFE_NAME = re.compile(r"^[^/\\\x00-\x1f]{1,60}$")
_PROFILE_MAGIC = b"SPK2"
_PROFILE_HEADER = struct.Struct("<4sIf")


def _target(name: str) -> str:
    if not _SAFE_NAME.fullmatch(name.strip()):
        raise ValueError("Speaker name must be 1-60 characters and cannot contain slashes")
    return PROFILE_TARGET_PREFIX + name.strip()


def list_profiles() -> tuple[str, ...]:
    try:
        value = load_token(PROFILE_INDEX_TARGET)
    except OSError:
        return ()
    if not value:
        return ()
    try:
        names = json.loads(value)
    except json.JSONDecodeError:
        return ()
    return tuple(sorted(str(name) for name in names if _SAFE_NAME.fullmatch(str(name))))


def save_profile(name: str, embedding: np.ndarray) -> None:
    normalized = np.asarray(embedding, dtype="<f4").reshape(-1)
    norm = float(np.linalg.norm(normalized))
    if not normalized.size or not np.isfinite(normalized).all() or norm < 1e-8:
        raise ValueError("Speaker embedding is empty or invalid")
    normalized /= norm
    # Windows Credential Manager accepts at most 2,560 bytes. The old Base64
    # JSON was encoded as UTF-16 and exceeded that limit (WinError 1783).
    # Int8 quantization keeps the credential compact with negligible cosine
    # similarity loss for speaker matching.
    max_abs = float(np.max(np.abs(normalized)))
    scale = max(max_abs / 127.0, np.finfo(np.float32).tiny)
    quantized = np.clip(np.rint(normalized / scale), -127, 127).astype(np.int8)
    payload = _PROFILE_HEADER.pack(_PROFILE_MAGIC, int(normalized.size), scale) + quantized.tobytes()
    save_secret_bytes(payload, _target(name))
    names = set(list_profiles())
    names.add(name.strip())
    save_token(json.dumps(sorted(names), ensure_ascii=False), PROFILE_INDEX_TARGET)


def load_profile(name: str) -> np.ndarray:
    raw = load_secret_bytes(_target(name))
    if raw.startswith(_PROFILE_MAGIC):
        if len(raw) < _PROFILE_HEADER.size:
            raise ValueError("Stored speaker profile is damaged")
        magic, dimension, scale = _PROFILE_HEADER.unpack_from(raw)
        quantized = np.frombuffer(raw, dtype=np.int8, offset=_PROFILE_HEADER.size)
        if magic != _PROFILE_MAGIC or quantized.size != dimension or not np.isfinite(scale):
            raise ValueError("Stored speaker profile is damaged")
        embedding = quantized.astype(np.float32) * scale
        norm = float(np.linalg.norm(embedding))
        if norm < 1e-8:
            raise ValueError("Stored speaker profile is damaged")
        return embedding / norm

    # Backward compatibility for any profile written by the original format.
    payload = json.loads(raw.decode("utf-16-le"))
    legacy = zlib.decompress(base64.b64decode(payload["data"]))
    embedding = np.frombuffer(legacy, dtype="<f4").copy()
    if embedding.size != int(payload["dimension"]):
        raise ValueError("Stored speaker profile is damaged")
    return embedding


def delete_profile(name: str) -> None:
    forget_token(_target(name))
    names = set(list_profiles())
    names.discard(name)
    if names:
        save_token(json.dumps(sorted(names), ensure_ascii=False), PROFILE_INDEX_TARGET)
    else:
        forget_token(PROFILE_INDEX_TARGET)


def match_speaker_profiles(
    labels: Sequence[str],
    embeddings: np.ndarray | None,
    *,
    threshold: float = 0.55,
) -> dict[str, tuple[str, float]]:
    if embeddings is None:
        return {}
    profiles = {name: load_profile(name) for name in list_profiles()}
    if not profiles:
        return {}
    cluster_rows = [
        np.asarray(row, dtype=np.float32).reshape(-1) for row in np.asarray(embeddings)
    ]
    usable_profiles = [
        (name, np.asarray(profile, dtype=np.float32).reshape(-1))
        for name, profile in profiles.items()
        if cluster_rows and np.asarray(profile).size == cluster_rows[0].size
    ]
    if not cluster_rows or not usable_profiles:
        return {}
    clusters = np.vstack(cluster_rows)
    clusters /= np.maximum(np.linalg.norm(clusters, axis=1, keepdims=True), 1e-8)
    profile_names = [name for name, _ in usable_profiles]
    profile_matrix = np.vstack([profile for _, profile in usable_profiles])
    profile_matrix /= np.maximum(np.linalg.norm(profile_matrix, axis=1, keepdims=True), 1e-8)
    scores = clusters @ profile_matrix.T

    # A single enrolled person can still anchor one of the two call clusters.
    # Requiring separation from the other cluster makes this safer than simply
    # lowering the absolute cross-channel similarity threshold.
    if len(profile_names) == 1 and len(cluster_rows) >= 2:
        order = np.argsort(scores[:, 0])[::-1]
        best, second = int(order[0]), int(order[1])
        best_score = float(scores[best, 0])
        if best_score >= 0.30 and best_score - float(scores[second, 0]) >= 0.08:
            return {str(labels[best]): (profile_names[0], best_score)}
        return {}

    # Match globally so iteration order cannot claim a profile before a
    # stronger cluster/profile pair is considered.
    matches: dict[str, tuple[str, float]] = {}
    used_clusters: set[int] = set()
    used_profiles: set[int] = set()
    pairs = sorted(
        (
            (float(scores[cluster, profile]), cluster, profile)
            for cluster in range(scores.shape[0])
            for profile in range(scores.shape[1])
        ),
        reverse=True,
    )
    for score, cluster, profile in pairs:
        if score < threshold:
            break
        if cluster in used_clusters or profile in used_profiles:
            continue
        matches[str(labels[cluster])] = (profile_names[profile], score)
        used_clusters.add(cluster)
        used_profiles.add(profile)
    return matches


def enroll_profile(
    name: str,
    audio_path: str | Path,
    hf_token: str,
    device: str = "auto",
) -> int:
    """Extract a one-speaker Community-1 centroid and store no enrollment audio."""
    from .core import Transcriber
    from .diarization import _load_diarization_pipeline

    waveform = Transcriber._decode_audio(Path(audio_path))
    if waveform.size < 10 * 16_000:
        raise ValueError("Speaker enrollment needs at least 10 seconds of clean speech")
    pipeline, torch, _ = _load_diarization_pipeline(hf_token, device, None)
    audio = {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": 16_000}
    output = pipeline(audio, num_speakers=1)
    embeddings = getattr(output, "speaker_embeddings", None)
    if embeddings is None or len(embeddings) != 1:
        raise RuntimeError("The speaker model did not return a usable voice embedding")
    save_profile(name, np.asarray(embeddings[0]))
    dimension = int(np.asarray(embeddings[0]).size)
    del output, pipeline, audio
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dimension
