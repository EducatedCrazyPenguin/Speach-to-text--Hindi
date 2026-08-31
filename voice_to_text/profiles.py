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
    matches: dict[str, tuple[str, float]] = {}
    used: set[str] = set()
    for label, cluster in zip(labels, np.asarray(embeddings), strict=False):
        cluster = np.asarray(cluster, dtype=np.float32).reshape(-1)
        cluster_norm = float(np.linalg.norm(cluster))
        candidates = []
        for name, profile in profiles.items():
            if name in used or profile.size != cluster.size:
                continue
            score = float(np.dot(cluster, profile) / max(1e-8, cluster_norm * np.linalg.norm(profile)))
            candidates.append((score, name))
        if candidates:
            score, name = max(candidates)
            if score >= threshold:
                matches[str(label)] = (name, score)
                used.add(name)
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
