"""Local conversation transcription tools."""

import os

# hf-xet can hang indefinitely on authenticated Windows downloads. The stable
# HTTP downloader is fast enough for these comparatively small speech models.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from .core import ENGINE_CHOICES, LANGUAGES, MODEL_CHOICES, Segment, TranscriptionResult, Transcriber

__all__ = [
    "ENGINE_CHOICES",
    "LANGUAGES",
    "MODEL_CHOICES",
    "Segment",
    "TranscriptionResult",
    "Transcriber",
]

