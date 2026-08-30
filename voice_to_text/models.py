from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .alignment import (
    deduplicate_overlapping_segments,
    proportional_word_timings,
    speech_weighted_word_timings,
)
from .core import (
    ProgressCallback,
    Segment,
    Transcriber,
    TranscriptionResult,
    WordTiming,
    _notify,
    _register_torch_cuda_dlls,
)


CANDIDATE_LABELS = {
    "sravaani": "SraVaani 1.0 TDT (recommended)",
    "qwen3": "Qwen3-ASR 1.7B",
    "srota": "Srota Qwen3-ASR Hindi/Hinglish 0.6B (evaluation)",
    "ai4bharat600": "AI4Bharat IndicConformer 600M RNNT",
    "vaani-whisper": "Vaani Whisper Large-v3 Hindi",
    "orato": "Orato Hindi beta (evaluation only)",
    "legacy": "Legacy AI4Bharat 120M greedy CTC",
    "whisper-large": "Whisper Large-v3",
}

MODEL_IDS = {
    "qwen3": "Qwen/Qwen3-ASR-1.7B",
    "srota": "moorlee/qwen3-asr-0.6b-hinglish",
    "ai4bharat600": "ai4bharat/indic-conformer-600m-multilingual",
    "vaani-whisper": "ARTPARK-IISc/whisper-large-v3-vaani-hindi",
    "orato": "tryorato/orato-asr-hindi-v1",
}


def _cached_snapshot_or_model_id(model_id: str) -> tuple[str, bool]:
    """Prefer a complete local snapshot so routine inference never calls the Hub."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_id, local_files_only=True), True
    except Exception:
        return model_id, False


def optional_model_availability() -> dict[str, bool]:
    transformers = bool(importlib.util.find_spec("transformers"))
    qwen = bool(importlib.util.find_spec("qwen_asr"))
    orato_cached = False
    if qwen:
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(MODEL_IDS["orato"], local_files_only=True)
            orato_cached = True
        except Exception:
            pass
    return {
        "sravaani": True,
        "legacy": True,
        "whisper-large": True,
        "qwen3": qwen,
        "srota": qwen,
        "orato": qwen and orato_cached,
        "ai4bharat600": transformers,
        "vaani-whisper": transformers,
    }


def _providers(device: str):
    if device == "cpu":
        return ["CPUExecutionProvider"]
    try:
        _register_torch_cuda_dlls()
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except ImportError:
        pass
    if device == "cuda":
        raise RuntimeError("ONNX CUDA is unavailable; run setup-gpu.ps1")
    return ["CPUExecutionProvider"]


def vad_regions(waveform: np.ndarray, device: str = "auto") -> tuple[tuple[int, int], ...]:
    """Return ~25 second speech chunks with 1.5 seconds of context overlap."""
    try:
        import onnx_asr

        vad = onnx_asr.load_vad("silero", providers=_providers(device))
        batches = vad.segment_batch(
            waveform.reshape(1, -1).astype(np.float32),
            np.asarray([waveform.size], dtype=np.int64),
            16_000,
            min_speech_duration_ms=200,
            min_silence_duration_ms=500,
            max_speech_duration_s=25,
            speech_pad_ms=750,
        )
        regions = tuple(next(batches))
        if regions:
            return regions
    except Exception:
        pass
    chunk = 25 * 16_000
    stride = int(23.5 * 16_000)
    return tuple(
        (start, min(waveform.size, start + chunk))
        for start in range(0, waveform.size, stride)
        if min(waveform.size, start + chunk) - start >= 1600
    )


def _segments_from_text_chunks(
    chunks: Iterable[tuple[int, int, str, float | None]],
    waveform: np.ndarray | None = None,
) -> tuple[Segment, ...]:
    segments = []
    for start_sample, end_sample, text, confidence in chunks:
        text = text.strip()
        if not text:
            continue
        start, end = start_sample / 16_000, end_sample / 16_000
        words = (
            speech_weighted_word_timings(text, waveform[start_sample:end_sample], start, confidence)
            if waveform is not None
            else proportional_word_timings(text, start, end, confidence)
        )
        segments.append(
            Segment(
                start,
                end,
                text,
                confidence=confidence,
                words=words,
            )
        )
    return deduplicate_overlapping_segments(segments)


def _transcribe_qwen(
    candidate: str,
    source: Path,
    waveform: np.ndarray,
    device: str,
    prompt: str,
    progress: ProgressCallback | None,
) -> TranscriptionResult:
    if not importlib.util.find_spec("qwen_asr"):
        raise RuntimeError("Qwen3-ASR is not installed. Run setup-accuracy.ps1 first.")
    import torch
    from qwen_asr import Qwen3ASRModel

    model_id = MODEL_IDS[candidate]
    model_path, loaded_offline = _cached_snapshot_or_model_id(model_id)
    selected = "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
    # Segment the complete waveform before allocating almost all 16 GB to Qwen.
    # Loading Qwen first leaves too little CUDA memory for long-call Silero VAD.
    regions = vad_regions(waveform, "cpu")
    model = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if selected == "cuda" else torch.float32,
        device_map="cuda:0" if selected == "cuda" else "cpu",
        max_inference_batch_size=2 if selected == "cuda" else 1,
        max_new_tokens=512,
    )
    recognized: list[tuple[int, int, str, float | None]] = []
    context = "Conversation vocabulary: " + prompt if prompt else ""
    inference_batch = 2 if selected == "cuda" else 1
    for batch_start in range(0, len(regions), inference_batch):
        batch_regions = regions[batch_start : batch_start + inference_batch]
        transcribe_options = {
            "audio": [(waveform[start:end], 16_000) for start, end in batch_regions],
            # Srota's model card explicitly requires language=None for its
            # Hindi/English code-mixed fine-tune.
            "language": [None if candidate == "srota" else "Hindi"] * len(batch_regions),
        }
        if candidate != "srota":
            transcribe_options["context"] = [context] * len(batch_regions)
        responses = model.transcribe(**transcribe_options)
        for offset, ((start, end), response) in enumerate(zip(batch_regions, responses, strict=True), 1):
            index = batch_start + offset
            recognized.append((start, end, str(response.text), None))
            _notify(
                progress,
                f"{CANDIDATE_LABELS[candidate]} chunk {index}/{len(regions)}",
                index / max(1, len(regions)),
            )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return TranscriptionResult(
        source=source,
        model=model_id,
        device=selected,
        language="hi",
        language_probability=1.0,
        duration=waveform.size / 16_000,
        segments=_segments_from_text_chunks(recognized, waveform),
        provenance=(
            f"{candidate}: 25 s VAD chunks with 1.5 s overlap; speech-active fallback timings",
            "loaded from local Hugging Face cache" if loaded_offline else "downloaded from Hugging Face on first use",
        ),
    )


def _transcribe_ai4bharat600(
    source: Path,
    waveform: np.ndarray,
    device: str,
    token: str,
    progress: ProgressCallback | None,
) -> TranscriptionResult:
    if not importlib.util.find_spec("transformers"):
        raise RuntimeError("Transformers is not installed. Run setup-accuracy.ps1 first.")
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        MODEL_IDS["ai4bharat600"], trust_remote_code=True, token=token or None
    )
    regions = vad_regions(waveform, device)
    recognized = []
    for index, (start, end) in enumerate(regions, 1):
        audio = torch.from_numpy(waveform[start:end]).unsqueeze(0)
        text = model(audio, "hi", "rnnt")
        if isinstance(text, (tuple, list)):
            text = text[0] if text else ""
        recognized.append((start, end, str(text), None))
        _notify(progress, f"AI4Bharat 600M RNNT chunk {index}/{len(regions)}", index / max(1, len(regions)))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return TranscriptionResult(
        source=source,
        model=MODEL_IDS["ai4bharat600"],
        device="cuda" if device != "cpu" and torch.cuda.is_available() else "cpu",
        language="hi",
        language_probability=1.0,
        duration=waveform.size / 16_000,
        segments=_segments_from_text_chunks(recognized, waveform),
        provenance=("AI4Bharat 600M RNNT; 25 s VAD chunks with 1.5 s overlap",),
    )


def _transcribe_vaani_whisper(
    source: Path,
    waveform: np.ndarray,
    device: str,
    token: str,
    progress: ProgressCallback | None,
) -> TranscriptionResult:
    if not importlib.util.find_spec("transformers"):
        raise RuntimeError("Transformers is not installed. Run setup-accuracy.ps1 first.")
    import torch
    from transformers import pipeline

    selected = 0 if device != "cpu" and torch.cuda.is_available() else -1
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=MODEL_IDS["vaani-whisper"],
        device=selected,
        token=token or None,
        dtype=torch.float16 if selected == 0 else torch.float32,
        chunk_length_s=25,
        stride_length_s=(1.5, 1.5),
    )
    _notify(progress, "Running Vaani Whisper Hindi...", 0.1)
    output = recognizer(
        {"array": waveform, "sampling_rate": 16_000},
        return_timestamps="word",
        generate_kwargs={"language": "hi", "task": "transcribe"},
    )
    words = []
    for chunk in output.get("chunks", []):
        bounds = chunk.get("timestamp") or (None, None)
        if bounds[0] is None or bounds[1] is None or float(bounds[1]) <= float(bounds[0]):
            continue
        words.append(WordTiming(float(bounds[0]), float(bounds[1]), str(chunk.get("text", "")).strip()))
    if words:
        from .alignment import words_to_segments

        segments = words_to_segments(words, max_gap=0.8)
    else:
        text = str(output.get("text", "")).strip()
        segments = (
            Segment(0.0, waveform.size / 16_000, text, words=proportional_word_timings(text, 0.0, waveform.size / 16_000)),
        ) if text else ()
    del recognizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return TranscriptionResult(
        source=source,
        model=MODEL_IDS["vaani-whisper"],
        device="cuda" if selected == 0 else "cpu",
        language="hi",
        language_probability=1.0,
        duration=waveform.size / 16_000,
        segments=tuple(segments),
        provenance=("Vaani Whisper native word timestamps",),
    )


def transcribe_candidate(
    candidate: str,
    source: str | Path,
    waveform: np.ndarray,
    *,
    device: str = "auto",
    language: str | None = "hi",
    prompt: str = "",
    hf_token: str = "",
    progress: ProgressCallback | None = None,
) -> TranscriptionResult:
    source_path = Path(source).resolve()
    if candidate not in CANDIDATE_LABELS:
        raise ValueError(f"Unknown ASR candidate: {candidate}")
    if candidate in {"qwen3", "srota", "orato"}:
        return _transcribe_qwen(candidate, source_path, waveform, device, prompt, progress)
    if candidate == "ai4bharat600":
        return _transcribe_ai4bharat600(source_path, waveform, device, hf_token, progress)
    if candidate == "vaani-whisper":
        return _transcribe_vaani_whisper(source_path, waveform, device, hf_token, progress)

    engine = "sravaani" if candidate == "sravaani" else "indicconformer" if candidate == "legacy" else "whisper"
    model_size = "large-v3" if candidate == "whisper-large" else "medium"
    transcriber = Transcriber(model_size=model_size, device=device, engine=engine, progress_callback=progress)
    result = transcriber._transcribe_indicconformer(source_path, language) if engine != "whisper" else transcriber.transcribe(source_path, language, prompt)
    return result
