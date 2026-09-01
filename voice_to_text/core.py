from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


LANGUAGES: dict[str, str | None] = {
    "Auto-detect": None,
    "Assamese": "as",
    "Bengali": "bn",
    "Bodo": "brx",
    "Dogri": "doi",
    "English": "en",
    "Gujarati": "gu",
    "Hindi / mixed Hindi + English": "hi",
    "Kannada": "kn",
    "Kashmiri": "ks",
    "Konkani": "kok",
    "Maithili": "mai",
    "Malayalam": "ml",
    "Manipuri": "mni",
    "Marathi": "mr",
    "Nepali": "ne",
    "Odia": "or",
    "Punjabi": "pa",
    "Sanskrit": "sa",
    "Santali": "sat",
    "Sindhi": "sd",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur",
}

ENGINE_CHOICES: dict[str, str] = {
    "Maximum local accuracy (SraVaani TDT)": "sravaani",
    "Whisper - best for mixed Hindi + English": "whisper",
    "Legacy AI4Bharat 120M CTC": "indicconformer",
}

MODEL_CHOICES: dict[str, str] = {
    "Small - quick test": "small",
    "Medium - recommended and already installed": "medium",
    "Large v3 Turbo - optional first-time download": "turbo",
    "Large v3 - highest accuracy": "large-v3",
}

INDICCONFORMER_MODELS: dict[str, str] = {
    code: f"OpenVoiceOS/ai4bharat-indicconformer-{code}-onnx"
    for code in (
        "as",
        "bn",
        "brx",
        "doi",
        "gu",
        "hi",
        "kn",
        "ks",
        "kok",
        "mai",
        "ml",
        "mni",
        "mr",
        "ne",
        "or",
        "pa",
        "sa",
        "sat",
        "sd",
        "ta",
        "te",
        "ur",
    )
}

ProgressCallback = Callable[[str, float | None], None]
_DLL_HANDLES: list[object] = []
SRAVAANI_MODEL = "OpenVoiceOS/artpark-iisc-vaani-fastconformer-multi-onnx"


@dataclass(frozen=True)
class WordTiming:
    """A recognized word with validated timing and optional speaker metadata."""

    start: float
    end: float
    text: str
    confidence: float | None = None
    speaker: str | None = None
    overlap: bool = False
    origin: str = "primary"
    alternatives: tuple[str, ...] = ()
    uncertain: bool = False
    repetition_removed: bool = False


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    confidence: float | None = None
    overlap: bool = False
    words: tuple[WordTiming, ...] = ()
    uncertain: bool = False


@dataclass(frozen=True)
class TranscriptionResult:
    source: Path
    model: str
    device: str
    language: str
    language_probability: float
    duration: float
    segments: tuple[Segment, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    provenance: tuple[str, ...] = ()
    readable_text: str | None = None
    raw_segments: tuple[Segment, ...] = ()

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments).strip()


def _notify(callback: ProgressCallback | None, message: str, progress: float | None = None) -> None:
    if callback:
        callback(message, progress)


def _cuda_is_visible() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _register_torch_cuda_dlls() -> None:
    """Let CTranslate2 reuse CUDA/cuDNN DLLs from an optional PyTorch install."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    try:
        import torch

        dll_dir = Path(torch.__file__).resolve().parent / "lib"
        if dll_dir.is_dir():
            handle = os.add_dll_directory(str(dll_dir))
            _DLL_HANDLES.append(handle)
    except (ImportError, OSError):
        pass


class Transcriber:
    """Load a Whisper model once and transcribe one or more recordings."""

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "auto",
        engine: str = "whisper",
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        if model_size not in MODEL_CHOICES.values():
            raise ValueError(f"Unsupported model: {model_size}")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if engine not in ENGINE_CHOICES.values():
            raise ValueError(f"Unsupported engine: {engine}")

        self.model_size = model_size
        self.requested_device = device
        self.engine = engine
        self.progress_callback = progress_callback
        self.device = "cpu"
        self._model = None
        self._loaded_language: str | None = None

    def _create_model(self, device: str):
        if device == "cuda":
            _register_torch_cuda_dlls()
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Run setup.ps1 first."
            ) from exc

        compute_type = "float16" if device == "cuda" else "int8"
        return WhisperModel(self.model_size, device=device, compute_type=compute_type)

    def _load_whisper(self) -> None:
        if self._model is not None:
            return

        preferred = self.requested_device
        if preferred == "auto":
            preferred = "cuda" if _cuda_is_visible() else "cpu"

        _notify(
            self.progress_callback,
            f"Loading {self.model_size} on {preferred}. The first run downloads the model...",
            0.02,
        )
        try:
            self._model = self._create_model(preferred)
            self.device = preferred
        except Exception as exc:
            if self.requested_device != "auto" or preferred != "cuda":
                raise
            _notify(
                self.progress_callback,
                f"GPU libraries are unavailable ({exc}). Falling back to CPU...",
                0.02,
            )
            self._model = self._create_model("cpu")
            self.device = "cpu"

    def _load_indicconformer(self, language: str | None) -> None:
        if self.engine == "indicconformer" and not language:
            raise ValueError(
                "AI4Bharat IndicConformer needs a selected Indian language; Auto-detect is unavailable."
            )
        if self.engine == "indicconformer" and language == "en":
            raise ValueError("AI4Bharat IndicConformer does not include English. Use Whisper for English.")
        model_id = (
            SRAVAANI_MODEL
            if self.engine == "sravaani"
            else INDICCONFORMER_MODELS.get(str(language))
        )
        if not model_id:
            raise ValueError(f"No IndicConformer model is configured for language '{language}'.")
        loaded_key = f"{self.engine}:{language or 'multi'}"
        if self._model is not None and self._loaded_language == loaded_key:
            return

        try:
            import onnx_asr
        except ImportError as exc:
            raise RuntimeError("onnx-asr is not installed. Run setup.ps1 again.") from exc

        providers = None
        selected_device = "cpu"
        if self.requested_device in {"auto", "cuda"}:
            try:
                _register_torch_cuda_dlls()
                import onnxruntime as ort

                if "CUDAExecutionProvider" in ort.get_available_providers():
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    selected_device = "cuda"
                elif self.requested_device == "cuda":
                    raise RuntimeError(
                        "ONNX CUDA support is not installed. Choose auto/CPU or install onnxruntime-gpu."
                    )
            except ImportError:
                if self.requested_device == "cuda":
                    raise RuntimeError("onnxruntime is not installed")

        _notify(
            self.progress_callback,
            (
                f"Loading SraVaani TDT on {selected_device}. First run downloads about 1 GB..."
                if self.engine == "sravaani"
                else f"Loading AI4Bharat {language} model on {selected_device}. First run downloads it..."
            ),
            0.02,
        )
        vad = onnx_asr.load_vad("silero", providers=providers)
        base_model = onnx_asr.load_model(model_id, providers=providers)
        self._model = base_model.with_vad(
            vad,
            batch_size=4,
            # Long context materially helps conversational Hindi. Padding on both
            # sides also creates a 1.5 s overlap when VAD splits long speech.
            min_silence_duration_ms=500,
            max_speech_duration_s=25,
            speech_pad_ms=750,
        ).with_timestamps()
        self._loaded_language = loaded_key
        self.device = selected_device

    @staticmethod
    def _decode_audio(audio_path: Path):
        """Decode any PyAV-supported file to mono float32 audio at 16 kHz."""
        try:
            import av
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Audio decoding dependencies are missing. Run setup.ps1 again.") from exc

        chunks = []
        with av.open(str(audio_path)) as container:
            audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
            if audio_stream is None:
                raise ValueError(f"No audio track found in {audio_path.name}")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16_000)
            for frame in container.decode(audio_stream):
                for converted in resampler.resample(frame):
                    chunks.append(converted.to_ndarray().reshape(-1))
            for converted in resampler.resample(None):
                chunks.append(converted.to_ndarray().reshape(-1))
        if not chunks:
            raise ValueError(f"No audio samples found in {audio_path.name}")
        return np.concatenate(chunks).astype("float32") / 32768.0

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        source = Path(audio_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Recording not found: {source}")

        if self.engine in {"indicconformer", "sravaani"}:
            return self._transcribe_indicconformer(source, language)

        self._load_whisper()
        try:
            return self._transcribe_whisper(source, language, initial_prompt)
        except Exception as exc:
            if self.requested_device != "auto" or self.device != "cuda":
                raise
            _notify(
                self.progress_callback,
                f"GPU inference could not start ({exc}). Retrying on CPU...",
                0.02,
            )
            self._model = self._create_model("cpu")
            self.device = "cpu"
            return self._transcribe_whisper(source, language, initial_prompt)

    def _transcribe_whisper(
        self,
        source: Path,
        language: str | None,
        initial_prompt: str | None,
    ) -> TranscriptionResult:
        _notify(self.progress_callback, "Detecting speech and transcribing...", 0.05)

        raw_segments, info = self._model.transcribe(
            str(source),
            language=language,
            task="transcribe",
            beam_size=5,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            # WhisperX and Buzz use independent VAD chunks to reduce looping/hallucination.
            condition_on_previous_text=False,
            # This field contains names and unusual terms, so use Whisper's hotword
            # biasing instead of a prompt that can be copied into short transcripts.
            hotwords=initial_prompt or None,
            word_timestamps=True,
            hallucination_silence_threshold=1.0,
            # Auto mode may contain Hindi/English code-switching. Re-detect for each
            # speech chunk, and use more than the first short chunk for initial detection.
            multilingual=language is None,
            language_detection_segments=3,
        )

        duration = float(getattr(info, "duration", 0.0) or 0.0)
        segments: list[Segment] = []
        for raw in raw_segments:
            text = raw.text.strip()
            start = max(0.0, float(raw.start))
            end = float(raw.end)
            if duration:
                # Timestamp-token hallucinations can extend far beyond a short file.
                # Never export text outside the recording that was actually decoded.
                if start >= duration:
                    continue
                end = min(end, duration)
            if text and end > start:
                words: list[WordTiming] = []
                for word in getattr(raw, "words", ()) or ():
                    word_start = max(start, float(getattr(word, "start", start)))
                    word_end = min(end, float(getattr(word, "end", end)))
                    word_text = str(getattr(word, "word", "")).strip()
                    if word_text and word_end > word_start:
                        words.append(
                            WordTiming(
                                word_start,
                                word_end,
                                word_text,
                                float(getattr(word, "probability", 0.0) or 0.0),
                            )
                        )
                segments.append(
                    Segment(
                        start,
                        end,
                        text,
                        confidence=float(getattr(raw, "avg_logprob", 0.0) or 0.0),
                        words=tuple(words),
                    )
                )
            if duration:
                progress = min(0.98, max(0.05, float(raw.end) / duration))
                _notify(self.progress_callback, f"Transcribed {raw.end:.0f} of {duration:.0f} seconds", progress)

        _notify(self.progress_callback, "Transcription complete", 1.0)
        return TranscriptionResult(
            source=source,
            model=self.model_size,
            device=self.device,
            language=str(getattr(info, "language", language or "unknown")),
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            duration=duration,
            segments=tuple(segments),
        )

    def _transcribe_indicconformer(
        self, source: Path, language: str | None
    ) -> TranscriptionResult:
        self._load_indicconformer(language)
        _notify(self.progress_callback, "Decoding the recording...", 0.04)
        waveform = self._decode_audio(source)
        duration = len(waveform) / 16_000
        _notify(self.progress_callback, "Detecting speech and transcribing...", 0.08)

        segments: list[Segment] = []
        for raw in self._model.recognize(waveform, sample_rate=16_000, channel="mean"):
            text = str(raw.text).strip()
            if self.engine == "sravaani" and language == "hi":
                from .scripts import to_devanagari

                text = to_devanagari(text)
            if text:
                from .alignment import token_timings_to_words

                start, end = float(raw.start), float(raw.end)
                segments.append(
                    Segment(
                        start,
                        end,
                        text,
                        words=token_timings_to_words(
                            text=text,
                            start=start,
                            end=end,
                            tokens=getattr(raw, "tokens", None),
                            timestamps=getattr(raw, "timestamps", None),
                            logprobs=getattr(raw, "logprobs", None),
                        ),
                    )
                )
            progress = min(0.98, max(0.08, float(raw.end) / duration)) if duration else None
            _notify(self.progress_callback, f"Transcribed through {raw.end:.0f} seconds", progress)

        _notify(self.progress_callback, "Transcription complete", 1.0)
        return TranscriptionResult(
            source=source,
            model=(SRAVAANI_MODEL if self.engine == "sravaani" else INDICCONFORMER_MODELS[str(language)]),
            device=self.device,
            language=str(language or "hi/multilingual"),
            language_probability=1.0,
            duration=duration,
            segments=tuple(segments),
        )

    def transcribe_indicconformer_regions(
        self,
        audio_path: str | Path,
        language: str | None,
        regions: Sequence[Segment],
        waveform=None,
    ) -> TranscriptionResult:
        """Transcribe pre-cut, single-speaker regions with IndicConformer.

        ``regions`` supplies the exact timestamps and speaker labels produced by
        diarization. Each region is decoded independently so one person's speech
        cannot contaminate the other person's ASR chunk.
        """
        if self.engine != "indicconformer":
            raise ValueError("Region transcription is only available for IndicConformer")

        source = Path(audio_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Recording not found: {source}")
        self._load_indicconformer(language)
        if waveform is None:
            _notify(self.progress_callback, "Decoding the recording...", 0.55)
            waveform = self._decode_audio(source)

        duration = len(waveform) / 16_000
        completed: list[Segment] = []
        total = max(1, len(regions))
        for index, region in enumerate(regions, 1):
            # A little context prevents clipped first/last syllables. The regions
            # themselves retain their exact diarization timestamps in the output.
            start = max(0.0, region.start - 0.12)
            end = min(duration, region.end + 0.12)
            clip = waveform[int(start * 16_000) : int(end * 16_000)]
            pieces = self._model.recognize(clip, sample_rate=16_000, channel="mean")
            text = " ".join(str(piece.text).strip() for piece in pieces).strip()
            if text:
                completed.append(
                    Segment(region.start, region.end, text, speaker=region.speaker)
                )
            progress = 0.55 + (0.43 * index / total)
            _notify(
                self.progress_callback,
                f"Transcribed speaker turn {index} of {total}",
                progress,
            )

        _notify(self.progress_callback, "Transcription complete", 1.0)
        return TranscriptionResult(
            source=source,
            model=INDICCONFORMER_MODELS[str(language)],
            device=self.device,
            language=str(language),
            language_probability=1.0,
            duration=duration,
            segments=tuple(completed),
        )


def supported_language_names() -> Iterable[str]:
    return LANGUAGES.keys()
