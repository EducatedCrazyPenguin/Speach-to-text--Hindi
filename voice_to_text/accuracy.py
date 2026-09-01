from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
from typing import Iterable

from .audio import analyze_audio, prepare_audio_variant, write_pcm16_wav
from .consensus import consensus_result
from .core import ProgressCallback, Transcriber, TranscriptionResult, _notify
from .diarization import diarize_two_speakers
from .exports import result_from_dict, result_to_dict
from .forced_alignment import force_align_hindi_words
from .models import CANDIDATE_LABELS, optional_model_availability, transcribe_candidate
from .readable import generate_readable_copy
from .recovery import recover_transcript


_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{8,}")
_CHECKPOINT_FORMAT_VERSION = 2


@dataclass(frozen=True)
class AccuracySettings:
    device: str = "auto"
    language: str = "hi"
    audio_mode: str = "original"
    candidates: tuple[str, ...] = ()
    use_consensus: bool = False
    diarize: bool = True
    speaker_names: tuple[str, str] = ("Speaker 1", "Speaker 2")
    glossary: tuple[str, ...] = ()
    hf_token: str = ""
    generate_readable: bool = True
    use_readable_model: bool = True
    allow_model_downloads: bool = False
    force_align_words: bool = True
    enhanced_recovery: bool = False


def recommended_candidates(project_root: Path) -> tuple[str, ...]:
    promoted = project_root / "benchmarks" / "promoted.json"
    if promoted.is_file():
        try:
            payload = json.loads(promoted.read_text(encoding="utf-8"))
            selected = tuple(item for item in payload.get("candidates", []) if item in CANDIDATE_LABELS)
            if selected:
                return selected[:2]
        except (OSError, json.JSONDecodeError):
            pass
    available = optional_model_availability()
    # Qwen is the provisional fallback after the supplied 47.664 s call scored
    # materially better than SraVaani. The permanent corrected benchmark still
    # controls promotion through benchmarks/promoted.json.
    if available.get("qwen3"):
        return ("qwen3",)
    return ("sravaani",)


def promoted_defaults(project_root: Path) -> dict[str, object]:
    path = project_root / "benchmarks" / "promoted.json"
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return {
                "audio_mode": payload.get("audio_mode", "original"),
                "use_consensus": bool(payload.get("use_consensus", False)),
                "target_met": bool(payload.get("target_met", False)),
            }
        except (OSError, json.JSONDecodeError):
            pass
    return {"audio_mode": "original", "use_consensus": False, "target_met": False}


class AccuracyPipeline:
    """Resumable, ASR-first maximum-accuracy local pipeline."""

    def __init__(
        self,
        project_root: str | Path,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.progress_callback = progress_callback

    def _checkpoint_path(self, directory: Path, name: str) -> Path:
        return directory / f"{name}.v{_CHECKPOINT_FORMAT_VERSION}.json"

    def _save_result(self, directory: Path, name: str, result: TranscriptionResult) -> None:
        path = self._checkpoint_path(directory, name)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result_to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _load_result(self, directory: Path, name: str) -> TranscriptionResult | None:
        path = self._checkpoint_path(directory, name)
        if not path.is_file():
            return None
        try:
            return result_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return None

    def transcribe(
        self,
        source: str | Path,
        settings: AccuracySettings,
        *,
        checkpoint_dir: str | Path | None = None,
    ) -> TranscriptionResult:
        source_path = Path(source).resolve()
        if settings.hf_token:
            # Keep the credential in process memory only. This lets public/gated
            # model loaders reuse the one token stored in Windows Credential Manager.
            os.environ["HF_TOKEN"] = settings.hf_token
        checkpoint = Path(checkpoint_dir or self.project_root / ".jobs" / source_path.stem)
        checkpoint.mkdir(parents=True, exist_ok=True)

        _notify(self.progress_callback, "Analyzing recording quality...", 0.02)
        waveform = Transcriber._decode_audio(source_path)
        diagnostics = analyze_audio(waveform)
        variant, processing = prepare_audio_variant(waveform, diagnostics, settings.audio_mode)
        input_path = source_path
        if settings.audio_mode != "original":
            input_path = write_pcm16_wav(checkpoint / "asr-input.wav", variant)
        (checkpoint / "diagnostics.json").write_text(
            json.dumps({**diagnostics.to_dict(), **processing}, indent=2), encoding="utf-8"
        )

        candidates = settings.candidates or recommended_candidates(self.project_root)
        candidates = tuple(dict.fromkeys(item for item in candidates if item in CANDIDATE_LABELS))
        if not candidates:
            candidates = ("sravaani",)
        results: list[TranscriptionResult] = []
        errors: dict[str, str] = {}
        span = 0.58 / len(candidates)
        for index, candidate in enumerate(candidates):
            cached = self._load_result(checkpoint, f"asr-{candidate}")
            if cached is not None:
                results.append(replace(cached, source=source_path))
                _notify(self.progress_callback, f"Resumed {CANDIDATE_LABELS[candidate]} result", 0.10 + span * (index + 1))
                continue

            base = 0.08 + span * index

            def candidate_progress(message: str, value: float | None) -> None:
                mapped = base + span * max(0.0, min(1.0, value or 0.0))
                _notify(self.progress_callback, message, mapped)

            _notify(self.progress_callback, f"Running {CANDIDATE_LABELS[candidate]}...", base)
            try:
                result = transcribe_candidate(
                    candidate,
                    input_path,
                    variant,
                    device=settings.device,
                    language=settings.language,
                    prompt=", ".join(settings.glossary),
                    hf_token=settings.hf_token,
                    progress=candidate_progress,
                )
                result = replace(result, source=source_path)
                self._save_result(checkpoint, f"asr-{candidate}", result)
                results.append(result)
            except Exception as exc:
                errors[candidate] = _TOKEN_PATTERN.sub("hf_[redacted]", str(exc))

        if not results:
            details = "; ".join(f"{name}: {message}" for name, message in errors.items())
            raise RuntimeError(f"All selected local ASR models failed. {details}")

        result = results[0]
        if settings.use_consensus and len(results) >= 2:
            _notify(self.progress_callback, "Building the two-model consensus...", 0.68)
            result = consensus_result(results[0], results[1])

        combined_diagnostics = {
            **diagnostics.to_dict(),
            "audio_processing": processing,
            "candidate_models": [item.model for item in results],
            "candidate_errors": errors,
            "alignment_note": "Approximate word times are replaced by multilingual Hindi forced alignment before diarization.",
        }
        result = replace(
            result,
            source=source_path,
            diagnostics=combined_diagnostics,
            provenance=result.provenance + ("ASR before diarization",),
            raw_segments=result.raw_segments or result.segments,
        )
        self._save_result(checkpoint, "selected-asr-recovery1", result)

        if settings.force_align_words:
            aligned = self._load_result(checkpoint, "forced-aligned-recovery1")
            if aligned is None:
                def alignment_progress(message: str, value: float | None) -> None:
                    _notify(self.progress_callback, message, 0.68 + 0.10 * max(0.0, min(1.0, value or 0.0)))

                _notify(self.progress_callback, "Forced-aligning Hindi words to the recording...", 0.68)
                try:
                    result = force_align_hindi_words(
                        result,
                        variant,
                        settings.device,
                        alignment_progress,
                    )
                except Exception as exc:
                    alignment_diagnostics = dict(result.diagnostics)
                    alignment_diagnostics["word_alignment_error"] = _TOKEN_PATTERN.sub(
                        "hf_[redacted]", str(exc)
                    )
                    result = replace(result, diagnostics=alignment_diagnostics)
                self._save_result(checkpoint, "forced-aligned-recovery1", result)
            else:
                result = replace(aligned, source=source_path)
                _notify(self.progress_callback, "Resumed Hindi word alignment", 0.78)

        if settings.enhanced_recovery:
            recovered = self._load_result(checkpoint, "evidence-recovered3")
            if recovered is None:
                def recovery_progress(message: str, value: float | None) -> None:
                    _notify(
                        self.progress_callback,
                        message,
                        0.78 + 0.12 * max(0.0, min(1.0, value or 0.0)),
                    )

                result = recover_transcript(
                    result,
                    variant,
                    device=settings.device,
                    glossary=settings.glossary,
                    allow_downloads=settings.allow_model_downloads,
                    progress_callback=recovery_progress,
                )
                self._save_result(checkpoint, "evidence-recovered3", result)
            else:
                result = replace(recovered, source=source_path)
                _notify(self.progress_callback, "Resumed evidence-grounded ASR recovery", 0.90)

        if settings.diarize:
            diarization_checkpoint = (
                "diarized-aligned-embedding4-recovery3"
                if settings.force_align_words and settings.enhanced_recovery
                else "diarized-aligned-embedding4"
                if settings.force_align_words
                else "diarized"
            )
            diarized = self._load_result(checkpoint, diarization_checkpoint)
            if diarized is None:
                def diarization_progress(message: str, value: float | None) -> None:
                    _notify(self.progress_callback, message, 0.90 + 0.05 * max(0.0, min(1.0, value or 0.0)))

                result = diarize_two_speakers(
                    result,
                    settings.hf_token,
                    settings.speaker_names,
                    settings.device,
                    diarization_progress,
                )
                self._save_result(checkpoint, diarization_checkpoint, result)
            else:
                result = replace(diarized, source=source_path)
                _notify(self.progress_callback, "Resumed speaker identification result", 0.95)

        if settings.generate_readable:
            _notify(self.progress_callback, "Creating the separate readable Hindi copy...", 0.96)
            readable, method = generate_readable_copy(
                result,
                settings.glossary,
                use_local_model=settings.use_readable_model,
                allow_download=settings.allow_model_downloads,
            )
            diagnostics_with_readable = dict(result.diagnostics)
            diagnostics_with_readable["readable_copy_method"] = method
            result = replace(result, readable_text=readable, diagnostics=diagnostics_with_readable)

        self._save_result(checkpoint, "complete", result)
        _notify(self.progress_callback, "Maximum-accuracy transcription complete", 1.0)
        return result
