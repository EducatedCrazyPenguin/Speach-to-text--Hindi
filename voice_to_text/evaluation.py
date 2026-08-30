from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Iterable

from .accuracy import AccuracyPipeline, AccuracySettings
from .benchmark import benchmark_root, item_reference, load_manifest
from .consensus import consensus_result
from .core import ProgressCallback, Segment, TranscriptionResult, _notify
from .exports import result_from_dict, result_to_dict
from .metrics import AccuracyReport, evaluate
from .models import CANDIDATE_LABELS, optional_model_availability


def _offset_segments(segments: Iterable[Segment], offset: float) -> tuple[Segment, ...]:
    return tuple(replace(item, start=item.start + offset, end=item.end + offset) for item in segments)


def _aggregate(
    manifest: dict,
    results: dict[str, TranscriptionResult],
    names: Iterable[str],
) -> AccuracyReport:
    reference_texts: list[str] = []
    hypothesis_texts: list[str] = []
    reference_segments: list[Segment] = []
    hypothesis_segments: list[Segment] = []
    offset = 0.0
    for item in manifest["items"]:
        reference_text, gold = item_reference(item)
        result = results[item["id"]]
        reference_texts.append(reference_text)
        hypothesis_texts.append(result.text)
        reference_segments.extend(_offset_segments(gold, offset))
        hypothesis_segments.extend(_offset_segments(result.segments, offset))
        offset += float(item["duration"]) + 1.0
    return evaluate(
        " ".join(reference_texts),
        " ".join(hypothesis_texts),
        names=names,
        reference_segments=reference_segments,
        hypothesis_segments=hypothesis_segments,
    )


def run_benchmark(
    project_root: str | Path,
    *,
    candidates: Iterable[str],
    device: str = "auto",
    hf_token: str = "",
    speaker_names: tuple[str, str] = ("Speaker 1", "Speaker 2"),
    glossary: tuple[str, ...] = (),
    progress_callback: ProgressCallback | None = None,
) -> dict:
    project = Path(project_root).resolve()
    manifest = load_manifest(project)
    if not all(item.get("complete") for item in manifest["items"]):
        raise ValueError("Correct all six benchmark clips before running model evaluation")
    selected = tuple(dict.fromkeys(item for item in candidates if item in CANDIDATE_LABELS))
    availability = optional_model_availability()
    selected = tuple(item for item in selected if availability.get(item, False))
    if not selected:
        raise ValueError("None of the selected model runtimes is installed")
    root = benchmark_root(project)
    result_root = root / "results"
    result_root.mkdir(parents=True, exist_ok=True)
    total_runs = len(selected) * 2 * len(manifest["items"])
    completed_runs = 0
    reports: dict[str, dict] = {}
    all_results: dict[tuple[str, str], dict[str, TranscriptionResult]] = {}
    successful_candidates: list[str] = []

    for candidate in selected:
        candidate_succeeded = True
        for audio_mode in ("original", "telephone"):
            key = f"{candidate}:{audio_mode}"
            model_results: dict[str, TranscriptionResult] = {}
            started = time.perf_counter()
            destination = result_root / candidate / audio_mode
            destination.mkdir(parents=True, exist_ok=True)
            timing_path = destination / "runtime.json"
            torch_runtime = None
            try:
                import torch

                if torch.cuda.is_available() and device != "cpu":
                    torch.cuda.reset_peak_memory_stats()
                    torch_runtime = torch
            except ImportError:
                pass
            try:
                for item in manifest["items"]:
                    item_id = item["id"]
                    cached_path = destination / f"{item_id}.json"
                    if cached_path.is_file():
                        result = result_from_dict(json.loads(cached_path.read_text(encoding="utf-8")))
                    else:
                        pipeline = AccuracyPipeline(project)
                        result = pipeline.transcribe(
                            project / item["audio"],
                            AccuracySettings(
                                device=device,
                                audio_mode=audio_mode,
                                candidates=(candidate,),
                                diarize=True,
                                speaker_names=speaker_names,
                                glossary=glossary,
                                hf_token=hf_token,
                                generate_readable=False,
                            ),
                            checkpoint_dir=destination / f"{item_id}.checkpoint",
                        )
                        cached_path.write_text(
                            json.dumps(result_to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                    model_results[item_id] = result
                    completed_runs += 1
                    _notify(
                        progress_callback,
                        f"Benchmarked {CANDIDATE_LABELS[candidate]} ({audio_mode}) on {item_id}",
                        completed_runs / max(1, total_runs),
                    )
            except Exception as exc:
                reports[key] = {"error": str(exc)}
                candidate_succeeded = False
                break
            runtime = time.perf_counter() - started
            peak_gpu_gb = (
                float(torch_runtime.cuda.max_memory_allocated()) / (1024**3)
                if torch_runtime is not None
                else 0.0
            )
            if timing_path.is_file() and all((destination / f"{item['id']}.json").is_file() for item in manifest["items"]):
                try:
                    timing = json.loads(timing_path.read_text(encoding="utf-8"))
                    runtime = float(timing["runtime_seconds_10min"])
                    peak_gpu_gb = float(timing.get("peak_gpu_memory_gb", peak_gpu_gb))
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    pass
            else:
                timing_path.write_text(
                    json.dumps(
                        {"runtime_seconds_10min": runtime, "peak_gpu_memory_gb": peak_gpu_gb}
                    ),
                    encoding="utf-8",
                )
            report = _aggregate(manifest, model_results, glossary)
            reports[key] = {
                **report.to_dict(),
                "runtime_seconds_10min": runtime,
                "estimated_runtime_seconds_40min": runtime * 4,
                "peak_gpu_memory_gb": peak_gpu_gb,
            }
            all_results[(candidate, audio_mode)] = model_results
        if candidate_succeeded:
            successful_candidates.append(candidate)

    selected = tuple(successful_candidates)
    if not selected:
        raise RuntimeError(f"Every installed candidate failed: {reports}")

    # Promote processing only when it clears the one-point and entity gates.
    best_variants: dict[str, str] = {}
    for candidate in selected:
        original = reports[f"{candidate}:original"]
        processed = reports[f"{candidate}:telephone"]
        if (
            original["wer"] - processed["wer"] >= 0.01
            and processed["number_accuracy"] >= original["number_accuracy"]
            and processed["name_accuracy"] >= original["name_accuracy"]
        ):
            best_variants[candidate] = "telephone"
        else:
            best_variants[candidate] = "original"

    primary = min(selected, key=lambda item: reports[f"{item}:{best_variants[item]}"]["wer"])
    primary_report = reports[f"{primary}:{best_variants[primary]}"]
    secondary = next(
        (
            item
            for item in sorted(
                (candidate for candidate in selected if candidate != primary),
                key=lambda value: reports[f"{value}:{best_variants[value]}"]["wer"],
            )
        ),
        None,
    )
    consensus_payload = None
    use_consensus = False
    if secondary:
        consensus_results = {
            item["id"]: consensus_result(
                all_results[(primary, best_variants[primary])][item["id"]],
                all_results[(secondary, best_variants[secondary])][item["id"]],
            )
            for item in manifest["items"]
        }
        consensus_report = _aggregate(manifest, consensus_results, glossary)
        ensemble_runtime = (
            primary_report["estimated_runtime_seconds_40min"]
            + reports[f"{secondary}:{best_variants[secondary]}"]["estimated_runtime_seconds_40min"]
        )
        consensus_payload = {
            **consensus_report.to_dict(),
            "estimated_runtime_seconds_40min": ensemble_runtime,
        }
        use_consensus = (
            primary_report["wer"] - consensus_report.wer >= 0.015
            and consensus_report.number_accuracy >= primary_report["number_accuracy"]
            and consensus_report.name_accuracy >= primary_report["name_accuracy"]
            and ensemble_runtime <= 1800
        )

    promoted_candidates = [primary]
    if use_consensus and secondary:
        promoted_candidates.append(secondary)
    winning_report = consensus_payload if use_consensus and consensus_payload else primary_report
    promoted_runtime = float(winning_report["estimated_runtime_seconds_40min"])
    promoted = {
        "candidates": promoted_candidates,
        "audio_mode": best_variants[primary],
        "use_consensus": use_consensus,
        "target_met": bool(winning_report["meets_target"] and promoted_runtime <= 1800),
        "primary_report": primary_report,
        "consensus_report": consensus_payload,
        "gates": {
            "wer_max": 0.15,
            "cer_max": 0.08,
            "name_number_min": 0.90,
            "speaker_min": 0.95,
            "runtime_seconds_40min_max": 1800,
        },
    }
    (project / "benchmarks" / "promoted.json").write_text(
        json.dumps(promoted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    payload = {"reports": reports, "promoted": promoted}
    (root / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
