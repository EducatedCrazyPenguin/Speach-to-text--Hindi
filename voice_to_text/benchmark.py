from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from .audio import write_pcm16_wav
from .core import Segment, Transcriber
from .metrics import AccuracyReport, evaluate


BENCHMARK_WINDOWS = (
    ("mum-complete", "mum", 0.0, 47.664),
    ("wifey-opening", "wifey", 0.0, 72.336),
    ("wifey-09m", "wifey", 540.0, 660.0),
    ("wifey-19m", "wifey", 1140.0, 1260.0),
    ("wifey-29m", "wifey", 1740.0, 1860.0),
    ("wifey-ending", "wifey", 2213.124, 2333.124),
)
EXPECTED_DURATION = 600.0


def benchmark_root(project_root: str | Path) -> Path:
    return Path(project_root).resolve() / "benchmarks" / "gold-10min"


def _discover_sources(project_root: Path) -> dict[str, Path]:
    mum_candidates = [
        Path.home() / "Downloads" / "Telegram Desktop" / "Mum-2608291821.mp3",
        *project_root.glob("recordings/*Mum*.mp3"),
    ]
    wifey_candidates = [
        project_root / "recordings" / "20260829_214719_9e7143_Wifey_-2608291826.mp3",
        *project_root.glob("recordings/*Wifey*.mp3"),
    ]
    mum = next((path for path in mum_candidates if path.is_file()), None)
    wifey = next(
        (
            path
            for path in wifey_candidates
            if path.is_file() and len(Transcriber._decode_audio(path)) >= int(2333.124 * 16_000)
        ),
        None,
    )
    if mum is None or wifey is None:
        raise FileNotFoundError("Could not find both the 47-second Mum sample and 38:53 Wifey recording")
    return {"mum": mum, "wifey": wifey}


def prepare_benchmark(
    project_root: str | Path,
    *,
    mum_source: str | Path | None = None,
    wifey_source: str | Path | None = None,
) -> Path:
    """Create the fixed 600-second held-out set without modifying source recordings."""
    project = Path(project_root).resolve()
    sources = _discover_sources(project)
    if mum_source:
        sources["mum"] = Path(mum_source).resolve()
    if wifey_source:
        sources["wifey"] = Path(wifey_source).resolve()
    waveforms = {name: Transcriber._decode_audio(path) for name, path in sources.items()}
    root = benchmark_root(project)
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    total = 0.0
    for item_id, source_key, start, end in BENCHMARK_WINDOWS:
        first, last = round(start * 16_000), round(end * 16_000)
        waveform = waveforms[source_key]
        if last > waveform.size:
            raise ValueError(f"{sources[source_key].name} is too short for benchmark window {item_id}")
        output = write_pcm16_wav(audio_dir / f"{item_id}.wav", waveform[first:last])
        duration = (last - first) / 16_000
        total += duration
        entries.append(
            {
                "id": item_id,
                "audio": str(output.relative_to(project)).replace("\\", "/"),
                "source_name": sources[source_key].name,
                "source_start": start,
                "source_end": end,
                "duration": duration,
                "gold_segments": [],
                "complete": False,
            }
        )
    if abs(total - EXPECTED_DURATION) > 0.001:
        raise AssertionError(f"Benchmark must be exactly 600 seconds, got {total}")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "held_out": True,
                "training_excluded": True,
                "duration_seconds": total,
                "items": entries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def load_manifest(project_root: str | Path) -> dict[str, Any]:
    path = benchmark_root(project_root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError("The 10-minute benchmark has not been prepared yet")
    return json.loads(path.read_text(encoding="utf-8"))


def save_gold_segments(project_root: str | Path, item_id: str, segments: Iterable[dict]) -> dict[str, Any]:
    root = benchmark_root(project_root)
    manifest_path = root / "manifest.json"
    manifest = load_manifest(project_root)
    item = next((entry for entry in manifest["items"] if entry["id"] == item_id), None)
    if item is None:
        raise KeyError(f"Unknown benchmark item: {item_id}")
    validated = []
    last_start = 0.0
    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        text = str(segment.get("text", "")).strip()
        speaker = str(segment.get("speaker", "")).strip() or None
        if start < last_start or end <= start or end > float(item["duration"]) + 0.01:
            raise ValueError("Gold segment timestamps must be ordered and inside the clip")
        if text:
            validated.append({"start": start, "end": end, "text": text, "speaker": speaker})
        last_start = start
    item["gold_segments"] = validated
    item["complete"] = bool(validated)
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    return item


def item_reference(item: dict[str, Any]) -> tuple[str, tuple[Segment, ...]]:
    segments = tuple(
        Segment(float(entry["start"]), float(entry["end"]), str(entry["text"]), entry.get("speaker"))
        for entry in item.get("gold_segments", [])
    )
    return " ".join(segment.text for segment in segments), segments


def evaluate_item(
    item: dict[str, Any],
    hypothesis_text: str,
    hypothesis_segments: tuple[Segment, ...],
    names: Iterable[str] = (),
) -> AccuracyReport:
    text, segments = item_reference(item)
    return evaluate(
        text,
        hypothesis_text,
        names=names,
        reference_segments=segments,
        hypothesis_segments=hypothesis_segments,
    )


@dataclass(frozen=True)
class PromotionDecision:
    primary: str
    secondary: str | None
    use_consensus: bool
    reason: str


def promotion_decision(
    model_reports: dict[str, AccuracyReport],
    consensus_reports: dict[str, AccuracyReport],
) -> PromotionDecision:
    if not model_reports:
        raise ValueError("No model reports were supplied")
    primary = min(model_reports, key=lambda name: model_reports[name].wer)
    best = model_reports[primary]
    eligible_consensus = {
        name: report
        for name, report in consensus_reports.items()
        if best.wer - report.wer >= 0.015
        and report.number_accuracy >= best.number_accuracy
        and report.name_accuracy >= best.name_accuracy
    }
    if eligible_consensus:
        name = min(eligible_consensus, key=lambda key: eligible_consensus[key].wer)
        pieces = name.split("+", 1)
        secondary = pieces[1] if len(pieces) == 2 else None
        return PromotionDecision(primary, secondary, True, "Consensus improved WER by at least 1.5 points")
    return PromotionDecision(primary, None, False, "Best single model retained; consensus gate was not met")
