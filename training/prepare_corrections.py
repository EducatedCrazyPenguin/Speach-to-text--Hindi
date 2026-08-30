from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import random
import sys
import tarfile
import wave

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from voice_to_text.core import Transcriber  # noqa: E402


def wav_bytes(waveform: np.ndarray) -> bytes:
    output = BytesIO()
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(pcm.tobytes())
    return output.getvalue()


def split_calls(calls: list[dict]) -> dict[str, list[dict]]:
    if len(calls) < 3:
        raise ValueError("At least three independently corrected calls are needed for call-level train/val/test splits")
    shuffled = list(calls)
    random.Random(20260830).shuffle(shuffled)
    count = len(shuffled)
    train_count = max(1, round(count * 0.8))
    val_count = max(1, round(count * 0.1))
    if train_count + val_count >= count:
        train_count = count - 2
        val_count = 1
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare verified call corrections as SraVaani tarred shards")
    parser.add_argument("--corrections", type=Path, default=PROJECT_ROOT / "corrections")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "training" / "data")
    parser.add_argument("--minimum-hours", type=float, default=3.0)
    parser.add_argument("--allow-less", action="store_true")
    args = parser.parse_args()

    calls = []
    total_duration = 0.0
    for path in sorted(args.corrections.glob("*.corrected.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("correction_metadata", {})
        if not metadata.get("verified") or metadata.get("held_out"):
            continue
        source = Path(payload["source"])
        if not source.is_file():
            raise FileNotFoundError(f"Retained source audio is missing for {path.name}: {source}")
        duration = sum(float(item["end"]) - float(item["start"]) for item in payload.get("segments", []))
        total_duration += duration
        calls.append({"path": path, "payload": payload, "duration": duration})
    if total_duration < args.minimum_hours * 3600 and not args.allow_less:
        raise ValueError(
            f"Only {total_duration / 3600:.2f} verified hours are available; collect {args.minimum_hours:.1f} hours first"
        )

    splits = split_calls(calls)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"verified_hours": total_duration / 3600, "held_out_benchmark_included": False, "splits": {}}
    for split, split_calls_list in splits.items():
        split_dir = args.output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        tar_path = split_dir / "shard_0000.tar"
        manifest_path = split_dir / "shard_0000.json"
        manifest_lines = []
        utterance_index = 0
        with tarfile.open(tar_path, "w") as archive:
            for call in split_calls_list:
                payload = call["payload"]
                waveform = Transcriber._decode_audio(Path(payload["source"]))
                for segment in payload.get("segments", []):
                    start, end = float(segment["start"]), float(segment["end"])
                    text = str(segment.get("text", "")).strip()
                    if not text or end - start < 0.2:
                        continue
                    clip = waveform[round(start * 16_000) : round(end * 16_000)]
                    name = f"{split}_{utterance_index:06d}.wav"
                    content = wav_bytes(clip)
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    archive.addfile(info, BytesIO(content))
                    manifest_lines.append(
                        json.dumps(
                            {"audio_filepath": name, "text": text, "duration": len(clip) / 16_000},
                            ensure_ascii=False,
                        )
                    )
                    utterance_index += 1
        manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        summary["splits"][split] = {
            "calls": len(split_calls_list),
            "utterances": utterance_index,
            "hours": sum(call["duration"] for call in split_calls_list) / 3600,
        }
    (args.output / "split-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
