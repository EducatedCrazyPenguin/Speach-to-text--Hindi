from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys


os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")


QWEN_MODEL = "Qwen/Qwen3-ASR-1.7B"
VAANI_MODEL = "ARTPARK-IISc/whisper-large-v3-vaani-hindi"
READABLE_MODEL = "Qwen/Qwen3.5-4B"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
READABLE_RUNTIME = PROJECT_ROOT / ".cache" / "readable-transformers"


def _cached_snapshot(model_id: str) -> str | None:
    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(model_id, local_files_only=True)
        return snapshot if _snapshot_has_weights(Path(snapshot)) else None
    except Exception:
        return None


def _snapshot_has_weights(snapshot: Path) -> bool:
    indexes = tuple(snapshot.glob("*.safetensors.index.json"))
    for index in indexes:
        try:
            files = set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values())
        except (OSError, json.JSONDecodeError):
            return False
        if files and all((snapshot / name).is_file() for name in files):
            return True
    if indexes:
        return False
    return any(snapshot.glob("*.safetensors")) or any(snapshot.glob("pytorch_model*.bin"))


def _download_snapshot(model_id: str) -> str:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import enable_progress_bars

    enable_progress_bars()
    return snapshot_download(model_id, max_workers=1)


def _ensure_readable_runtime() -> None:
    if (READABLE_RUNTIME / "transformers" / "__init__.py").is_file():
        return
    READABLE_RUNTIME.mkdir(parents=True, exist_ok=True)
    print("Installing the isolated Qwen3.5 Transformers runtime...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(READABLE_RUNTIME),
            "--upgrade",
            "transformers>=5.0.0,<6",
        ]
    )


def _ensure_torchvision(torch) -> None:
    try:
        import torchvision  # noqa: F401

        return
    except (ImportError, RuntimeError, OSError):
        pass
    version = "torchvision==0.23.0+cu129" if torch.version.cuda else "torchvision==0.23.0"
    command = [sys.executable, "-m", "pip", "install", "--no-deps", version]
    if torch.version.cuda:
        command.extend(["--index-url", "https://download.pytorch.org/whl/cu129"])
    print(f"Installing {version} for the Qwen3.5 processor...")
    subprocess.check_call(command)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the local maximum-accuracy runtime.")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--ensure-qwen", action="store_true")
    parser.add_argument("--ensure-alignment", action="store_true")
    parser.add_argument("--ensure-recovery", action="store_true")
    parser.add_argument("--ensure-readable", action="store_true")
    args = parser.parse_args()

    import torch

    if args.require_gpu and not torch.cuda.is_available():
        print("ERROR: CUDA is not visible to PyTorch.")
        return 2
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback'}")

    snapshot = _cached_snapshot(QWEN_MODEL)
    if snapshot is None and args.ensure_qwen:
        print("Qwen3-ASR 1.7B is not cached. Downloading it once (about 4.7 GB)...")
        snapshot = _download_snapshot(QWEN_MODEL)
    print(f"Qwen3-ASR: {'cached locally' if snapshot else 'downloads on first use'}")

    from torchaudio.pipelines import MMS_FA

    alignment_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(MMS_FA._path).name
    if not alignment_path.is_file() and args.ensure_alignment:
        print("Hindi word aligner is not cached. Downloading it once (about 1.2 GB)...")
        alignment_model = MMS_FA.get_model()
        del alignment_model
        gc.collect()
    print(
        f"Hindi forced alignment: {'cached locally' if alignment_path.is_file() else 'downloads on first use'}"
    )

    vaani = _cached_snapshot(VAANI_MODEL)
    if vaani is None and args.ensure_recovery:
        print("Vaani Whisper Hindi is not cached. Downloading it once (about 6.2 GB)...")
        vaani = _download_snapshot(VAANI_MODEL)
    print(f"Vaani retry model: {'cached locally' if vaani else 'not installed'}")

    readable = _cached_snapshot(READABLE_MODEL)
    if args.ensure_readable:
        _ensure_readable_runtime()
        _ensure_torchvision(torch)
        if readable is None:
            print("Qwen3.5-4B is not cached. Downloading it once (about 9.3 GB)...")
            readable = _download_snapshot(READABLE_MODEL)
    runtime_ready = (READABLE_RUNTIME / "transformers" / "__init__.py").is_file()
    print(
        "Readable-copy model: "
        + ("cached with isolated runtime" if readable and runtime_ready else "not fully installed")
    )

    from .diarization import speaker_model_is_cached
    from .secrets import has_token

    speaker_ready = speaker_model_is_cached()
    credential_ready = has_token()
    if speaker_ready:
        print("Speaker identification: model cached locally")
    elif credential_ready:
        print("Speaker identification: read token stored securely; model downloads on first use")
    else:
        print("Speaker identification: optional setup still needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
