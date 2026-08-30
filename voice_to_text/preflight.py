from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path


QWEN_MODEL = "Qwen/Qwen3-ASR-1.7B"


def _cached_snapshot(model_id: str) -> str | None:
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the local maximum-accuracy runtime.")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--ensure-qwen", action="store_true")
    parser.add_argument("--ensure-alignment", action="store_true")
    args = parser.parse_args()

    import torch

    if args.require_gpu and not torch.cuda.is_available():
        print("ERROR: CUDA is not visible to PyTorch.")
        return 2
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback'}")

    snapshot = _cached_snapshot(QWEN_MODEL)
    if snapshot is None and args.ensure_qwen:
        print("Qwen3-ASR 1.7B is not cached. Downloading it once (about 4.7 GB)...")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(QWEN_MODEL)
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
