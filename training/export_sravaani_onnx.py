from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def main() -> int:
    parser = argparse.ArgumentParser(description="Export an accepted SraVaani .nemo checkpoint for onnx-asr")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.restore_from(str(args.checkpoint)).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "set_export_config"):
        model.set_export_config({"decoder_type": "tdt"})
    model.export(str(args.output / "model.onnx"))

    # NeMo versions use either prefix below for hybrid transducer exports.
    aliases = {
        "model_encoder.onnx": "encoder-model.onnx",
        "model_decoder_joint.onnx": "decoder_joint-model.onnx",
        "model-encoder.onnx": "encoder-model.onnx",
        "model-decoder_joint.onnx": "decoder_joint-model.onnx",
    }
    for source_name, destination_name in aliases.items():
        source = args.output / source_name
        if source.is_file() and not (args.output / destination_name).exists():
            shutil.move(source, args.output / destination_name)
    required = [args.output / "encoder-model.onnx", args.output / "decoder_joint-model.onnx"]
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"NeMo export did not produce the expected TDT graphs: {required}")
    vocabulary = [*model.tokenizer.vocab, "<blk>"]
    with (args.output / "vocab.txt").open("w", encoding="utf-8") as handle:
        for index, token in enumerate(vocabulary):
            handle.write(f"{token} {index}\n")
    (args.output / "config.json").write_text(
        json.dumps(
            {
                "model_type": "nemo-conformer-tdt",
                "features_size": int(getattr(model.cfg.preprocessor, "features", 128)),
                "subsampling_factor": 8,
                "max_tokens_per_step": 10,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
