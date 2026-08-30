from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from voice_to_text.core import ENGINE_CHOICES, LANGUAGES, MODEL_CHOICES, Transcriber
from voice_to_text.diarization import diarize_then_transcribe_two_speakers, diarize_two_speakers
from voice_to_text.exports import write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe an Indian-language conversation locally."
    )
    parser.add_argument("audio", type=Path, help="audio or video recording")
    parser.add_argument(
        "--engine",
        choices=list(ENGINE_CHOICES.values()),
        default="whisper",
        help="Whisper for mixed speech or AI4Bharat IndicConformer for one Indic language",
    )
    parser.add_argument(
        "--language",
        choices=[code for code in LANGUAGES.values() if code],
        help="spoken language code; omit to auto-detect",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CHOICES.values()),
        default="medium",
        help="accuracy/speed tradeoff (default: medium)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="processing device (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transcripts"),
        help="folder for TXT, SRT, Markdown and JSON files",
    )
    parser.add_argument(
        "--prompt",
        help="optional names or uncommon words that may improve recognition",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="identify exactly two voices; requires setup-diarization.ps1 and HF_TOKEN",
    )
    parser.add_argument(
        "--speaker-names",
        default="Speaker 1,Speaker 2",
        help="two comma-separated names, in order of first appearance",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def progress(message: str, value: float | None) -> None:
        suffix = f" ({value:.0%})" if value is not None else ""
        print(f"{message}{suffix}", file=sys.stderr, flush=True)

    try:
        transcriber = Transcriber(
            model_size=args.model,
            device=args.device,
            engine=args.engine,
            progress_callback=progress,
        )
        if args.diarize:
            token = os.environ.get("HF_TOKEN", "")
            names = tuple(name.strip() for name in args.speaker_names.split(",") if name.strip())
            if len(names) != 2:
                raise ValueError("--speaker-names must contain exactly two comma-separated names")
            if args.engine == "indicconformer":
                result = diarize_then_transcribe_two_speakers(
                    transcriber, args.audio, args.language, token, names, args.device, progress
                )
            else:
                result = transcriber.transcribe(args.audio, args.language, args.prompt)
                result = diarize_two_speakers(result, token, names, args.device, progress)
        else:
            result = transcriber.transcribe(args.audio, args.language, args.prompt)
        paths = write_outputs(result, args.output_dir)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(paths["txt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
