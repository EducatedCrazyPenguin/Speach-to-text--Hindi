from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from voice_to_text.core import ENGINE_CHOICES, LANGUAGES, MODEL_CHOICES, Transcriber
from voice_to_text.accuracy import AccuracyPipeline, AccuracySettings
from voice_to_text.diarization import diarize_two_speakers
from voice_to_text.exports import write_outputs
from voice_to_text.models import CANDIDATE_LABELS
from voice_to_text.secrets import load_token


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe an Indian-language conversation locally."
    )
    parser.add_argument("audio", type=Path, help="audio or video recording")
    parser.add_argument(
        "--preset", choices=["fast", "maximum"], default="fast",
        help="maximum uses the ASR-first accuracy pipeline; fast preserves legacy behavior",
    )
    parser.add_argument("--audio-mode", choices=["original", "telephone", "auto"], default="original")
    parser.add_argument("--candidate", choices=list(CANDIDATE_LABELS), default="sravaani")
    parser.add_argument("--secondary-candidate", choices=list(CANDIDATE_LABELS))
    parser.add_argument("--consensus", action="store_true")
    parser.add_argument("--readable", action="store_true")
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
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            try:
                token = load_token()
            except OSError:
                token = ""
        names = tuple(name.strip() for name in args.speaker_names.split(",") if name.strip())
        if len(names) != 2:
            raise ValueError("--speaker-names must contain exactly two comma-separated names")
        if args.preset == "maximum":
            candidates = tuple(item for item in (args.candidate, args.secondary_candidate) if item)
            result = AccuracyPipeline(Path(__file__).resolve().parent, progress).transcribe(
                args.audio,
                AccuracySettings(
                    device=args.device,
                    language=args.language or "hi",
                    audio_mode=args.audio_mode,
                    candidates=candidates,
                    use_consensus=args.consensus,
                    diarize=args.diarize,
                    speaker_names=(names[0], names[1]),
                    glossary=tuple(item.strip() for item in (args.prompt or "").split(",") if item.strip()),
                    hf_token=token,
                    generate_readable=args.readable,
                ),
            )
        else:
            transcriber = Transcriber(
                model_size=args.model,
                device=args.device,
                engine=args.engine,
                progress_callback=progress,
            )
            result = transcriber.transcribe(args.audio, args.language, args.prompt)
            if args.diarize:
                result = diarize_two_speakers(result, token, names, args.device, progress)
        paths = write_outputs(result, args.output_dir)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(paths["txt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
