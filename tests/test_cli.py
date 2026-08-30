from transcribe import build_parser


def test_cli_defaults() -> None:
    args = build_parser().parse_args(["conversation.m4a"])
    assert args.model == "medium"
    assert args.device == "auto"
    assert args.engine == "whisper"
    assert args.language is None

