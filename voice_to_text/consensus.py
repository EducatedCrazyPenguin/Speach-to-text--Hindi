from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .alignment import flatten_words, normalized_token, words_to_segments
from .core import TranscriptionResult, WordTiming


def _alignment_path(primary: Sequence[str], secondary: Sequence[str]) -> list[tuple[int | None, int | None]]:
    rows, columns = len(primary) + 1, len(secondary) + 1
    scores = [[0] * columns for _ in range(rows)]
    moves = [[""] * columns for _ in range(rows)]
    for row in range(1, rows):
        scores[row][0], moves[row][0] = row, "up"
    for column in range(1, columns):
        scores[0][column], moves[0][column] = column, "left"
    for row in range(1, rows):
        for column in range(1, columns):
            options = (
                (scores[row - 1][column - 1] + (primary[row - 1] != secondary[column - 1]), "diag"),
                (scores[row - 1][column] + 1, "up"),
                (scores[row][column - 1] + 1, "left"),
            )
            scores[row][column], moves[row][column] = min(options, key=lambda item: item[0])
    path: list[tuple[int | None, int | None]] = []
    row, column = len(primary), len(secondary)
    while row or column:
        move = moves[row][column]
        if move == "diag":
            row -= 1
            column -= 1
            path.append((row, column))
        elif move == "up":
            row -= 1
            path.append((row, None))
        else:
            column -= 1
            path.append((None, column))
    return list(reversed(path))


def consensus_result(
    primary: TranscriptionResult,
    secondary: TranscriptionResult,
    *,
    secondary_margin: float = 0.15,
) -> TranscriptionResult:
    """Build a conservative two-model transcript.

    The primary model wins ties and all unconfident disagreements. This result is
    only intended for promotion after gold-set evaluation proves a WER benefit.
    """
    first = list(flatten_words(primary.segments))
    second = list(flatten_words(secondary.segments))
    first_tokens = [normalized_token(word.text) for word in first]
    second_tokens = [normalized_token(word.text) for word in second]
    chosen: list[WordTiming] = []
    for primary_index, secondary_index in _alignment_path(first_tokens, second_tokens):
        if primary_index is None:
            candidate = second[secondary_index]  # type: ignore[index]
            if (candidate.confidence or 0.0) >= 0.85:
                chosen.append(candidate)
            continue
        primary_word = first[primary_index]
        if secondary_index is None or first_tokens[primary_index] == second_tokens[secondary_index]:
            chosen.append(primary_word)
            continue
        secondary_word = second[secondary_index]
        primary_confidence = primary_word.confidence if primary_word.confidence is not None else 0.5
        secondary_confidence = secondary_word.confidence if secondary_word.confidence is not None else 0.5
        if secondary_confidence >= primary_confidence + secondary_margin:
            chosen.append(
                replace(
                    secondary_word,
                    start=primary_word.start,
                    end=primary_word.end,
                    speaker=primary_word.speaker,
                    overlap=primary_word.overlap,
                )
            )
        else:
            chosen.append(primary_word)
    segments = words_to_segments(chosen)
    return replace(
        primary,
        model=f"consensus({primary.model}, {secondary.model})",
        segments=segments,
        provenance=primary.provenance + secondary.provenance + ("conservative token consensus",),
    )
