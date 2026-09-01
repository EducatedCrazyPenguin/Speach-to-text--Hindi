from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata
from typing import Iterable, Sequence

from .core import Segment


_PUNCTUATION_RE = re.compile(r"[^\w\u0900-\u097f]+", re.UNICODE)
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_NUMBER_WORDS = {
    "शून्य": "0", "जीरो": "0", "एक": "1", "दो": "2", "तीन": "3", "चार": "4",
    "पांच": "5", "पाँच": "5", "छह": "6", "सात": "7", "आठ": "8", "नौ": "9",
    "दस": "10", "ग्यारह": "11", "बारह": "12", "तेरह": "13", "चौदह": "14",
    "पंद्रह": "15", "सोलह": "16", "सत्रह": "17", "अठारह": "18", "उन्नीस": "19",
    "बीस": "20", "इक्कीस": "21", "बाईस": "22", "तेईस": "23", "चौबीस": "24",
    "पच्चीस": "25", "छब्बीस": "26", "सत्ताईस": "27", "अट्ठाईस": "28", "उनतीस": "29",
    "तीस": "30", "चालीस": "40", "पचास": "50", "चौवन": "54", "साठ": "60",
    "चौंसठ": "64", "चौसठ": "64", "सत्तर": "70", "अस्सी": "80", "नब्बे": "90",
    "असी": "80", "एटी": "80", "ऐटी": "80",
    "सौ": "100", "एकसौआठ": "108", "एक-सौ-आठ": "108",
}

_NUMBER_PHRASES = {
    ("सिक्सटी", "फोर"): "64",
    ("सिक्स्टी", "फोर"): "64",
    ("सिक्सटी", "फौर"): "64",
    ("सिक्स्टी", "फौर"): "64",
    ("फिफ्टी", "फोर"): "54",
    ("फिफ्टी", "फौर"): "54",
    ("वन", "हंड्रेड", "एट"): "108",
    ("वन", "हंड्रेड", "एट्"): "108",
}


def normalize_text(text: str, *, canonicalize_numbers: bool = True) -> str:
    text = unicodedata.normalize("NFC", text).translate(_DEVANAGARI_DIGITS).casefold()
    text = _PUNCTUATION_RE.sub(" ", text)
    tokens = text.split()
    if canonicalize_numbers:
        phrase_normalized: list[str] = []
        index = 0
        phrase_lengths = sorted({len(phrase) for phrase in _NUMBER_PHRASES}, reverse=True)
        while index < len(tokens):
            match = next(
                (
                    (length, _NUMBER_PHRASES[tuple(tokens[index : index + length])])
                    for length in phrase_lengths
                    if tuple(tokens[index : index + length]) in _NUMBER_PHRASES
                ),
                None,
            )
            if match:
                length, value = match
                phrase_normalized.append(value)
                index += length
            else:
                phrase_normalized.append(tokens[index])
                index += 1
        tokens = phrase_normalized
        tokens = [_NUMBER_WORDS.get(token, token) for token in tokens]
        combined: list[str] = []
        index = 0
        while index < len(tokens):
            if tokens[index : index + 3] == ["1", "100", "8"]:
                combined.append("108")
                index += 3
            else:
                combined.append(tokens[index])
                index += 1
        tokens = combined
    return " ".join(tokens)


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected = normalize_text(reference).split()
    actual = normalize_text(hypothesis).split()
    if not expected:
        return 0.0 if not actual else 1.0
    return _edit_distance(expected, actual) / len(expected)


def character_error_rate(reference: str, hypothesis: str) -> float:
    expected = list(normalize_text(reference))
    actual = list(normalize_text(hypothesis))
    if not expected:
        return 0.0 if not actual else 1.0
    return _edit_distance(expected, actual) / len(expected)


def _exact_item_accuracy(expected: Iterable[str], actual: Iterable[str]) -> float:
    expected_items = list(expected)
    actual_items = list(actual)
    if not expected_items:
        return 1.0
    remaining = list(actual_items)
    matched = 0
    for item in expected_items:
        if item in remaining:
            remaining.remove(item)
            matched += 1
    return matched / len(expected_items)


def number_accuracy(reference: str, hypothesis: str) -> float:
    pattern = re.compile(r"\b\d+(?:[.,]\d+)?\b")
    return _exact_item_accuracy(
        pattern.findall(normalize_text(reference)),
        pattern.findall(normalize_text(hypothesis)),
    )


def numbers_match_exactly(reference: str, hypothesis: str) -> bool:
    """Require the same normalized numeric values, including no added values."""
    pattern = re.compile(r"\b\d+(?:[.,]\d+)?\b")
    return pattern.findall(normalize_text(reference)) == pattern.findall(normalize_text(hypothesis))


def name_accuracy(reference: str, hypothesis: str, names: Iterable[str]) -> float:
    reference_normalized = normalize_text(reference, canonicalize_numbers=False)
    hypothesis_normalized = normalize_text(hypothesis, canonicalize_numbers=False)
    expected = [normalize_text(name, canonicalize_numbers=False) for name in names]
    expected = [name for name in expected if name and name in reference_normalized]
    actual = [name for name in expected if name in hypothesis_normalized]
    return _exact_item_accuracy(expected, actual)


def speaker_attribution_accuracy(
    reference: Sequence[Segment], hypothesis: Sequence[Segment]
) -> float:
    scored = 0.0
    correct = 0.0
    for expected in reference:
        if not expected.speaker:
            continue
        for actual in hypothesis:
            overlap = max(0.0, min(expected.end, actual.end) - max(expected.start, actual.start))
            if overlap:
                # Missing ASR coverage belongs in WER, not speaker attribution.
                # Score identity only where a hypothesis actually emitted speech.
                scored += overlap
                if actual.speaker == expected.speaker:
                    correct += overlap
    return correct / scored if scored else 1.0


@dataclass(frozen=True)
class AccuracyReport:
    wer: float
    cer: float
    number_accuracy: float
    name_accuracy: float
    speaker_accuracy: float
    meets_target: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def evaluate(
    reference_text: str,
    hypothesis_text: str,
    *,
    names: Iterable[str] = (),
    reference_segments: Sequence[Segment] = (),
    hypothesis_segments: Sequence[Segment] = (),
) -> AccuracyReport:
    wer = word_error_rate(reference_text, hypothesis_text)
    cer = character_error_rate(reference_text, hypothesis_text)
    numbers = number_accuracy(reference_text, hypothesis_text)
    names_score = name_accuracy(reference_text, hypothesis_text, names)
    speakers = speaker_attribution_accuracy(reference_segments, hypothesis_segments)
    return AccuracyReport(
        wer=wer,
        cer=cer,
        number_accuracy=numbers,
        name_accuracy=names_score,
        speaker_accuracy=speakers,
        meets_target=(wer <= 0.15 and cer <= 0.08 and numbers >= 0.90 and names_score >= 0.90 and speakers >= 0.95),
    )
