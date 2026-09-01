from pathlib import Path

import numpy as np

from voice_to_text.alignment import (
    deduplicate_overlapping_segments,
    proportional_word_timings,
    speech_weighted_word_timings,
    token_timings_to_words,
    validate_word_timings,
)
from voice_to_text.audio import analyze_audio, prepare_audio_variant
from voice_to_text.benchmark import BENCHMARK_WINDOWS, EXPECTED_DURATION
from voice_to_text.core import Segment, WordTiming
from voice_to_text.metrics import evaluate, normalize_text, speaker_attribution_accuracy, word_error_rate
from voice_to_text.forced_alignment import normalize_for_mms
from voice_to_text.models import _snapshot_has_weights, prompt_copy_detected
from voice_to_text.readable import validate_readable_copy, validate_structured_turns
from voice_to_text.readable_worker import _extract_json
from voice_to_text.recovery import (
    RecoverySpan,
    candidate_score,
    context_copy_rejected,
    merge_recovery_spans,
    repetition_count,
)
from voice_to_text.scripts import to_devanagari


def test_benchmark_windows_are_exactly_ten_minutes() -> None:
    assert sum(end - start for _, _, start, end in BENCHMARK_WINDOWS) == EXPECTED_DURATION


def test_overlapping_chunk_text_is_deduplicated() -> None:
    first = Segment(0, 4, "हाँ माँ अभी चौसठ", words=proportional_word_timings("हाँ माँ अभी चौसठ", 0, 4))
    second = Segment(3, 7, "अभी चौसठ चल रही है", words=proportional_word_timings("अभी चौसठ चल रही है", 3, 7))

    result = deduplicate_overlapping_segments((first, second))

    assert result[1].text == "चल रही है"


def test_bad_token_alignment_falls_back_to_valid_monotonic_words() -> None:
    words = token_timings_to_words(
        text="नमस्ते माँ",
        start=1.0,
        end=3.0,
        tokens=[" नमस्ते", " माँ"],
        timestamps=[2.0, 0.0],
    )

    assert [word.text for word in words] == ["नमस्ते", "माँ"]
    assert validate_word_timings(words, 3.0) == words


def test_speech_weighted_fallback_skips_long_silence() -> None:
    waveform = np.zeros(16_000 * 4, dtype=np.float32)
    time = np.arange(16_000, dtype=np.float32) / 16_000
    tone = 0.2 * np.sin(2 * np.pi * 300 * time)
    waveform[:16_000] = tone
    waveform[48_000:] = tone

    words = speech_weighted_word_timings("पहला दूसरा", waveform, 0.0)

    assert words[0].end < 1.5
    assert words[1].start > 2.5


def test_audio_diagnostics_detect_narrowband_and_processing_is_bounded() -> None:
    time = np.arange(16_000 * 2, dtype=np.float32) / 16_000
    waveform = (0.1 * np.sin(2 * np.pi * 1000 * time)).astype(np.float32)

    diagnostics = analyze_audio(waveform)
    processed, metadata = prepare_audio_variant(waveform, diagnostics, "telephone")

    assert diagnostics.bandwidth_limited is True
    assert np.max(np.abs(processed)) <= 10 ** (-1 / 20) + 1e-5
    assert metadata["dereverb_applied"] is False


def test_metrics_canonicalize_common_spoken_numbers() -> None:
    assert normalize_text("एक सौ आठ और चौसठ") == "108 और 64"
    assert normalize_text("सिक्सटी फोर, फिफ्टी फोर और असी") == "64 54 और 80"
    assert word_error_rate("अभी चौसठ चल रही है", "अभी 64 चल रही है") == 0.0
    report = evaluate("मोहित ने अस्सी कहा", "मोहित ने 80 कहा", names=("मोहित",))
    assert report.number_accuracy == 1.0
    assert report.name_accuracy == 1.0


def test_speaker_accuracy_does_not_penalize_missing_asr_coverage_twice() -> None:
    reference = (Segment(0, 10, "हाँ", "Mohit"),)
    hypothesis = (Segment(2, 4, "हाँ", "Mohit"),)
    assert speaker_attribution_accuracy(reference, hypothesis) == 1.0


def test_readable_validator_rejects_changed_number_or_added_content() -> None:
    source = "माँ: अभी 64 चल रही है।"
    assert validate_readable_copy(source, "माँ: अभी 64 चल रही है।", ("माँ",))
    assert not validate_readable_copy(source, "माँ: अभी 80 चल रही है।", ("माँ",))
    assert not validate_readable_copy(source, "माँ: अभी 64 चल रही है और कल बाजार जाना है।", ("माँ",))
    assert not validate_readable_copy("सब सही है", "सब सही है 9 बजे")


def test_structured_readable_validation_rejects_missing_or_reordered_turns() -> None:
    segments = (Segment(0, 1, "हाँ जी"), Segment(1, 2, "सब सही है"))
    assert validate_structured_turns(segments, [{"id": 2, "text": "हाँ जी"}]) is None
    assert validate_structured_turns(
        segments,
        [{"id": 2, "text": "हाँ जी"}, {"id": 1, "text": "सब सही है"}],
    ) is None


def test_readable_worker_extracts_one_array_without_greedy_extra_data() -> None:
    payload = _extract_json(
        '```json\n[{"id":1,"text":"हाँ जी।"}]\n```\nExplanation with [extra] text.'
    )
    assert payload == [{"id": 1, "text": "हाँ जी।"}]


def test_recovery_detects_repeats_and_merges_bounded_windows() -> None:
    assert repetition_count("सही है सही है") > 0
    merged = merge_recovery_spans(
        (RecoverySpan(2, 3, ("repeat",)), RecoverySpan(4, 5, ("fragment",))),
        30,
    )
    assert len(merged) == 1
    assert 6 <= merged[0].end - merged[0].start <= 18
    assert set(merged[0].reasons) == {"repeat", "fragment"}


def test_recovery_rejects_context_copy_and_penalizes_repetition() -> None:
    assert context_copy_rejected(
        "कल मंदिर में जाकर वापस आना है",
        "Following: कल मंदिर में जाकर वापस आना है",
        "आज घर पर हूँ",
    )
    plain = (WordTiming(0, 1, "सही", 0.8), WordTiming(1, 2, "है", 0.8))
    repeated = plain + (WordTiming(2, 3, "सही", 0.8), WordTiming(3, 4, "है", 0.8))
    assert candidate_score(plain, "सही है", ("सही है",), 2) > candidate_score(
        repeated, "सही है सही है", ("सही है सही है",), 4
    )


def test_neighboring_indic_scripts_are_phonetically_mapped_to_devanagari() -> None:
    assert to_devanagari("ਹਾਂ ਮਾ") == "हां मा"
    assert to_devanagari("કામ") == "काम"


def test_mms_normalization_keeps_one_alignment_unit_per_source_word() -> None:
    class FakeRomanizer:
        @staticmethod
        def romanize_string(_text, lcode=None):
            assert lcode == "hin"
            return "HaaM, 64"

    assert normalize_for_mms("हाँ64", FakeRomanizer()) == "haam"


def test_context_copy_is_detected_for_acoustic_recheck() -> None:
    prompt = "Vocabulary: केडी busy call cramps मंदिर"
    assert prompt_copy_detected("केडी busy call cramps मंदिर केडी busy call", prompt)
    assert not prompt_copy_detected("केडी को काम दिया और फिर call किया", prompt)


def test_incomplete_sharded_model_is_not_reported_as_cached(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"model-00001-of-00002.safetensors",'
        '"b":"model-00002-of-00002.safetensors"}}',
        encoding="utf-8",
    )
    (tmp_path / "model-00002-of-00002.safetensors").touch()
    assert not _snapshot_has_weights(tmp_path)
    (tmp_path / "model-00001-of-00002.safetensors").touch()
    assert _snapshot_has_weights(tmp_path)
