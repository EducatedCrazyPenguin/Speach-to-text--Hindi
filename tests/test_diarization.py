import numpy as np

from voice_to_text.core import Segment, WordTiming
from voice_to_text.diarization import (
    SpeakerTurn,
    _choose_pitch_label,
    assign_speakers,
    assign_word_speakers,
    merge_speaker_turns,
    refine_speaker_words_with_embeddings,
    smooth_speaker_turns,
)


def test_pitch_fallback_requires_clear_call_level_separation() -> None:
    assert _choose_pitch_label(124.0, {"Mohit": 126.0, "Wife": 177.0}) == "Mohit"
    assert _choose_pitch_label(187.0, {"Mohit": 126.0, "Wife": 177.0}) == "Wife"
    assert _choose_pitch_label(150.0, {"A": 145.0, "B": 155.0}) is None


def test_assigns_two_speakers_by_overlap_and_first_appearance() -> None:
    segments = (
        Segment(0.0, 2.0, "Hello"),
        Segment(2.0, 4.0, "नमस्ते"),
        Segment(4.0, 5.0, "Okay"),
    )
    turns = (
        SpeakerTurn(0.0, 1.9, "SPEAKER_07"),
        SpeakerTurn(2.1, 3.9, "SPEAKER_02"),
        SpeakerTurn(4.0, 5.0, "SPEAKER_07"),
    )

    labelled = assign_speakers(segments, turns, ("Mohit", "Wife"))

    assert [item.speaker for item in labelled] == ["Mohit", "Wife", "Mohit"]


def test_merges_nearby_same_speaker_turns_but_not_other_speakers() -> None:
    turns = (
        SpeakerTurn(0.0, 1.0, "A"),
        SpeakerTurn(1.2, 2.0, "A"),
        SpeakerTurn(2.0, 3.0, "B"),
        SpeakerTurn(3.1, 4.0, "A"),
        SpeakerTurn(4.1, 4.2, "A"),
    )

    merged = merge_speaker_turns(turns)

    assert merged == (
        SpeakerTurn(0.0, 2.0, "A"),
        SpeakerTurn(2.0, 3.0, "B"),
        SpeakerTurn(3.1, 4.0, "A"),
    )


def test_assigns_speakers_after_asr_and_marks_real_overlap() -> None:
    words = (
        WordTiming(0.0, 0.8, "हाँ"),
        WordTiming(0.8, 1.6, "माँ"),
    )
    segments = (Segment(0.0, 1.6, "हाँ माँ", words=words),)
    exclusive = (SpeakerTurn(0.0, 0.8, "A"), SpeakerTurn(0.8, 1.6, "B"))
    raw = (
        SpeakerTurn(0.0, 1.3, "A"),
        SpeakerTurn(0.7, 1.6, "B"),
    )

    labelled = assign_word_speakers(segments, exclusive, raw, ("Mohit", "Mum"))

    assert [item.speaker for item in labelled] == ["Mohit", "Mum"]
    assert labelled[1].overlap is True


def test_smooths_sub_300ms_flip_and_joins_sub_350ms_gap() -> None:
    turns = (
        SpeakerTurn(0.0, 1.0, "A"),
        SpeakerTurn(1.0, 1.2, "B"),
        SpeakerTurn(1.2, 2.0, "A"),
        SpeakerTurn(2.3, 3.0, "A"),
    )

    assert smooth_speaker_turns(turns) == (SpeakerTurn(0.0, 3.0, "A"),)


def test_short_embedding_padding_is_excluded_from_speech_mask() -> None:
    captured: dict[str, float] = {}

    class FakeEmbedding:
        min_num_samples = 16_000

        def __call__(self, _waveforms, masks):
            captured["mask_samples"] = float(masks.sum())
            return np.array([[1.0, 0.0]], dtype=np.float32)

    class FakePipeline:
        _embedding = FakeEmbedding()
        embedding_batch_size = 16

    word = WordTiming(0.10, 0.50, "नमस्ते", 1.0, "Speaker 1")
    segment = Segment(0.10, 0.50, "नमस्ते", "Speaker 1", words=(word,))
    refine_speaker_words_with_embeddings(
        (segment,),
        np.zeros(32_000, dtype=np.float32),
        FakePipeline(),
        ("A", "B"),
        np.eye(2, dtype=np.float32),
        (SpeakerTurn(0.0, 1.0, "A"), SpeakerTurn(1.0, 2.0, "B")),
        ("Speaker 1", "Speaker 2"),
    )

    assert captured["mask_samples"] == 7_680
