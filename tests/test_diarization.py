from voice_to_text.core import Segment
from voice_to_text.diarization import SpeakerTurn, assign_speakers, merge_speaker_turns


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
