import numpy as np

import voice_to_text.profiles as profiles


def test_profile_matching_is_one_to_one_and_thresholded(monkeypatch) -> None:
    monkeypatch.setattr(profiles, "list_profiles", lambda: ("Wife", "Mum"))
    monkeypatch.setattr(
        profiles,
        "load_profile",
        lambda name: np.array([1.0, 0.0], dtype=np.float32) if name == "Wife" else np.array([0.0, 1.0], dtype=np.float32),
    )

    matches = profiles.match_speaker_profiles(
        ("A", "B"), np.array([[0.99, 0.01], [0.02, 0.98]], dtype=np.float32)
    )

    assert matches["A"][0] == "Wife"
    assert matches["B"][0] == "Mum"
