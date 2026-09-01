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


def test_profile_uses_compact_binary_credential(monkeypatch) -> None:
    stored: dict[str, bytes] = {}
    monkeypatch.setattr(
        profiles,
        "save_secret_bytes",
        lambda value, target: stored.__setitem__(target, value),
    )
    monkeypatch.setattr(
        profiles,
        "load_secret_bytes",
        lambda target: stored.get(target, b""),
    )
    monkeypatch.setattr(profiles, "list_profiles", lambda: ())
    monkeypatch.setattr(profiles, "save_token", lambda *_args, **_kwargs: None)

    source = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
    profiles.save_profile("Mohit", source)
    payload = stored[profiles._target("Mohit")]
    restored = profiles.load_profile("Mohit")
    expected = source / np.linalg.norm(source)

    assert payload.startswith(b"SPK2")
    assert len(payload) < 2560
    assert float(np.dot(expected, restored)) > 0.9999


def test_single_profile_anchors_best_separated_call_cluster(monkeypatch) -> None:
    monkeypatch.setattr(profiles, "list_profiles", lambda: ("Mohit",))
    monkeypatch.setattr(
        profiles,
        "load_profile",
        lambda _name: np.array([1.0, 0.0], dtype=np.float32),
    )

    matches = profiles.match_speaker_profiles(
        ("wife-cluster", "mohit-cluster"),
        np.array([[0.10, 0.99], [0.80, 0.20]], dtype=np.float32),
    )

    assert matches["mohit-cluster"][0] == "Mohit"
    assert "wife-cluster" not in matches


def test_profile_matching_ignores_case_only_duplicate_names(monkeypatch) -> None:
    monkeypatch.setattr(profiles, "list_profiles", lambda: ("Mohit", "Wife", "mohit"))
    monkeypatch.setattr(
        profiles,
        "load_profile",
        lambda name: np.array([1.0, 0.0], dtype=np.float32)
        if name.casefold() == "mohit"
        else np.array([0.0, 1.0], dtype=np.float32),
    )
    matches = profiles.match_speaker_profiles(
        ("A", "B"), np.array([[0.99, 0.01], [0.01, 0.99]], dtype=np.float32)
    )
    assert {value[0] for value in matches.values()} == {"Mohit", "Wife"}
