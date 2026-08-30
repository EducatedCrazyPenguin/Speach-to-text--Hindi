from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from pathlib import Path
import wave

import numpy as np


SAMPLE_RATE = 16_000


def _db(value: float, floor: float = -120.0) -> float:
    return max(floor, 20.0 * math.log10(max(1e-12, value)))


@dataclass(frozen=True)
class AudioDiagnostics:
    sample_rate: int
    duration_seconds: float
    peak_dbfs: float
    rms_dbfs: float
    clipping_percent: float
    dc_offset: float
    silence_percent: float
    estimated_snr_db: float
    spectral_centroid_hz: float
    rolloff_99_hz: float
    telephone_band_energy_percent: float
    high_band_energy_percent: float
    hum_50_60_percent: float
    reverb_proxy: float
    bandwidth_limited: bool
    noisy: bool
    clipped: bool
    reverb_detected: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def _frame_rms(waveform: np.ndarray, frame_size: int = 480) -> np.ndarray:
    if waveform.size < frame_size:
        return np.array([float(np.sqrt(np.mean(waveform * waveform) + 1e-12))])
    usable = waveform[: waveform.size - waveform.size % frame_size]
    frames = usable.reshape(-1, frame_size)
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)


def analyze_audio(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> AudioDiagnostics:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if not waveform.size:
        raise ValueError("Cannot analyze empty audio")
    peak = float(np.max(np.abs(waveform)))
    rms = float(np.sqrt(np.mean(waveform * waveform) + 1e-12))
    frame_rms = _frame_rms(waveform)
    noise_floor = float(np.percentile(frame_rms, 15))
    speech_level = float(np.percentile(frame_rms, 85))
    snr = max(0.0, min(60.0, _db(speech_level / max(noise_floor, 1e-8), floor=0.0)))
    silence_threshold = max(10 ** (-55 / 20), noise_floor * 1.5)
    silence = float(np.mean(frame_rms < silence_threshold) * 100.0)

    window_size = min(waveform.size, sample_rate * 120)
    spectral_audio = waveform[:window_size]
    window = np.hanning(spectral_audio.size)
    power = np.abs(np.fft.rfft(spectral_audio * window)) ** 2
    frequencies = np.fft.rfftfreq(spectral_audio.size, 1.0 / sample_rate)
    total_power = float(np.sum(power) + 1e-12)

    def band(low: float, high: float) -> float:
        mask = (frequencies >= low) & (frequencies < high)
        return float(np.sum(power[mask]) / total_power * 100.0)

    centroid = float(np.sum(frequencies * power) / total_power)
    cumulative = np.cumsum(power)
    rolloff_index = min(len(frequencies) - 1, int(np.searchsorted(cumulative, total_power * 0.99)))
    rolloff = float(frequencies[rolloff_index])
    telephone_energy = band(200, 3400)
    high_energy = band(4000, min(sample_rate / 2, 7900))
    hum = band(47, 53) + band(57, 63)

    # A stable late autocorrelation peak is a conservative proxy for room echo.
    reverb_window = spectral_audio[: min(spectral_audio.size, sample_rate * 20)]
    reverb_window = reverb_window - float(np.mean(reverb_window))
    if reverb_window.size > sample_rate // 2:
        fft_size = 1 << (2 * reverb_window.size - 1).bit_length()
        spectrum = np.fft.rfft(reverb_window, fft_size)
        correlation = np.fft.irfft(spectrum * np.conj(spectrum), fft_size)[: reverb_window.size]
        correlation /= max(float(correlation[0]), 1e-12)
        late = correlation[int(0.05 * sample_rate) : int(0.35 * sample_rate)]
        reverb_proxy = float(np.max(np.abs(late))) if late.size else 0.0
    else:
        reverb_proxy = 0.0

    clipping = float(np.mean(np.abs(waveform) >= 0.999) * 100.0)
    bandwidth_limited = rolloff < 3800 and high_energy < 2.0
    return AudioDiagnostics(
        sample_rate=sample_rate,
        duration_seconds=waveform.size / sample_rate,
        peak_dbfs=_db(peak),
        rms_dbfs=_db(rms),
        clipping_percent=clipping,
        dc_offset=float(np.mean(waveform)),
        silence_percent=silence,
        estimated_snr_db=snr,
        spectral_centroid_hz=centroid,
        rolloff_99_hz=rolloff,
        telephone_band_energy_percent=telephone_energy,
        high_band_energy_percent=high_energy,
        hum_50_60_percent=hum,
        reverb_proxy=reverb_proxy,
        bandwidth_limited=bandwidth_limited,
        noisy=snr < 20.0,
        clipped=clipping > 0.05,
        reverb_detected=reverb_proxy > 0.18,
    )


def _highpass(waveform: np.ndarray, cutoff: float, sample_rate: int) -> np.ndarray:
    rc = 1.0 / (2.0 * math.pi * cutoff)
    dt = 1.0 / sample_rate
    alpha = rc / (rc + dt)
    try:
        from scipy.signal import lfilter

        return lfilter([alpha, -alpha], [1.0, -alpha], waveform).astype(np.float32)
    except ImportError:
        pass
    output = np.empty_like(waveform)
    previous_output = 0.0
    previous_input = float(waveform[0])
    for index, value in enumerate(waveform):
        previous_output = alpha * (previous_output + float(value) - previous_input)
        output[index] = previous_output
        previous_input = float(value)
    return output


def _speech_eq(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    spectrum = np.fft.rfft(waveform)
    frequencies = np.fft.rfftfreq(waveform.size, 1.0 / sample_rate)
    gain = np.ones_like(frequencies)
    # A small, smooth presence lift only; never synthesize missing bandwidth.
    mask = (frequencies >= 1800) & (frequencies <= 3400)
    if np.any(mask):
        phase = (frequencies[mask] - 1800) / 1600 * math.pi
        gain[mask] = 10 ** ((1.5 * np.sin(phase)) / 20.0)
    return np.fft.irfft(spectrum * gain, n=waveform.size).astype(np.float32)


def _spectral_noise_reduction(waveform: np.ndarray, max_attenuation_db: float = 6.0) -> np.ndarray:
    try:
        from scipy.signal import istft, stft
    except ImportError:
        return waveform
    _, _, matrix = stft(waveform, fs=SAMPLE_RATE, nperseg=512, noverlap=384, boundary="zeros")
    magnitude = np.abs(matrix)
    noise = np.percentile(magnitude, 15, axis=1, keepdims=True)
    floor = 10 ** (-max_attenuation_db / 20.0)
    gain = np.maximum(floor, 1.0 - noise / np.maximum(magnitude, 1e-8))
    _, restored = istft(matrix * gain, fs=SAMPLE_RATE, nperseg=512, noverlap=384)
    if restored.size < waveform.size:
        restored = np.pad(restored, (0, waveform.size - restored.size))
    return restored[: waveform.size].astype(np.float32)


def _normalize_loudness(waveform: np.ndarray, target_lufs: float = -20.0) -> np.ndarray:
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(SAMPLE_RATE)
        loudness = float(meter.integrated_loudness(waveform))
        if math.isfinite(loudness):
            return pyln.normalize.loudness(waveform, loudness, target_lufs).astype(np.float32)
    except (ImportError, ValueError, OverflowError):
        pass
    rms = float(np.sqrt(np.mean(waveform * waveform) + 1e-12))
    target_rms = 10 ** (target_lufs / 20.0)
    return (waveform * min(10.0, target_rms / max(rms, 1e-8))).astype(np.float32)


def prepare_audio_variant(
    waveform: np.ndarray,
    diagnostics: AudioDiagnostics,
    mode: str = "original",
) -> tuple[np.ndarray, dict[str, bool | str]]:
    """Create a reversible ASR input variant without altering the source file."""
    original = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if mode not in {"original", "telephone", "auto"}:
        raise ValueError("audio mode must be original, telephone, or auto")
    if mode == "original":
        return original.copy(), {
            "variant": "original",
            "noise_reduction_applied": False,
            "dereverb_applied": False,
        }
    processed = _speech_eq(_highpass(original, 70.0, SAMPLE_RATE), SAMPLE_RATE)
    noise_applied = diagnostics.noisy and mode == "auto"
    if noise_applied:
        processed = _spectral_noise_reduction(processed)
    processed = _normalize_loudness(processed)
    peak_limit = 10 ** (-1.0 / 20.0)
    processed = np.clip(processed, -peak_limit, peak_limit).astype(np.float32)
    return processed, {
        "variant": "telephone-conservative",
        "noise_reduction_applied": noise_applied,
        # Blind dereverberation is intentionally not applied until the gold set
        # proves it helps; the diagnostic still exposes candidate recordings.
        "dereverb_applied": False,
    }


def write_pcm16_wav(path: str | Path, waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return destination
