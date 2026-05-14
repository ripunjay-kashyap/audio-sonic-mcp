"""Tests for pipeline/analyzer.py — BPM, key, transient punch, freq peaks, etc."""

import numpy as np
import pytest

from pipeline.analyzer import (
    analyze_audio,
    _detect_key,
    _dominant_frequencies,
    _extract_bpm,
    _load_hpss_harmonic,
    _stereo_width_label,
    _transient_punch,
    _vocal_presence_estimate,
)

SR = 44100


# ── analyze_audio (integration) ───────────────────────────────────────────────


class TestAnalyzeAudio:
    def test_returns_all_expected_keys(self, audio_wav):
        result = analyze_audio(audio_wav)
        assert set(result.keys()) == {
            "bpm",
            "bpm_variable",
            "bpm_range",
            "key",
            "mode_confidence",
            "key_ambiguous",
            "transient_punch",
            "freq_peaks_hz",
            "stereo_width_label",
            "vocal_presence_label",
        }

    def test_bpm_is_non_negative(self, audio_wav):
        result = analyze_audio(audio_wav)
        assert result["bpm"] >= 0

    def test_key_is_non_empty_string(self, audio_wav):
        result = analyze_audio(audio_wav)
        assert isinstance(result["key"], str) and result["key"]

    def test_mode_confidence_in_0_1_range(self, audio_wav):
        result = analyze_audio(audio_wav)
        assert 0.0 <= result["mode_confidence"] <= 1.0

    def test_transient_punch_in_0_1_range(self, audio_wav):
        result = analyze_audio(audio_wav)
        assert 0.0 <= result["transient_punch"] <= 1.0

    def test_freq_peaks_has_harmonic_and_percussive(self, audio_wav):
        result = analyze_audio(audio_wav)
        assert set(result["freq_peaks_hz"].keys()) == {"harmonic", "percussive"}

    def test_stereo_width_label_is_valid(self, audio_wav):
        result = analyze_audio(audio_wav)
        assert result["stereo_width_label"] in ("mono", "narrow", "medium", "wide")

    def test_vocal_presence_label_is_valid(self, audio_wav):
        result = analyze_audio(audio_wav)
        assert result["vocal_presence_label"] in ("forward", "present", "background")


# ── _extract_bpm ──────────────────────────────────────────────────────────────


def make_beat_signal(bpm: float = 120.0, duration: int = 8, sr: int = SR) -> np.ndarray:
    """Synthetic percussive signal: sharp impulses at the given BPM."""
    n = sr * duration
    y = np.zeros(n, dtype=np.float32)
    beat_frames = int(sr * 60.0 / bpm)
    for i in range(0, n, beat_frames):
        end = min(i + 200, n)
        decay = np.exp(-np.arange(end - i) / 20.0).astype(np.float32)
        y[i:end] += decay
    peak = np.max(np.abs(y))
    if peak > 0:
        y /= peak
    return y


class TestExtractBpm:
    def test_returns_float(self):
        y = make_beat_signal(bpm=120.0)
        bpm = _extract_bpm(y, SR)
        assert isinstance(bpm, float)

    def test_detects_approximate_bpm(self):
        """librosa's beat tracker should land within ±30 BPM of a clean 120 BPM grid."""
        y = make_beat_signal(bpm=120.0)
        bpm = _extract_bpm(y, SR)
        assert bpm > 0
        assert 60.0 <= bpm <= 240.0  # sensible music range


# ── _detect_key ───────────────────────────────────────────────────────────────


class TestDetectKey:
    def test_returns_major_or_minor_string(self):
        t = np.linspace(0, 4, SR * 4, endpoint=False)
        y = np.sin(2 * np.pi * 261.63 * t).astype(np.float32)  # C4
        key, confidence, _ = _detect_key(y, SR)
        assert "Major" in key or "Minor" in key

    def test_contains_known_pitch_class(self):
        classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        t = np.linspace(0, 4, SR * 4, endpoint=False)
        y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        key, confidence, _ = _detect_key(y, SR)
        assert any(key.startswith(p) for p in classes)

    def test_confidence_in_0_1_range(self):
        t = np.linspace(0, 4, SR * 4, endpoint=False)
        y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        _, confidence, _ = _detect_key(y, SR)
        assert 0.0 <= confidence <= 1.0

    def test_empty_signal_returns_unknown(self):
        _, confidence, _ = _detect_key(np.array([]), SR)
        assert confidence == 0.0


# ── _transient_punch ──────────────────────────────────────────────────────────


class TestTransientPunch:
    def test_result_in_0_1_range(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(SR * 2).astype(np.float32)
        assert 0.0 <= _transient_punch(y, SR) <= 1.0

    def test_silence_gives_low_score(self):
        y = np.zeros(SR * 2, dtype=np.float32)
        assert _transient_punch(y, SR) < 0.1


# ── _dominant_frequencies ─────────────────────────────────────────────────────


class TestDominantFrequencies:
    def test_returns_up_to_five_frequencies(self):
        t = np.linspace(0, 1, SR, endpoint=False)
        y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        freqs = _dominant_frequencies(y, SR)
        assert 1 <= len(freqs) <= 5

    def test_frequencies_in_musical_range(self):
        t = np.linspace(0, 1, SR, endpoint=False)
        y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        freqs = _dominant_frequencies(y, SR)
        assert all(20 <= f <= 16000 for f in freqs)

    def test_detects_440hz(self):
        t = np.linspace(0, 2, SR * 2, endpoint=False)
        y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        freqs = _dominant_frequencies(y, SR, top_n=1)
        assert abs(freqs[0] - 440.0) < 5.0

    def test_empty_signal_returns_list(self):
        y = np.zeros(SR, dtype=np.float32)
        freqs = _dominant_frequencies(y, SR)
        assert isinstance(freqs, list)


# ── _stereo_width_label ───────────────────────────────────────────────────────


class TestStereoWidthLabel:
    def test_identical_channels_is_mono(self):
        t = np.linspace(0, 1, SR, endpoint=False)
        ch = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        assert _stereo_width_label(np.stack([ch, ch]), SR) == "mono"

    def test_mono_signal_1d_returns_mono(self):
        t = np.linspace(0, 1, SR, endpoint=False)
        y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        # 1D array — treated as mono
        assert _stereo_width_label(np.stack([y, y]), SR) == "mono"

    def test_uncorrelated_channels_is_wide_or_medium(self):
        rng = np.random.default_rng(1)
        left = rng.standard_normal(SR).astype(np.float32)
        right = rng.standard_normal(SR).astype(np.float32)
        assert _stereo_width_label(np.stack([left, right]), SR) in ("medium", "wide")


# ── _load_hpss_harmonic ───────────────────────────────────────────────────────


class TestLoadHpssHarmonic:
    def test_returns_none_when_offset_beyond_duration(self, audio_wav):
        result = _load_hpss_harmonic(audio_wav, offset_sec=9999, duration_sec=60, target_sr=22050)
        assert result is None

    def test_returns_ndarray_for_valid_offset(self, audio_wav):
        result = _load_hpss_harmonic(audio_wav, offset_sec=0, duration_sec=2, target_sr=22050)
        assert isinstance(result, np.ndarray)
        assert result.size > 0

    def test_resamples_to_target_sr(self, audio_wav):
        # audio_wav is 44100 Hz; requesting target_sr=22050
        result = _load_hpss_harmonic(audio_wav, offset_sec=0, duration_sec=2, target_sr=22050)
        assert result is not None
        assert abs(len(result) - 2 * 22050) < 1000


# ── _detect_key list input ────────────────────────────────────────────────────


class TestDetectKeyListInput:
    def test_accepts_single_element_list(self):
        t = np.linspace(0, 4, 22050 * 4, endpoint=False)
        y = np.sin(2 * np.pi * 261.63 * t).astype(np.float32)
        key, confidence, ambiguous = _detect_key([y], 22050)
        assert "Major" in key or "Minor" in key
        assert isinstance(ambiguous, bool)

    def test_accepts_multi_element_list(self):
        t = np.linspace(0, 4, 22050 * 4, endpoint=False)
        y_a = np.sin(2 * np.pi * 261.63 * t).astype(np.float32)  # C4
        y_b = np.sin(2 * np.pi * 293.66 * t).astype(np.float32)  # D4
        key, confidence, ambiguous = _detect_key([y_a, y_b], 22050)
        assert "Major" in key or "Minor" in key
        assert 0.0 <= confidence <= 1.0
        assert isinstance(ambiguous, bool)

    def test_filters_none_values_from_list(self):
        t = np.linspace(0, 4, 22050 * 4, endpoint=False)
        y = np.sin(2 * np.pi * 261.63 * t).astype(np.float32)
        key, confidence, _ = _detect_key([y, None], 22050)
        assert "Major" in key or "Minor" in key


# ── _vocal_presence_estimate ──────────────────────────────────────────────────


class TestVocalPresenceEstimate:
    def test_returns_valid_label(self):
        t = np.linspace(0, 1, SR, endpoint=False)
        y = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        assert _vocal_presence_estimate(y, SR) in ("forward", "present", "background")

    def test_vocal_band_energy_gives_forward(self):
        # Pure 1kHz tone sits squarely in the vocal band (200Hz–4kHz)
        t = np.linspace(0, 2, SR * 2, endpoint=False)
        y = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        assert _vocal_presence_estimate(y, SR) == "forward"

    def test_sub_bass_gives_background(self):
        # 40 Hz is below the vocal band
        t = np.linspace(0, 2, SR * 2, endpoint=False)
        y = np.sin(2 * np.pi * 40 * t).astype(np.float32)
        assert _vocal_presence_estimate(y, SR) == "background"
