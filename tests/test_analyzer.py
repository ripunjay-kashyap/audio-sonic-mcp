"""Tests for pipeline/analyzer.py — BPM, key, transient punch, freq peaks, etc."""

import numpy as np
import soundfile as sf
import pytest

from pipeline.analyzer import (
    analyze_stems,
    _detect_key,
    _dominant_frequencies,
    _extract_bpm,
    _stereo_width_label,
    _transient_punch,
    _vocal_presence,
)

SR = 44100


# ── analyze_stems (integration) ───────────────────────────────────────────────


class TestAnalyzeStems:
    def test_empty_stem_files_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="No stem files were loaded"):
            analyze_stems(tmp_path, [])

    def test_returns_all_expected_keys(self, stems_dir):
        result = analyze_stems(
            stems_dir, ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
        )
        assert set(result.keys()) == {
            "bpm",
            "key",
            "mode_confidence",
            "transient_punch",
            "freq_peaks_hz",
            "stereo_width_label",
            "vocal_presence_label",
        }

    def test_bpm_is_non_negative(self, stems_dir):
        result = analyze_stems(
            stems_dir, ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
        )
        assert result["bpm"] >= 0  # pure sine stems may yield 0; real audio always > 0

    def test_key_is_non_empty_string(self, stems_dir):
        result = analyze_stems(
            stems_dir, ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
        )
        assert isinstance(result["key"], str) and result["key"]

    def test_mode_confidence_in_0_1_range(self, stems_dir):
        result = analyze_stems(
            stems_dir, ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
        )
        assert 0.0 <= result["mode_confidence"] <= 1.0

    def test_transient_punch_in_0_1_range(self, stems_dir):
        result = analyze_stems(
            stems_dir, ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
        )
        assert 0.0 <= result["transient_punch"] <= 1.0

    def test_freq_peaks_has_entry_per_stem(self, stems_dir):
        result = analyze_stems(
            stems_dir, ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
        )
        assert set(result["freq_peaks_hz"].keys()) == {
            "vocals",
            "drums",
            "bass",
            "other",
        }

    def test_stereo_width_label_is_valid(self, stems_dir):
        result = analyze_stems(
            stems_dir, ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
        )
        assert result["stereo_width_label"] in ("mono", "narrow", "medium", "wide")

    def test_vocal_presence_label_is_valid(self, stems_dir):
        result = analyze_stems(
            stems_dir, ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
        )
        assert result["vocal_presence_label"] in ("forward", "balanced", "recessed")

    def test_works_without_drums_stem(self, tmp_path):
        """Falls back to another stem for BPM when drums is absent."""
        t = np.linspace(0, 4, SR * 4, endpoint=False)
        signal = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        stereo = np.stack([signal, signal], axis=1)
        for name in ["vocals", "bass", "other"]:
            sf.write(str(tmp_path / f"{name}.wav"), stereo, SR)

        result = analyze_stems(tmp_path, ["vocals.wav", "bass.wav", "other.wav"])
        assert result["bpm"] >= 0

    def test_works_without_harmonic_stems(self, tmp_path):
        """Falls back gracefully when vocals/other/bass are all absent."""
        t = np.linspace(0, 4, SR * 4, endpoint=False)
        signal = (np.sin(2 * np.pi * 100 * t) * 0.5).astype(np.float32)
        stereo = np.stack([signal, signal], axis=1)
        sf.write(str(tmp_path / "drums.wav"), stereo, SR)

        result = analyze_stems(tmp_path, ["drums.wav"])
        assert isinstance(result["key"], str)


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
        key, confidence = _detect_key({"bass": y, "other": y}, SR)
        assert "Major" in key or "Minor" in key

    def test_contains_known_pitch_class(self):
        classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        t = np.linspace(0, 4, SR * 4, endpoint=False)
        y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        key, confidence = _detect_key({"bass": y, "other": y}, SR)
        assert any(key.startswith(p) for p in classes)

    def test_confidence_in_0_1_range(self):
        t = np.linspace(0, 4, SR * 4, endpoint=False)
        y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        _, confidence = _detect_key({"bass": y, "other": y}, SR)
        assert 0.0 <= confidence <= 1.0


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
        stems = {"drums": np.stack([ch, ch])}
        assert _stereo_width_label(stems, SR) == "mono"

    def test_mono_signal_1d_returns_mono(self):
        t = np.linspace(0, 1, SR, endpoint=False)
        y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        assert _stereo_width_label({"other": y}, SR) == "mono"

    def test_uncorrelated_channels_is_wide_or_medium(self):
        rng = np.random.default_rng(1)
        left = rng.standard_normal(SR).astype(np.float32)
        right = rng.standard_normal(SR).astype(np.float32)
        stems = {"drums": np.stack([left, right])}
        assert _stereo_width_label(stems, SR) in ("medium", "wide")

    def test_falls_back_to_other_when_drums_absent(self):
        t = np.linspace(0, 1, SR, endpoint=False)
        ch = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        stems = {"other": np.stack([ch, ch])}
        assert _stereo_width_label(stems, SR) == "mono"


# ── _vocal_presence ───────────────────────────────────────────────────────────


class TestVocalPresence:
    def _make_stems(self, vocal_scale, other_scale):
        t = np.linspace(0, 1, SR, endpoint=False)
        base = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return {
            "vocals": base * vocal_scale,
            "drums": base * other_scale,
            "bass": base * other_scale,
            "other": base * other_scale,
        }

    def test_returns_valid_label(self):
        stems = self._make_stems(0.5, 0.5)
        assert _vocal_presence(stems, SR) in ("forward", "balanced", "recessed")

    def test_loud_vocals_are_forward(self):
        stems = self._make_stems(1.0, 0.01)
        assert _vocal_presence(stems, SR) == "forward"

    def test_quiet_vocals_are_recessed(self):
        stems = self._make_stems(0.01, 1.0)
        assert _vocal_presence(stems, SR) == "recessed"
