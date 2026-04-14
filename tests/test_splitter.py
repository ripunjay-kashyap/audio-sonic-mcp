"""Tests for pipeline/splitter.py — stem layout detection, verification, proxy SDR."""

import numpy as np
import soundfile as sf
import pytest

from pipeline.splitter import _find_stems_dir, _verify_stems, _compute_proxy_sdr


class TestFindStemsDir:
    def test_flat_layout_with_four_wavs(self, stems_dir):
        # stems_dir fixture already has ≥4 WAVs in root
        result = _find_stems_dir(stems_dir, "htdemucs")
        assert result == stems_dir

    def test_nested_layout_finds_vocals_parent(self, tmp_path):
        # Demucs nested output: <out>/<model>/<trackname>/vocals.wav
        nested = tmp_path / "htdemucs" / "my_track"
        nested.mkdir(parents=True)
        (nested / "vocals.wav").write_bytes(b"")

        result = _find_stems_dir(tmp_path, "htdemucs")
        assert result == nested

    def test_empty_dir_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _find_stems_dir(tmp_path, "htdemucs")

    def test_partial_wavs_falls_through_to_nested_search(self, tmp_path):
        # Only 2 WAVs at root → not flat; should look for nested vocals.wav
        (tmp_path / "one.wav").write_bytes(b"")
        (tmp_path / "two.wav").write_bytes(b"")
        nested = tmp_path / "htdemucs" / "track"
        nested.mkdir(parents=True)
        (nested / "vocals.wav").write_bytes(b"")

        result = _find_stems_dir(tmp_path, "htdemucs")
        assert result == nested


class TestVerifyStems:
    def test_all_four_stems_present(self, stems_dir):
        result = _verify_stems(stems_dir)
        assert set(result) == {"vocals.wav", "drums.wav", "bass.wav", "other.wav"}

    def test_missing_one_stem_raises(self, tmp_path):
        for name in ["vocals", "drums", "bass"]:  # 'other' missing
            (tmp_path / f"{name}.wav").write_bytes(b"")
        with pytest.raises(FileNotFoundError, match="Missing expected stems"):
            _verify_stems(tmp_path)

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _verify_stems(tmp_path)


class TestComputeProxySDR:
    def test_returns_float(self, tmp_path, stems_dir, synthetic_stereo_wav):
        stem_files = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
        sdr = _compute_proxy_sdr(synthetic_stereo_wav, stems_dir, stem_files)
        assert isinstance(sdr, float)

    def test_reasonable_range(self, tmp_path, stems_dir, synthetic_stereo_wav):
        stem_files = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
        sdr = _compute_proxy_sdr(synthetic_stereo_wav, stems_dir, stem_files)
        # SDR is in dB; any real separation should be between -20 and 40
        assert -20.0 <= sdr <= 40.0

    def test_fallback_on_bad_paths(self, tmp_path):
        # Missing files → exception caught internally → returns default 8.0
        sdr = _compute_proxy_sdr(
            tmp_path / "nonexistent.wav",
            tmp_path,
            ["vocals.wav"],
        )
        assert sdr == 8.0

    def test_perfect_reconstruction_yields_high_sdr(self, tmp_path):
        """If stems sum exactly to the original, SDR should be very high."""
        sr = 44100
        duration = 2
        t = np.linspace(0, duration, sr * duration, endpoint=False)
        signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        stereo = np.stack([signal, signal], axis=1)

        original = tmp_path / "original.wav"
        sf.write(str(original), stereo, sr)

        stems = tmp_path / "stems"
        stems.mkdir()
        # Split signal evenly across two stems so they sum back to original
        half = (signal * 0.5).reshape(-1, 1).repeat(2, axis=1)
        sf.write(str(stems / "vocals.wav"), half, sr)
        sf.write(str(stems / "drums.wav"), half, sr)
        sf.write(str(stems / "bass.wav"), np.zeros_like(half), sr)
        sf.write(str(stems / "other.wav"), np.zeros_like(half), sr)

        sdr = _compute_proxy_sdr(
            original, stems, ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
        )
        assert sdr > 15.0
