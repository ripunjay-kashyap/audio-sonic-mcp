"""Tests for pipeline/vectorizer.py — librosa fallback vector and mix loading."""

import numpy as np
from unittest.mock import patch

from pipeline.vectorizer import (
    VECTOR_DIM,
    _librosa_fallback_vector,
    _load_mix,
    generate_vibe_vector,
)


# ── _load_mix ─────────────────────────────────────────────────────────────────


class TestLoadMix:
    def test_returns_1d_numpy_array(self, stems_dir):
        mix = _load_mix(stems_dir, ["vocals.wav", "drums.wav"], sr=22050)
        assert isinstance(mix, np.ndarray)
        assert mix.ndim == 1

    def test_empty_file_list_returns_silence(self, tmp_path):
        mix = _load_mix(tmp_path, [], sr=22050)
        assert len(mix) == 22050 * 10
        assert np.all(mix == 0.0)

    def test_peak_normalized_to_1(self, stems_dir):
        mix = _load_mix(stems_dir, ["vocals.wav", "drums.wav"], sr=22050)
        assert np.max(np.abs(mix)) <= 1.0 + 1e-6

    def test_all_four_stems_mixed(self, stems_dir):
        stem_files = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
        mix_all = _load_mix(stems_dir, stem_files, sr=22050)
        mix_one = _load_mix(stems_dir, ["vocals.wav"], sr=22050)
        # Mix of 4 stems should be a different signal than just one stem
        assert not np.allclose(mix_all, mix_one)


# ── _librosa_fallback_vector ──────────────────────────────────────────────────


class TestLibrosaFallbackVector:
    def test_returns_exactly_512_dims(self, stems_dir):
        vec = _librosa_fallback_vector(
            stems_dir, ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
        )
        assert len(vec) == VECTOR_DIM

    def test_all_values_are_floats(self, stems_dir):
        vec = _librosa_fallback_vector(
            stems_dir, ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
        )
        assert all(isinstance(v, float) for v in vec)

    def test_l2_norm_is_1(self, stems_dir):
        vec = _librosa_fallback_vector(
            stems_dir, ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
        )
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-4

    def test_different_audio_produces_different_vectors(self, stems_dir, tmp_path):
        import soundfile as sf

        t = np.linspace(0, 4, 44100 * 4, endpoint=False)
        noise = np.random.default_rng(42).standard_normal(len(t)).astype(np.float32)
        sf.write(str(tmp_path / "vocals.wav"), noise, 44100)

        vec_a = _librosa_fallback_vector(stems_dir, ["vocals.wav"])
        vec_b = _librosa_fallback_vector(tmp_path, ["vocals.wav"])
        assert not np.allclose(vec_a, vec_b, atol=1e-3)


# ── generate_vibe_vector ──────────────────────────────────────────────────────


class TestGenerateVibeVector:
    def test_falls_back_to_librosa_when_clap_unavailable(self, stems_dir):
        stem_files = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
        with patch(
            "pipeline.vectorizer._clap_vector",
            side_effect=ImportError("no transformers"),
        ):
            vec = generate_vibe_vector(stems_dir, stem_files)
        assert len(vec) == VECTOR_DIM

    def test_falls_back_on_any_clap_exception(self, stems_dir):
        stem_files = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
        with patch(
            "pipeline.vectorizer._clap_vector", side_effect=RuntimeError("GPU OOM")
        ):
            vec = generate_vibe_vector(stems_dir, stem_files)
        assert len(vec) == VECTOR_DIM
