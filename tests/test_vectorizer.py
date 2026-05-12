"""Tests for pipeline/vectorizer.py — librosa fallback vector and audio loading."""

import numpy as np
from unittest.mock import patch

from pipeline.vectorizer import (
    VECTOR_DIM,
    _librosa_fallback_vector,
    _load_audio,
    generate_vibe_vector,
)


# ── _load_audio ───────────────────────────────────────────────────────────────


class TestLoadAudio:
    def test_returns_1d_numpy_array(self, audio_wav):
        y = _load_audio(audio_wav, sr=22050)
        assert isinstance(y, np.ndarray)
        assert y.ndim == 1

    def test_peak_normalized_to_1(self, audio_wav):
        y = _load_audio(audio_wav, sr=22050)
        assert np.max(np.abs(y)) <= 1.0 + 1e-6

    def test_resamples_to_requested_sr(self, audio_wav):
        y_22k = _load_audio(audio_wav, sr=22050)
        y_48k = _load_audio(audio_wav, sr=48000)
        # Higher SR → more samples for same duration
        assert len(y_48k) > len(y_22k)


# ── _librosa_fallback_vector ──────────────────────────────────────────────────


class TestLibrosaFallbackVector:
    def test_returns_exactly_512_dims(self, audio_wav):
        vec = _librosa_fallback_vector(audio_wav)
        assert len(vec) == VECTOR_DIM

    def test_all_values_are_floats(self, audio_wav):
        vec = _librosa_fallback_vector(audio_wav)
        assert all(isinstance(v, float) for v in vec)

    def test_l2_norm_is_1(self, audio_wav):
        vec = _librosa_fallback_vector(audio_wav)
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-4

    def test_different_audio_produces_different_vectors(self, audio_wav, tmp_path):
        import soundfile as sf

        t = np.linspace(0, 4, 44100 * 4, endpoint=False)
        noise = np.random.default_rng(42).standard_normal(len(t)).astype(np.float32)
        noise_path = tmp_path / "noise.wav"
        sf.write(str(noise_path), noise, 44100)

        vec_a = _librosa_fallback_vector(audio_wav)
        vec_b = _librosa_fallback_vector(noise_path)
        assert not np.allclose(vec_a, vec_b, atol=1e-3)


# ── generate_vibe_vector ──────────────────────────────────────────────────────


class TestGenerateVibeVector:
    def test_falls_back_to_librosa_when_clap_unavailable(self, audio_wav):
        with patch(
            "pipeline.vectorizer._clap_vector",
            side_effect=ImportError("no transformers"),
        ):
            vec = generate_vibe_vector(audio_wav)
        assert len(vec) == VECTOR_DIM

    def test_falls_back_on_any_clap_exception(self, audio_wav):
        with patch(
            "pipeline.vectorizer._clap_vector", side_effect=RuntimeError("GPU OOM")
        ):
            vec = generate_vibe_vector(audio_wav)
        assert len(vec) == VECTOR_DIM
