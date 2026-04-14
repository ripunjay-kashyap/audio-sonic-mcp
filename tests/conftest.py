"""
Shared pytest fixtures for the audio-stem-mcp test suite.
All audio fixtures use synthetic sine-wave signals — no real downloads needed.
"""

import numpy as np
import soundfile as sf
import pytest
from pathlib import Path

SR = 44100  # standard sample rate used throughout the pipeline
DURATION = 4  # seconds — enough for librosa BPM/key/chroma analysis


@pytest.fixture
def synthetic_stereo_wav(tmp_path) -> Path:
    """
    440 Hz + 880 Hz stereo WAV, 44.1 kHz, 4 seconds.
    Written to tmp_path/raw_audio.m4a (named as a raw download would be).
    """
    t = np.linspace(0, DURATION, SR * DURATION, endpoint=False)
    left = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    right = (np.sin(2 * np.pi * 880 * t) * 0.5).astype(np.float32)
    stereo = np.stack([left, right], axis=1)
    # Use .wav so soundfile can write it; converter tests mock ffmpeg anyway
    path = tmp_path / "raw_audio.wav"
    sf.write(str(path), stereo, SR)
    return path


@pytest.fixture
def stems_dir(tmp_path) -> Path:
    """
    Fake stems directory containing 4 stereo WAV files at 44.1 kHz.
    Each stem has a distinct fundamental frequency.
    """
    directory = tmp_path / "stems"
    directory.mkdir()

    t = np.linspace(0, DURATION, SR * DURATION, endpoint=False)
    freqs = {"vocals": 440, "drums": 100, "bass": 80, "other": 660}

    for name, freq in freqs.items():
        signal = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
        stereo = np.stack([signal, signal * 0.9], axis=1)  # slight L/R difference
        sf.write(str(directory / f"{name}.wav"), stereo, SR)

    return directory
