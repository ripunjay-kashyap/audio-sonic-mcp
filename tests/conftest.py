"""
Shared pytest fixtures for the audio-sonic-mcp test suite.
All audio fixtures use synthetic sine-wave signals — no real downloads needed.
"""

import numpy as np
import soundfile as sf
import pytest
from pathlib import Path
from unittest.mock import MagicMock

SR = 44100  # standard sample rate used throughout the pipeline
DURATION = 4  # seconds — enough for librosa BPM/key/chroma analysis


@pytest.fixture
def synthetic_stereo_wav(tmp_path) -> Path:
    """
    440 Hz + 880 Hz stereo WAV, 44.1 kHz, 4 seconds.
    Written to tmp_path/raw_audio.wav (named as a raw download would be).
    """
    t = np.linspace(0, DURATION, SR * DURATION, endpoint=False)
    left = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    right = (np.sin(2 * np.pi * 880 * t) * 0.5).astype(np.float32)
    stereo = np.stack([left, right], axis=1)
    path = tmp_path / "raw_audio.wav"
    sf.write(str(path), stereo, SR)
    return path


@pytest.fixture
def audio_wav(tmp_path) -> Path:
    """
    Mixed stereo WAV at 44.1 kHz, 4 seconds.
    Contains bass (80 Hz), mid (440 Hz), and high (660 Hz) frequency content
    to exercise BPM, key, transient, and frequency extraction paths.
    """
    t = np.linspace(0, DURATION, SR * DURATION, endpoint=False)
    bass = np.sin(2 * np.pi * 80 * t) * 0.4
    mid = np.sin(2 * np.pi * 440 * t) * 0.3
    high = np.sin(2 * np.pi * 660 * t) * 0.2
    mono = (bass + mid + high).astype(np.float32)
    # Slight L/R difference so stereo width tests see non-mono signal
    left = mono
    right = (mono * 0.85).astype(np.float32)
    stereo = np.stack([left, right], axis=1)
    path = tmp_path / "input.wav"
    sf.write(str(path), stereo, SR)
    return path


def make_ydl_mock(meta: dict = None, error: Exception = None) -> MagicMock:
    """Build a yt_dlp.YoutubeDL context-manager mock for the yt-dlp-backed tests.

    extract_info returns `meta` (or {}); if `error` is given it raises instead.
    download() is a no-op. Stubs out all network access in ingestion/downloader tests.
    """
    mock_ydl = MagicMock()
    if error is not None:
        mock_ydl.extract_info.side_effect = error
    else:
        mock_ydl.extract_info.return_value = meta or {}
    mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ydl.__exit__ = MagicMock(return_value=False)
    return mock_ydl
