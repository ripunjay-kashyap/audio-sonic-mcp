"""
Stage 5 — Feature Extraction
Uses librosa to extract BPM, musical key, transient punch, frequency peaks,
stereo width, and vocal presence from isolated stems.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def analyze_stems(stems_dir: Path, stem_files: list[str]) -> dict[str, Any]:
    """
    Runs librosa analysis on isolated stems to extract:
    - BPM (from drums stem — most accurate)
    - Musical key (from 'other' + 'vocals' for harmonic content)
    - Transient punch (drum stem)
    - Dominant frequency peaks (per stem)
    - Stereo width estimate
    - Vocal presence (RMS ratio of vocals vs. mix)
    """
    import librosa
    import soundfile as sf

    stems: dict[str, np.ndarray] = {}
    sr_common: int = 44100

    for fname in stem_files:
        name = fname.replace(".wav", "")
        path = stems_dir / fname
        y, sr = librosa.load(str(path), sr=None, mono=False)
        sr_common = sr
        # Keep stereo for width analysis, mono for most features
        stems[name] = y

    def to_mono(y: np.ndarray) -> np.ndarray:
        return librosa.to_mono(y) if y.ndim > 1 else y

    # ── BPM (drums stem is cleanest signal) ───────────────────────────────────
    bpm = _extract_bpm(to_mono(stems.get("drums", next(iter(stems.values())))), sr_common)

    # ── Key (harmonic stems: vocals + other) ──────────────────────────────────
    harmonic_stems = ["vocals", "other", "bass"]
    harmonic_y = np.zeros(max(to_mono(stems[k]).shape[0] for k in harmonic_stems if k in stems))
    for k in harmonic_stems:
        if k in stems:
            m = to_mono(stems[k])
            harmonic_y[: len(m)] += m
    key_str = _detect_key(harmonic_y, sr_common)

    # ── Drum transient punch ──────────────────────────────────────────────────
    transient_punch = _transient_punch(
        to_mono(stems["drums"]), sr_common
    ) if "drums" in stems else 0.5

    # ── Frequency peaks per stem ──────────────────────────────────────────────
    freq_peaks: dict[str, list[float]] = {}
    for name, y in stems.items():
        freq_peaks[name] = _dominant_frequencies(to_mono(y), sr_common)

    # ── Stereo width (left–right correlation) ─────────────────────────────────
    stereo_width = _stereo_width_label(stems, sr_common)

    # ── Vocal presence ────────────────────────────────────────────────────────
    vocal_presence = _vocal_presence(stems, sr_common)

    return {
        "bpm": round(bpm, 2),
        "key": key_str,
        "transient_punch": round(transient_punch, 3),
        "freq_peaks_hz": freq_peaks,
        "stereo_width_label": stereo_width,
        "vocal_presence_label": vocal_presence,
    }


# ── Sub-extractors ────────────────────────────────────────────────────────────

def _extract_bpm(y: np.ndarray, sr: int) -> float:
    import librosa
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # librosa may return array
    if hasattr(tempo, "__len__"):
        tempo = float(tempo[0])
    return float(tempo)


PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_TEMPLATE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_TEMPLATE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _detect_key(y: np.ndarray, sr: int) -> str:
    import librosa

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    best_score = -np.inf
    best_key = "Unknown"
    for i, root in enumerate(PITCH_CLASSES):
        rotated = np.roll(chroma_mean, -i)
        maj = np.corrcoef(rotated, MAJOR_TEMPLATE)[0, 1]
        min_ = np.corrcoef(rotated, MINOR_TEMPLATE)[0, 1]
        if maj > best_score:
            best_score, best_key = maj, f"{root} Major"
        if min_ > best_score:
            best_score, best_key = min_, f"{root} Minor"

    return best_key


def _transient_punch(y: np.ndarray, sr: int) -> float:
    """
    Returns a 0–1 score of percussive sharpness using onset strength.
    """
    import librosa

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peak = float(np.percentile(onset_env, 97))
    mean = float(onset_env.mean()) + 1e-6
    # Normalize to 0–1 range (empirically calibrated)
    score = np.clip((peak / mean - 1) / 20.0, 0, 1)
    return float(score)


def _dominant_frequencies(y: np.ndarray, sr: int, top_n: int = 5) -> list[float]:
    """Returns the top N dominant frequency bins (Hz) from the magnitude spectrum."""
    fft = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)

    # Focus on musically relevant range: 20Hz–16kHz
    mask = (freqs >= 20) & (freqs <= 16000)
    fft_masked = fft[mask]
    freqs_masked = freqs[mask]

    if len(fft_masked) == 0:
        return []

    top_idx = np.argsort(fft_masked)[-top_n:][::-1]
    return [round(float(freqs_masked[i]), 1) for i in top_idx]


def _stereo_width_label(stems: dict, sr: int) -> str:
    """Estimates stereo width via L–R correlation of the drum stem."""
    y = stems.get("drums") or stems.get("other") or next(iter(stems.values()))
    if y.ndim < 2 or y.shape[0] < 2:
        return "mono"

    left, right = y[0], y[1]
    min_len = min(len(left), len(right))
    corr = float(np.corrcoef(left[:min_len], right[:min_len])[0, 1])

    if corr > 0.95:
        return "mono"
    elif corr > 0.70:
        return "narrow"
    elif corr > 0.40:
        return "medium"
    else:
        return "wide"


def _vocal_presence(stems: dict, sr: int) -> str:
    """
    Compares vocal RMS to mix RMS to classify presence.
    """
    import librosa

    def rms(y):
        m = librosa.to_mono(y) if y.ndim > 1 else y
        return float(np.sqrt(np.mean(m ** 2)))

    vocal_rms = rms(stems.get("vocals", np.zeros(1)))
    all_rms = sum(rms(v) for v in stems.values()) / max(len(stems), 1)

    ratio = vocal_rms / (all_rms + 1e-6)
    if ratio > 0.55:
        return "forward"
    elif ratio > 0.35:
        return "balanced"
    else:
        return "recessed"
