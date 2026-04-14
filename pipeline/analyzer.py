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

    Memory strategy: stems are loaded one at a time. Only small intermediate
    results (12-element chroma arrays, RMS scalars, feature dicts) are kept
    across iterations — full-res arrays are freed immediately after extraction.
    """
    import librosa

    if not stem_files:
        raise ValueError(
            "No stem files were loaded — stem_files list was empty or all loads failed."
        )

    logger.info(
        "analyze_stems: thread started — loading %d stems from %s",
        len(stem_files),
        stems_dir,
    )

    TARGET_SR = 22050  # downsample target for BPM, punch, and key detection

    # Accumulated lightweight results
    bpm: float | None = None
    transient_punch: float = 0.5
    freq_peaks: dict[str, list[float]] = {}
    stereo_width: str | None = None  # set once from drums or other
    vocal_rms: float = 0.0
    rms_sum: float = 0.0
    stem_count: int = 0
    # 22050 Hz mono arrays for key detection — ~17 MB each vs. ~70 MB full-res stereo
    key_stems: dict[str, np.ndarray] = {}
    sr_common: int = 44100

    for fname in stem_files:
        name = fname.replace(".wav", "")
        path = stems_dir / fname

        # Load full-res stereo (needed for stereo width and accurate freq peaks)
        y, sr = librosa.load(str(path), sr=None, mono=False)

        y_mono = librosa.to_mono(y) if y.ndim > 1 else y

        # ── Stereo width — capture from the first stem we see ────────────────
        if stereo_width is None:
            stereo_width = _stereo_width_label({name: y}, sr)

        # ── Frequency peaks (uses full-res mono) ──────────────────────────────
        freq_peaks[name] = _dominant_frequencies(y_mono, sr)

        # ── RMS for vocal presence (scalar — tiny) ────────────────────────────
        rms_val = float(np.sqrt(np.mean(y_mono**2)))
        rms_sum += rms_val
        stem_count += 1
        if name == "vocals":
            vocal_rms = rms_val

        # ── Downsample to TARGET_SR for BPM, punch, and key detection ─────────
        if sr != TARGET_SR:
            y_low = librosa.resample(y_mono, orig_sr=sr, target_sr=TARGET_SR)
        else:
            y_low = y_mono  # already a view / same array; no extra copy

        # ── BPM — prefer drums; key_stems used as fallback after loop ──────────
        if name == "drums":
            bpm = _extract_bpm(y_low, TARGET_SR)

        # ── Transient punch — drums only ──────────────────────────────────────
        if name == "drums":
            transient_punch = _transient_punch(y_low, TARGET_SR)

        # ── Store downsampled mono for key detection (bass, other, vocals) ────
        if name in ("bass", "other", "vocals"):
            key_stems[name] = y_low  # ~17 MB, kept until _detect_key runs below

        # Free the full-res stereo and mono arrays — y_low stays only if needed
        del y, y_mono
        if name not in ("bass", "other", "vocals"):
            # y_low is only referenced by key_stems for harmonic stems; for
            # drums/unknown it's safe to drop now
            del y_low

    # ── BPM fallback: load first non-drums stem at reduced SR ─────────────────
    if bpm is None:
        if key_stems:
            first_key_stem = next(iter(key_stems.values()))
            bpm = _extract_bpm(first_key_stem, TARGET_SR)
        else:
            bpm = 0.0  # edge case: only drums, and drums failed to load

    # ── Key detection from downsampled stems (low memory) ────────────────────
    key_str, mode_confidence = _detect_key(key_stems, TARGET_SR)

    # Free key stems now that detection is done
    key_stems.clear()

    # ── Vocal presence from accumulated RMS scalars ────────────────────────────
    avg_rms = rms_sum / max(stem_count, 1)
    ratio = vocal_rms / (avg_rms + 1e-6)
    if ratio > 0.55:
        vocal_presence = "forward"
    elif ratio > 0.35:
        vocal_presence = "balanced"
    else:
        vocal_presence = "recessed"

    if stereo_width is None:
        stereo_width = "mono"

    return {
        "bpm": round(bpm, 2),
        "key": key_str,
        "mode_confidence": mode_confidence,
        "transient_punch": round(transient_punch, 3),
        "freq_peaks_hz": freq_peaks,
        "stereo_width_label": stereo_width,
        "vocal_presence_label": vocal_presence,
    }


# ── Sub-extractors ────────────────────────────────────────────────────────────


def _extract_bpm(y: np.ndarray, sr: int) -> float:
    import librosa

    # Nudge autocorrelation to look for higher tempos first (Rap/Trap focus)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120)

    if hasattr(tempo, "__len__"):
        tempo = float(tempo[0])

    # BPM Resolution Logic: If tempo is sub-100, check for High-Speed Trap "Performance" tempo
    # We prefer the 140-180 range for these genres.
    if tempo < 100:
        if 140 <= tempo * 4 <= 190:
            tempo *= 4
        elif 140 <= tempo * 2 <= 190:
            tempo *= 2
        elif 80 <= tempo * 2 <= 140:
            tempo *= 2

    return float(tempo)


PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_TEMPLATE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_TEMPLATE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


def _detect_key(stems: dict[str, np.ndarray], sr: int) -> str:
    import librosa

    target_sr = 22050

    def get_chroma_mean(y_orig):
        if y_orig is None or y_orig.size <= 1:
            return np.zeros(12)

        y_mono = librosa.to_mono(y_orig) if y_orig.ndim > 1 else y_orig

        # Downsample for CQT speed
        if sr != target_sr:
            y_low = librosa.resample(y_mono, orig_sr=sr, target_sr=target_sr)
        else:
            y_low = y_mono

        # 1. Spectral Flattening (HPSS) to remove percussive/transient noise leakage
        # This helps find the fundamental against tonal bloom
        y_harmonic = librosa.effects.harmonic(y_low, margin=3.0)

        # 2. Extract Chroma CQT
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=target_sr, hop_length=2048)

        # 2. Log-Amplitude Scaling (Roadmap: To find quiet minor/major thirds)
        chroma_db = librosa.amplitude_to_db(chroma, ref=np.max)

        # 3. Mean across time
        return chroma_db.mean(axis=1)

    # ── Weighted Fusion ──────────────────────────────────────────────────────
    # Roadmap: 60% Bass (Root consistency) / 40% Other (Harmonic mode)
    bass_chroma = get_chroma_mean(stems.get("bass"))
    other_chroma = get_chroma_mean(stems.get("other"))

    # If stems are empty, fallback to vocals or just zeros
    if np.max(np.abs(bass_chroma)) == 0 and np.max(np.abs(other_chroma)) == 0:
        combined_chroma = get_chroma_mean(stems.get("vocals"))
    else:
        combined_chroma = (0.6 * bass_chroma) + (0.4 * other_chroma)

    # Re-normalize for correlation matching
    if np.max(np.abs(combined_chroma)) > 0:
        combined_chroma = (combined_chroma - combined_chroma.mean()) / (
            combined_chroma.std() + 1e-6
        )

    best_score = -np.inf
    best_key = "Unknown"
    for i, root in enumerate(PITCH_CLASSES):
        rotated = np.roll(combined_chroma, -i)
        maj = np.corrcoef(rotated, MAJOR_TEMPLATE)[0, 1]
        min_ = np.corrcoef(rotated, MINOR_TEMPLATE)[0, 1]
        if maj > best_score:
            best_score, best_key = maj, f"{root} Major"
        if min_ > best_score:
            best_score, best_key = min_, f"{root} Minor"

    confidence = round(float(np.clip(best_score, 0.0, 1.0)), 2)
    return best_key, confidence


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
    _fallback_key = next(
        (k for k in ("drums", "other") if k in stems), next(iter(stems))
    )
    y = stems[_fallback_key]
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
        return float(np.sqrt(np.mean(m**2)))

    vocal_rms = rms(stems.get("vocals", np.zeros(1)))
    all_rms = sum(rms(v) for v in stems.values()) / max(len(stems), 1)

    ratio = vocal_rms / (all_rms + 1e-6)
    if ratio > 0.55:
        return "forward"
    elif ratio > 0.35:
        return "balanced"
    else:
        return "recessed"
