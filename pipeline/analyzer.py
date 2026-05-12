"""
Stage 4 — Audio Analysis
Loads the mixed WAV, applies HPSS (Harmonic-Percussive Source Separation)
in memory, then extracts BPM, key, transient punch, frequency peaks,
stereo width, and a vocal-presence estimate from the two signals.
No files are written — all separation is in-memory.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def analyze_audio(wav_path: Path, stems_dir: "Path | None" = None) -> dict[str, Any]:
    """
    Runs audio feature extraction on a single WAV file.

    When stems_dir is provided (Demucs output), uses isolated stems for each task:
      - BPM + transient punch  ← drums stem
      - Key detection          ← bass + other stems mixed
      - Vocal presence         ← vocals RMS vs full-mix RMS

    Falls back to in-memory HPSS when stems_dir is None or a stem is missing.

    Returns keys:
        bpm, key, mode_confidence, transient_punch, freq_peaks_hz,
        stereo_width_label, vocal_presence_label
    """
    import librosa
    import soundfile as sf

    logger.info("analyze_audio: loading %s", wav_path)

    from pipeline.window import pick_window

    TARGET_SR = 22050
    ANALYSIS_DURATION = 60.0  # also used by _load_stem for the cropped stems

    # Read directly via soundfile to avoid librosa's audioread fallback,
    # which spawns FFmpeg with inherited stdin and deadlocks under MCP.
    with sf.SoundFile(str(wav_path)) as snd:
        sr_orig = snd.samplerate
        offset_frames, frames_to_read = pick_window(snd.frames, sr_orig)
        snd.seek(offset_frames)
        raw = snd.read(frames=frames_to_read, dtype="float32", always_2d=True)
    y_stereo = raw.T
    if y_stereo.ndim == 1:
        y_stereo = np.stack([y_stereo, y_stereo])

    y_mono = librosa.to_mono(y_stereo)
    if sr_orig != TARGET_SR:
        y_mono = librosa.resample(y_mono, orig_sr=sr_orig, target_sr=TARGET_SR)

    # Try stem-based analysis; fall back to HPSS on any missing stem
    y_drums = y_bass = y_bass_other = y_vocals = None
    if stems_dir is not None:
        y_drums = _load_stem(stems_dir, "drums", TARGET_SR, ANALYSIS_DURATION)
        y_bass = _load_stem(stems_dir, "bass", TARGET_SR, ANALYSIS_DURATION)
        y_other = _load_stem(stems_dir, "other", TARGET_SR, ANALYSIS_DURATION)
        y_vocals = _load_stem(stems_dir, "vocals", TARGET_SR, ANALYSIS_DURATION)
        if y_bass is not None and y_other is not None:
            y_bass_other = y_bass + y_other

    if y_drums is not None and y_bass_other is not None:
        logger.info("analyze_audio: using Demucs stems for analysis")
        y_percussive = y_drums
        y_harmonic = y_bass_other
    else:
        logger.info("analyze_audio: running HPSS (stems unavailable)")
        y_harmonic, y_percussive = librosa.effects.hpss(y_mono)

    bpm = _extract_bpm(y_percussive, TARGET_SR, y_bass=y_bass)
    key_str, mode_confidence = _detect_key(y_harmonic, TARGET_SR, y_bass=y_bass)
    transient_punch = _transient_punch(y_percussive, TARGET_SR)
    freq_peaks_hz = {
        "harmonic": _dominant_frequencies(y_harmonic, TARGET_SR),
        "percussive": _dominant_frequencies(y_percussive, TARGET_SR),
    }
    stereo_width = _stereo_width_label(y_stereo, sr_orig)

    if y_vocals is not None:
        vocal_presence = _vocal_presence_from_stems(y_vocals, y_mono)
    else:
        vocal_presence = _vocal_presence_estimate(y_harmonic, TARGET_SR)

    logger.info(
        "analyze_audio: done bpm=%.1f key=%s confidence=%.2f",
        bpm,
        key_str,
        mode_confidence,
    )

    return {
        "bpm": round(bpm, 2),
        "key": key_str,
        "mode_confidence": mode_confidence,
        "transient_punch": round(transient_punch, 3),
        "freq_peaks_hz": freq_peaks_hz,
        "stereo_width_label": stereo_width,
        "vocal_presence_label": vocal_presence,
    }


# ── Stem helpers ─────────────────────────────────────────────────────────────


def _load_stem(
    stems_dir: Path, name: str, target_sr: int, duration: float
) -> "np.ndarray | None":
    """Loads a single Demucs stem WAV as a mono float32 array at target_sr."""
    import librosa
    import soundfile as sf

    stem_path = stems_dir / f"{name}.wav"
    if not stem_path.exists():
        return None
    try:
        with sf.SoundFile(str(stem_path)) as snd:
            frames = min(int(duration * snd.samplerate), snd.frames)
            raw = snd.read(frames=frames, dtype="float32", always_2d=True)
        y = librosa.to_mono(raw.T)
        if snd.samplerate != target_sr:
            y = librosa.resample(y, orig_sr=snd.samplerate, target_sr=target_sr)
        return y
    except Exception as exc:
        logger.warning("_load_stem: failed to load %s — %s", stem_path.name, exc)
        return None


def _vocal_presence_from_stems(y_vocals: np.ndarray, y_mix: np.ndarray) -> str:
    """Computes vocal presence as RMS(vocals) / RMS(full mix)."""
    n = min(len(y_vocals), len(y_mix))
    rms_vocals = float(np.sqrt(np.mean(y_vocals[:n] ** 2))) + 1e-10
    rms_mix = float(np.sqrt(np.mean(y_mix[:n] ** 2))) + 1e-10
    ratio = rms_vocals / rms_mix
    if ratio > 0.40:
        return "forward"
    elif ratio > 0.20:
        return "present"
    else:
        return "background"


# ── Sub-extractors ────────────────────────────────────────────────────────────


def _kick_autocorr_score(y_drums: np.ndarray, sr: int, tempo_bpm: float) -> float:
    """Returns autocorrelation strength at the period matching tempo_bpm.

    Uses only the onset envelope of the drums signal so hi-hat energy
    (which rides at 2× or 4× the kick period) doesn't contaminate the score.
    Higher score → drums pulse more strongly at this tempo.

    Args:
        y_drums: mono drum stem at `sr` sample rate
        sr:      sample rate
        tempo_bpm: candidate tempo to score

    Returns:
        Autocorrelation value in [-1, 1] at the lag matching tempo_bpm.
        Returns 0.0 if the signal is too short or tempo is invalid.
    """
    import librosa

    HOP = 512  # onset envelope hop — gives ~43 frames/sec at 22050 Hz

    if tempo_bpm <= 0 or y_drums is None or y_drums.size < sr:
        return 0.0

    env = librosa.onset.onset_strength(y=y_drums, sr=sr, hop_length=HOP)
    acf = librosa.autocorrelate(env, max_size=env.size)
    lag = round((60.0 / tempo_bpm) * (sr / HOP))
    if lag <= 0 or lag >= len(acf):
        return 0.0
    acf = acf / (acf[0] + 1e-10)
    return float(acf[lag])


def _extract_bpm(y: np.ndarray, sr: int, y_bass: "np.ndarray | None" = None) -> float:
    import librosa

    def _track(sig, start):
        t, _ = librosa.beat.beat_track(y=sig, sr=sr, start_bpm=start)
        return float(t[0]) if hasattr(t, "__len__") else float(t)

    # Three drum seeds expose octave ambiguity in the tracker.
    # drums(160) is the key seed for fast hip-hop where lower seeds latch onto sub-pulses.
    t_d120 = _track(y, 120)
    t_d80 = _track(y, 80)
    t_d160 = _track(y, 160)
    t_b = _track(y_bass, 90) if y_bass is not None else None

    # Case 1: drums(120) and drums(80) are in a clean 2:1 ratio.
    # Use the bass tempo as the arbiter: whichever drum candidate is closer
    # to the bass tempo is more likely the felt tempo. The bass typically
    # follows the actual chord-changing pulse, so bass-near-drum_hi suggests
    # the drums are tracking a half-time kick pattern (Levitating, DNA),
    # while bass-near-drum_lo suggests a slow ballad with off-beat hits
    # (Bruno Mars - When I Was Your Man). Falls back to lower if no bass.
    drum_lo, drum_hi = sorted([t_d120, t_d80])
    if drum_lo > 0 and 1.80 <= drum_hi / drum_lo <= 2.20:
        if t_b is not None and abs(t_b - drum_hi) < abs(t_b - drum_lo):
            logger.info(
                "_extract_bpm: drums 2:1 (%.1f, %.1f) bass=%.1f → higher %.1f",
                drum_lo, drum_hi, t_b, drum_hi,
            )
            return float(drum_hi)
        logger.info(
            "_extract_bpm: drums 2:1 (%.1f, %.1f) → lower %.1f",
            drum_lo, drum_hi, drum_lo,
        )
        return float(drum_lo)

    # Case 2: drums(80) and bass(90) are in a 2:1 ratio — triplet feel.
    # Drums lock to dotted-quarter (2/3 of true), bass to triplet-eighth (4/3 of true).
    # Their arithmetic mean equals the true tempo exactly.
    if t_b is not None:
        bd_lo, bd_hi = sorted([t_d80, t_b])
        if bd_lo > 0 and 1.80 <= bd_hi / bd_lo <= 2.20:
            tempo = (bd_lo + bd_hi) / 2
            logger.info("_extract_bpm: drums/bass 2:1 (%.1f, %.1f) → mean %.1f", bd_lo, bd_hi, tempo)
            return float(tempo)

    # Case 3: drums(160) found a much faster pulse than drums(120) — fast hip-hop.
    # When drums(120) latches onto a 2/3 sub-pulse (dotted-quarter), drums(160)
    # resists the pull-down and finds the actual beat. Triggers only when the
    # ratio is meaningful and drums(160) is in a plausible fast-tempo range.
    # Upper bound is 180 (not 200) to avoid catching double-time errors like
    # sig_yukon where d160 = 2× true tempo.
    if t_d160 > t_d120 * 1.25 and 130 <= t_d160 <= 180:
        logger.info(
            "_extract_bpm: drums(160)=%.1f >> drums(120)=%.1f → fast pulse %.1f",
            t_d160, t_d120, t_d160,
        )
        return float(t_d160)

    # Case 3.5: drums(160) landed in the mid-tempo range (90-135) while all
    # lower seeds were pulled to a half-time sub-pulse. The 160-seed resists
    # the gravitational pull of sparse hip-hop kick patterns. Trust it when
    # drums(120) is more than 15% below drums(160).
    if 90 <= t_d160 <= 135 and t_d120 < t_d160 * 0.85:
        logger.info(
            "_extract_bpm: case3.5 d160=%.1f in mid-range, d120=%.1f pulled low → %.1f",
            t_d160, t_d120, t_d160,
        )
        return float(t_d160)

    # Case 3.7: triplet-feel correction. When all low-BPM seeds (d80, d120, bass)
    # converge on the same value below 80, the drums are locked at a dotted-quarter
    # sub-pulse (2/3 of true tempo). Multiplying by 1.5 gives the actual quarter-note
    # pulse. Fires for songs like sig_yukon where d80=d120=bass=64.6 and true=96.
    seeds_converged = (
        abs(t_d120 - t_d80) < 2.0
        and (t_b is None or abs(t_d120 - t_b) < 2.0)
    )
    if seeds_converged and t_d120 < 80:
        candidate = t_d120 * 1.5
        if 90 <= candidate <= 150:
            logger.info(
                "_extract_bpm: case3.7 triplet d80=d120=bass=%.1f → ×1.5 = %.1f",
                t_d120, candidate,
            )
            return float(candidate)

    # Case 4: drums(120) is in the typical pop/R&B range — trust it directly.
    if 90 <= t_d120 <= 165:
        logger.info("_extract_bpm: drums(120)=%.1f in pop range → trust", t_d120)
        return float(t_d120)

    # Case 5: fall through to legacy filter + halving/doubling heuristics
    candidates = [t_d120, t_d80, t_d160] + ([t_b] if t_b is not None else [])
    in_range = [c for c in candidates if 60 <= c <= 130]
    if not in_range:
        in_range = [min(candidates)]
    tempo = min(in_range)

    if tempo > 140:
        half = tempo / 2
        if 60 <= half <= 100:
            return float(half)

    if tempo < 100:
        # Skip doubling for genuine slow songs: when bass is the *only* in-range
        # candidate AND drums also report a high tempo (split kick/hi-hat
        # pattern), the song is genuinely slow at the bass rate. Adele Hello:
        # bass=83, drums oscillate 56/172, true=79. Without this guard, Case 5
        # doubles 83 → 166. With it, we return 83 (closer to true).
        bass_only_in_range = t_b is not None and in_range == [t_b]
        has_high_drum = any(t > 140 for t in (t_d80, t_d120, t_d160))
        skip_double = bass_only_in_range and has_high_drum
        if not skip_double:
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


def _score_keys_for_chroma(combined: np.ndarray) -> dict[str, float]:
    """Returns {key_name: correlation_score} for all 24 keys against a chroma vector."""
    scores: dict[str, float] = {}
    for i, root in enumerate(PITCH_CLASSES):
        rotated = np.roll(combined, -i)
        scores[f"{root} Major"] = float(np.corrcoef(rotated, MAJOR_TEMPLATE)[0, 1])
        scores[f"{root} Minor"] = float(np.corrcoef(rotated, MINOR_TEMPLATE)[0, 1])
    return scores


def _normalized_chroma_mean(y: np.ndarray, sr: int) -> np.ndarray:
    """Compute CENS chroma, dB-convert, time-average, z-score normalize."""
    import librosa
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=2048)
    chroma_db = librosa.amplitude_to_db(chroma, ref=np.max)
    combined = chroma_db.mean(axis=1)
    if np.max(np.abs(combined)) > 0:
        combined = (combined - combined.mean()) / (combined.std() + 1e-6)
    return combined


def _detect_key(
    y_harmonic: np.ndarray,
    sr: int,
    y_bass: "np.ndarray | None" = None,
) -> tuple[str, float]:
    """Detects musical key from a harmonic signal via CQT chroma correlation.

    Uses multi-window consensus: splits y_harmonic into 3 overlapping 30s windows,
    scores all 24 keys per window, then sums per-key scores across windows. This
    smooths out single-window anomalies — e.g. when a verse is in a different mode
    than the chorus, or when one window catches a chromatic passage. For signals
    shorter than 45s, falls back to single-window analysis.

    When y_bass is provided, applies a bass-root bias: if the strongest pitch
    class in the bass stem conflicts with the chroma winner's root AND the bass
    is anchored on a single pitch (concentration > 1.15), the bass-root key is
    preferred when it scored within 25% of the winner.
    """
    import librosa

    if y_harmonic is None or y_harmonic.size <= 1:
        return "Unknown", 0.0

    # Multi-window scoring: split into 3 overlapping windows of equal size.
    # For a 60s signal at sr=22050, each window is 30s with 15s overlap.
    WINDOW_SEC = 30.0
    win_samples = int(WINDOW_SEC * sr)
    n = y_harmonic.size

    if n >= int(45 * sr):
        starts = [0, (n - win_samples) // 2, n - win_samples]
        per_window_scores = [
            _score_keys_for_chroma(_normalized_chroma_mean(y_harmonic[s:s + win_samples], sr))
            for s in starts
        ]
        all_scores = {
            k: sum(ws[k] for ws in per_window_scores) / len(per_window_scores)
            for k in per_window_scores[0]
        }
    else:
        all_scores = _score_keys_for_chroma(_normalized_chroma_mean(y_harmonic, sr))

    best_key, best_score = max(all_scores.items(), key=lambda kv: kv[1])

    # Bass-root bias: bass lines follow the tonic, resolving relative-key ties.
    # Only applies when the bass is anchored on a single pitch (concentration > 1.15).
    # Melodic bass lines (sig_ballad, sig_lauv_julia, sig_hiphop) have low
    # concentration and would otherwise misfire — bass plays the 3rd/5th/7th
    # which isn't the tonic. The 1.15 threshold cleanly separates anchored
    # bass (1.16+) from melodic bass (≤ 1.08) in the validated test set.
    BASS_CONCENTRATION_MIN = 1.15
    if y_bass is not None and y_bass.size > 1:
        bass_chroma_mean = librosa.feature.chroma_cens(
            y=y_bass, sr=sr, hop_length=2048
        ).mean(axis=1)
        sorted_pcs = np.sort(bass_chroma_mean)[::-1]
        bass_concentration = float(sorted_pcs[0] / (sorted_pcs[1] + 1e-6))
        bass_root = PITCH_CLASSES[int(np.argmax(bass_chroma_mean))]
        winner_root = best_key.split()[0]
        if bass_root != winner_root and bass_concentration > BASS_CONCENTRATION_MIN:
            # Check if a bass-root key scored close enough to override
            for mode in ("Minor", "Major"):
                candidate = f"{bass_root} {mode}"
                c_score = all_scores.get(candidate, -np.inf)
                if c_score >= best_score * 0.75:  # within 25% of best
                    logger.info(
                        "_detect_key: bass-root override %s → %s (%.3f vs %.3f, conc=%.2f)",
                        best_key, candidate, c_score, best_score, bass_concentration,
                    )
                    best_key, best_score = candidate, c_score
                    break
        elif bass_root != winner_root:
            logger.info(
                "_detect_key: skipping bass-root override (concentration %.2f ≤ %.2f, bass=%s, winner=%s)",
                bass_concentration, BASS_CONCENTRATION_MIN, bass_root, winner_root,
            )

    confidence = round(float(np.clip(best_score, 0.0, 1.0)), 2)
    return best_key, confidence


def _transient_punch(y: np.ndarray, sr: int) -> float:
    import librosa

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peak = float(np.percentile(onset_env, 97))
    mean = float(onset_env.mean()) + 1e-6
    return float(np.clip((peak / mean - 1) / 20.0, 0, 1))


def _dominant_frequencies(y: np.ndarray, sr: int, top_n: int = 5) -> list[float]:
    fft = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    mask = (freqs >= 20) & (freqs <= 16000)
    fft_m, freqs_m = fft[mask], freqs[mask]
    if len(fft_m) == 0:
        return []
    top_idx = np.argsort(fft_m)[-top_n:][::-1]
    return [round(float(freqs_m[i]), 1) for i in top_idx]


def _stereo_width_label(y_stereo: np.ndarray, sr: int) -> str:
    """Estimates stereo width via L-R channel correlation."""
    if y_stereo.ndim < 2 or y_stereo.shape[0] < 2:
        return "mono"
    left, right = y_stereo[0], y_stereo[1]
    n = min(len(left), len(right))
    corr = float(np.corrcoef(left[:n], right[:n])[0, 1])
    if corr > 0.95:
        return "mono"
    elif corr > 0.70:
        return "narrow"
    elif corr > 0.40:
        return "medium"
    else:
        return "wide"


def _vocal_presence_estimate(y_harmonic: np.ndarray, sr: int) -> str:
    """
    Estimates vocal presence from spectral energy in the vocal band (200Hz–4kHz)
    relative to total harmonic energy. Approximate — no voice isolation.
    """
    fft = np.abs(np.fft.rfft(y_harmonic))
    freqs = np.fft.rfftfreq(len(y_harmonic), d=1.0 / sr)
    total = float(np.sum(fft**2)) + 1e-10
    vocal_mask = (freqs >= 200) & (freqs <= 4000)
    vocal = float(np.sum(fft[vocal_mask] ** 2))
    ratio = vocal / total
    if ratio > 0.45:
        return "forward"
    elif ratio > 0.25:
        return "present"
    else:
        return "background"
