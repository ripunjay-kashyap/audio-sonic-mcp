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

    from pipeline.window import find_ssm_window, pick_window

    TARGET_SR = 22050
    ANALYSIS_DURATION = 60.0  # also used by _load_stem for the cropped stems

    # For long tracks, find the most structurally repeated section via SSM
    # (cheap 11 kHz scan, ~2 s). Short/medium tracks use the default 0 s offset.
    with sf.SoundFile(str(wav_path)) as snd:
        sr_orig = snd.samplerate
        total_sec = snd.frames / sr_orig

    chorus_start = find_ssm_window(wav_path, total_sec) if total_sec >= 75.0 else None

    # Read directly via soundfile to avoid librosa's audioread fallback,
    # which spawns FFmpeg with inherited stdin and deadlocks under MCP.
    with sf.SoundFile(str(wav_path)) as snd:
        sr_orig = snd.samplerate
        offset_frames, frames_to_read = pick_window(snd.frames, sr_orig, chorus_start)
        snd.seek(offset_frames)
        raw = snd.read(frames=frames_to_read, dtype="float32", always_2d=True)
    y_stereo = raw.T
    if y_stereo.ndim == 1:
        y_stereo = np.stack([y_stereo, y_stereo])

    y_mono = librosa.to_mono(y_stereo)
    if sr_orig != TARGET_SR:
        y_mono = librosa.resample(y_mono, orig_sr=sr_orig, target_sr=TARGET_SR)

    # Try stem-based analysis; fall back to HPSS on any missing stem.
    # stem_offset aligns the stem read with the SSM chorus window so BPM,
    # key, and vocal analysis all draw from the same section of the track.
    stem_offset = chorus_start if chorus_start is not None else 0.0
    y_drums = y_bass = y_bass_other = y_vocals = None
    if stems_dir is not None:
        y_drums = _load_stem(stems_dir, "drums", TARGET_SR, ANALYSIS_DURATION, stem_offset)
        y_bass = _load_stem(stems_dir, "bass", TARGET_SR, ANALYSIS_DURATION, stem_offset)
        y_other = _load_stem(stems_dir, "other", TARGET_SR, ANALYSIS_DURATION, stem_offset)
        y_vocals = _load_stem(stems_dir, "vocals", TARGET_SR, ANALYSIS_DURATION, stem_offset)
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
    bpm_variable, bpm_range = _detect_bpm_variable(y_percussive, TARGET_SR, bpm)

    # Build harmonic evidence list: section A (Demucs/HPSS from SSM window) +
    # one or two extra HPSS passes for broader song context.
    # When SSM is active: run ONE complementary HPSS pass in the verse region
    # (just before the SSM chorus window) so that post-hoc checks like
    # Check A can see whether major-scale characteristic tones are absent
    # across the full song, not just the chorus.
    # Without SSM: keep the original two fixed passes at 90s and 150s.
    harmonics = [y_harmonic]
    if chorus_start is None:
        y_harm_b = _load_hpss_harmonic(wav_path, 90, 60, TARGET_SR)
        y_harm_c = _load_hpss_harmonic(wav_path, 150, 60, TARGET_SR)
        if y_harm_b is not None:
            harmonics.append(y_harm_b)
        if y_harm_c is not None:
            harmonics.append(y_harm_c)
    else:
        verse_offset = max(30.0, chorus_start - 45.0)
        y_harm_verse = _load_hpss_harmonic(wav_path, verse_offset, 60, TARGET_SR)
        if y_harm_verse is not None:
            harmonics.append(y_harm_verse)

    key_str, mode_confidence, key_ambiguous = _detect_key(harmonics, TARGET_SR, y_bass=y_bass)
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
        "analyze_audio: done bpm=%.1f variable=%s key=%s confidence=%.2f",
        bpm,
        bpm_variable,
        key_str,
        mode_confidence,
    )

    return {
        "bpm": round(bpm, 2),
        "bpm_variable": bpm_variable,
        "bpm_range": bpm_range,
        "key": key_str,
        "mode_confidence": mode_confidence,
        "key_ambiguous": key_ambiguous,
        "transient_punch": round(transient_punch, 3),
        "freq_peaks_hz": freq_peaks_hz,
        "stereo_width_label": stereo_width,
        "vocal_presence_label": vocal_presence,
    }


# ── HPSS section loader ───────────────────────────────────────────────────────


def _load_hpss_harmonic(
    wav_path: "Path",
    offset_sec: float,
    duration_sec: float,
    target_sr: int,
) -> "np.ndarray | None":
    """Load a raw WAV segment, apply HPSS, return the harmonic component.

    Returns None when offset_sec is at or beyond the track's end.
    """
    import librosa
    import soundfile as sf

    try:
        with sf.SoundFile(str(wav_path)) as snd:
            sr_orig = snd.samplerate
            total_sec = snd.frames / sr_orig
            if offset_sec >= total_sec:
                return None
            offset_frames = int(offset_sec * sr_orig)
            frames_to_read = min(
                int(duration_sec * sr_orig),
                snd.frames - offset_frames,
            )
            if frames_to_read <= 0:
                return None
            snd.seek(offset_frames)
            raw = snd.read(frames=frames_to_read, dtype="float32", always_2d=True)
        y = librosa.to_mono(raw.T)
        if sr_orig != target_sr:
            y = librosa.resample(y, orig_sr=sr_orig, target_sr=target_sr)
        y_harmonic, _ = librosa.effects.hpss(y)
        return y_harmonic
    except Exception as exc:
        logger.warning(
            "_load_hpss_harmonic: failed %s@%.0fs — %s", wav_path.name, offset_sec, exc
        )
        return None


# ── Stem helpers ─────────────────────────────────────────────────────────────


def _load_stem(
    stems_dir: Path, name: str, target_sr: int, duration: float, offset_sec: float = 0.0
) -> "np.ndarray | None":
    """Loads a single Demucs stem WAV as a mono float32 array at target_sr.

    offset_sec allows reading from the SSM-found chorus window rather than
    always starting at the beginning of the stem file.
    """
    import librosa
    import soundfile as sf

    stem_path = stems_dir / f"{name}.wav"
    if not stem_path.exists():
        return None
    try:
        with sf.SoundFile(str(stem_path)) as snd:
            offset_frames = int(offset_sec * snd.samplerate)
            frames = min(int(duration * snd.samplerate), snd.frames - offset_frames)
            if frames <= 0:
                return None
            snd.seek(offset_frames)
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


_BLOCK_SEC = 15.0    # duration of each analysis block (seconds)
_BLOCK_COUNT = 4     # number of consecutive blocks to segment the drums stem
_STABLE_STD = 2.0    # std dev below which tempo is considered rock-solid
_VARIABLE_STD = 5.0  # std dev above which tempo is flagged as variable


def _tempogram_prefers_high(y: np.ndarray, sr: int, lo: float, hi: float) -> bool:
    """Return True when the tempogram shows enough energy at 'hi' to resolve a 2:1
    ambiguity in favour of the faster candidate.

    Fast hip-hop with half-time feel (true ~140-145 BPM): hi-hats and ghost notes
    create consistent onsets at the full tempo → tempogram peak at hi is ≥ 55% of
    the peak at lo.

    True slow songs (~70-80 BPM): no beat grid at 2×, so the hi peak is weak.
    """
    import librosa

    if not (60.0 <= lo <= 90.0 and 120.0 <= hi <= 180.0):
        return False

    odf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    tgram = librosa.feature.tempogram(onset_envelope=odf, sr=sr, hop_length=512)
    freqs = librosa.tempo_frequencies(tgram.shape[0], sr=sr, hop_length=512)
    avg = tgram.mean(axis=1)

    def _peak(target: float) -> float:
        mask = np.abs(freqs - target) <= 3.0
        return float(avg[mask].max()) if mask.any() else 0.0

    e_lo = _peak(lo)
    e_hi = _peak(hi)
    ratio = e_hi / (e_lo + 1e-6)
    logger.info(
        "_tempogram_prefers_high: lo=%.1f(e=%.3f) hi=%.1f(e=%.3f) ratio=%.2f",
        lo, e_lo, hi, e_hi, ratio,
    )
    return ratio >= 0.55


def _extract_bpm(y: np.ndarray, sr: int, y_bass: "np.ndarray | None" = None) -> float:
    """Primary BPM estimator: multi-seed beat tracking over the full analysis window
    with bass-stem arbitration to resolve octave ambiguity."""
    import librosa

    def _track(sig, start):
        t, _ = librosa.beat.beat_track(y=sig, sr=sr, start_bpm=start)
        return float(t[0]) if hasattr(t, "__len__") else float(t)

    t_d120 = _track(y, 120)
    t_d80 = _track(y, 80)
    t_d160 = _track(y, 160)
    t_b = _track(y_bass, 90) if y_bass is not None else None

    drum_lo, drum_hi = sorted([t_d120, t_d80])
    if drum_lo > 0 and 1.80 <= drum_hi / drum_lo <= 2.20:
        if t_b is not None and abs(t_b - drum_hi) < abs(t_b - drum_lo):
            logger.info(
                "_extract_bpm: drums 2:1 (%.1f, %.1f) bass=%.1f → higher %.1f",
                drum_lo, drum_hi, t_b, drum_hi,
            )
            return float(drum_hi)
        # Bass can't disambiguate (also at half-tempo or absent). Use the tempogram:
        # fast hip-hop (true ~140-145 BPM) has hi-hat energy at the full tempo even
        # when kick/snare follows a half-time feel; true slow songs (~70-80 BPM) don't.
        if _tempogram_prefers_high(y, sr, drum_lo, drum_hi):
            logger.info(
                "_extract_bpm: drums 2:1 (%.1f, %.1f) tempogram → higher %.1f",
                drum_lo, drum_hi, drum_hi,
            )
            return float(drum_hi)
        logger.info(
            "_extract_bpm: drums 2:1 (%.1f, %.1f) → lower %.1f",
            drum_lo, drum_hi, drum_lo,
        )
        return float(drum_lo)

    if t_b is not None:
        bd_lo, bd_hi = sorted([t_d80, t_b])
        if bd_lo > 0 and 1.80 <= bd_hi / bd_lo <= 2.20:
            tempo = (bd_lo + bd_hi) / 2
            logger.info("_extract_bpm: drums/bass 2:1 (%.1f, %.1f) → mean %.1f", bd_lo, bd_hi, tempo)
            return float(tempo)

    if t_d160 > t_d120 * 1.25 and 130 <= t_d160 <= 180:
        logger.info(
            "_extract_bpm: drums(160)=%.1f >> drums(120)=%.1f → fast pulse %.1f",
            t_d160, t_d120, t_d160,
        )
        return float(t_d160)

    if 90 <= t_d160 <= 135 and t_d120 < t_d160 * 0.85:
        logger.info(
            "_extract_bpm: case3.5 d160=%.1f in mid-range, d120=%.1f pulled low → %.1f",
            t_d160, t_d120, t_d160,
        )
        return float(t_d160)

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

    if 90 <= t_d120 <= 165:
        logger.info("_extract_bpm: drums(120)=%.1f in pop range → trust", t_d120)
        return float(t_d120)

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


def _detect_bpm_variable(
    y_drums: np.ndarray, sr: int, primary_bpm: float
) -> "tuple[bool, list[float] | None]":
    """
    Segments the drums stem into _BLOCK_COUNT × _BLOCK_SEC blocks, estimates
    tempo per block, snaps each estimate to the primary BPM's octave, then
    uses std dev to classify as stable or variable.

    Using primary_bpm as the snap anchor is reliable because the primary
    estimate is already accurate — block estimates that land at 2× or ½× the
    primary are octave errors that should be normalised before computing variance.

    Returns:
        bpm_variable — True when std dev across blocks exceeds _VARIABLE_STD
        bpm_range    — [lo, hi] of snapped estimates when variable, else None
    """
    import librosa

    block_samples = int(_BLOCK_SEC * sr)
    estimates: list[int] = []
    rms_values: list[float] = []

    for i in range(_BLOCK_COUNT):
        start = i * block_samples
        block = y_drums[start : start + block_samples]
        if len(block) < block_samples // 2:
            continue
        def _t(seed):
            r, _ = librosa.beat.beat_track(y=block, sr=sr, start_bpm=seed)
            return float(r[0]) if hasattr(r, "__len__") else float(r)
        val = _best_block_bpm(_t(80), _t(120), _t(160))
        estimates.append(round(val))
        rms_values.append(float(np.sqrt(np.mean(block ** 2))) + 1e-10)

    if len(estimates) < 2:
        return False, None

    # Drop low-energy blocks — breakdowns or near-silence produce noise
    # estimates that inflate std dev and falsely flag stable songs as variable.
    max_rms = max(rms_values)
    ENERGY_FLOOR = 0.20  # block must be ≥ 20% of the loudest block's RMS
    valid = [(e, r) for e, r in zip(estimates, rms_values)
             if r >= max_rms * ENERGY_FLOOR]
    if len(valid) < 2:
        return False, None
    estimates = [e for e, _ in valid]
    rms_values = [r for _, r in valid]

    snapped = _snap_to_primary(estimates, primary_bpm)
    std = float(np.std(snapped))
    lo, hi = float(min(snapped)), float(max(snapped))

    logger.info(
        "_detect_bpm_variable: blocks=%s snapped=%s std=%.1f (low-energy dropped: %d)",
        estimates, snapped, std, _BLOCK_COUNT - len(estimates),
    )

    if std < _VARIABLE_STD:
        return False, None

    majority = _find_majority(snapped)
    if majority is not None:
        logger.info(
            "_detect_bpm_variable: variable — majority=%.1f range=[%.1f, %.1f]",
            majority, lo, hi,
        )
        return True, [lo, hi]

    louder = float(_louder_pair_bpm(snapped, rms_values))
    logger.info(
        "_detect_bpm_variable: 50/50 — louder=%.1f range=[%.1f, %.1f]",
        louder, lo, hi,
    )
    return True, [lo, hi]


def _best_block_bpm(t80: float, t120: float, t160: float) -> float:
    """Pick the most plausible BPM from three seed estimates for a single block.

    Mirrors the old multi-seed heuristics at the per-block level:
    - Fast track: seed_160 found a significantly faster pulse than seed_120
      → trust it (catches fast hip-hop where seed_120 locks onto a sub-beat)
    - 2:1 split between seed_80 and seed_120 → prefer the lower value
      (the tracker latched onto a doubled pulse; the lower seed is correct)
    - Otherwise trust seed_120 when it lands in a plausible BPM range
    - Last resort: fall back to whichever seed gave the median value
    """
    if t160 > t120 * 1.25 and 110 <= t160 <= 185:
        return t160
    lo, hi = sorted([t80, t120])
    if lo > 0 and 1.80 <= hi / lo <= 2.20:
        return lo
    if 80 <= t120 <= 170:
        return t120
    return sorted([t80, t120, t160])[1]  # median as last resort


def _snap_to_primary(estimates: list[int], primary_bpm: float) -> list[int]:
    """Snap block estimates to the primary BPM's octave for variance computation."""
    result = []
    for v in estimates:
        if primary_bpm > 0 and abs(v / primary_bpm - 2.0) < 0.20:
            result.append(round(v / 2))
        elif primary_bpm > 0 and abs(v * 2 / primary_bpm - 1.0) < 0.20:
            result.append(round(v * 2))
        else:
            result.append(v)
    return result


def _octave_snap(estimates: list[int]) -> list[int]:
    """Snap values that are ~2× or ~½× the median to the majority octave."""
    if len(estimates) < 2:
        return list(estimates)
    mid = sorted(estimates)[len(estimates) // 2]
    result = []
    for v in estimates:
        if mid > 0 and abs(v / mid - 2.0) < 0.15:
            result.append(round(v / 2))
        elif mid > 0 and abs((v * 2) / mid - 1.0) < 0.15:
            result.append(round(v * 2))
        else:
            result.append(v)
    return result


def _find_majority(estimates: list[int]) -> "int | None":
    """Return mean of the majority cluster when 3+ values are within ±2 BPM."""
    for val in estimates:
        cluster = [v for v in estimates if abs(v - val) <= 2]
        if len(cluster) >= 3:
            return round(sum(cluster) / len(cluster))
    return None


def _louder_pair_bpm(estimates: list[int], rms_values: list[float]) -> int:
    """For a 50/50 tempo split, return the BPM of the louder pair of blocks."""
    unique = sorted(set(estimates))
    if len(unique) != 2:
        return round(sum(estimates) / len(estimates))
    rms_by_bpm: dict[int, float] = {u: 0.0 for u in unique}
    for bpm_val, rms in zip(estimates, rms_values):
        closest = min(unique, key=lambda u: abs(u - bpm_val))
        rms_by_bpm[closest] += rms
    return max(rms_by_bpm, key=lambda u: rms_by_bpm[u])


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
    harmonics: "np.ndarray | list[np.ndarray | None]",
    sr: int,
    y_bass: "np.ndarray | None" = None,
) -> tuple[str, float]:
    """Detects musical key by correlating CQT chroma against Krumhansl templates.

    Accepts either a single harmonic signal or a list of signals (e.g. Demucs
    section A + HPSS sections B and C). For each signal, applies 3-window scoring
    and sums all per-key correlation scores before picking the winner. This lets
    later song sections contribute evidence when the intro/verse hides the true key.

    When y_bass is provided, applies a bass-root bias: if the strongest pitch
    class in the bass stem conflicts with the chroma winner's root AND the bass
    is anchored on a single pitch (concentration > 1.15), the bass-root key is
    preferred when it scored within 25% of the winner.
    """
    import librosa

    # Normalise to a flat list, discarding None and empty arrays
    if isinstance(harmonics, np.ndarray):
        signals: list[np.ndarray] = [harmonics] if harmonics.size > 1 else []
    else:
        signals = [h for h in harmonics if h is not None and h.size > 1]

    if not signals:
        return "Unknown", 0.0, False

    WINDOW_SEC = 30.0
    win_samples = int(WINDOW_SEC * sr)

    # Accumulate per-key correlation scores across all signals and their windows.
    # Also collect chroma vectors for the characteristic-tone mode tiebreaker below.
    all_scores: dict[str, float] = {}
    all_chroma_vecs: list[np.ndarray] = []
    for y_harmonic in signals:
        n = y_harmonic.size
        if n >= int(45 * sr):
            starts = [0, (n - win_samples) // 2, n - win_samples]
            window_chromas = [
                _normalized_chroma_mean(y_harmonic[s:s + win_samples], sr) for s in starts
            ]
            per_window_scores = [_score_keys_for_chroma(c) for c in window_chromas]
            signal_scores = {
                k: sum(ws[k] for ws in per_window_scores) / len(per_window_scores)
                for k in per_window_scores[0]
            }
            all_chroma_vecs.extend(window_chromas)
        else:
            chroma_vec = _normalized_chroma_mean(y_harmonic, sr)
            signal_scores = _score_keys_for_chroma(chroma_vec)
            all_chroma_vecs.append(chroma_vec)
        for k, v in signal_scores.items():
            all_scores[k] = all_scores.get(k, 0.0) + v

    best_key, best_score = max(all_scores.items(), key=lambda kv: kv[1])
    correction_fired = False  # tracks whether any post-hoc rule changed best_key

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
                    correction_fired = True
                    break
        elif bass_root != winner_root:
            logger.info(
                "_detect_key: skipping bass-root override (concentration %.2f ≤ %.2f, bass=%s, winner=%s)",
                bass_concentration, BASS_CONCENTRATION_MIN, bass_root, winner_root,
            )

    # Characteristic-tone mode tiebreaker: when Krumhansl confidence is low
    # (< 0.40), compare scale-degree energy sums for minor (b3/b6/b7) vs major
    # (3/6/7) to resolve parallel-key ambiguity caused by modal sections in the
    # analyzed window (e.g., a verse that leans the opposite mode from the song's
    # overall key).
    if best_score < 0.40 and all_chroma_vecs:
        avg_chroma = np.mean(all_chroma_vecs, axis=0)
        winner_root_str, winner_mode = best_key.split()
        root_idx = PITCH_CLASSES.index(winner_root_str)
        minor_tones = float(
            avg_chroma[(root_idx + 3) % 12]    # b3
            + avg_chroma[(root_idx + 8) % 12]  # b6
            + avg_chroma[(root_idx + 10) % 12] # b7
        )
        major_tones = float(
            avg_chroma[(root_idx + 4) % 12]    # 3
            + avg_chroma[(root_idx + 9) % 12]  # 6
            + avg_chroma[(root_idx + 11) % 12] # 7
        )
        char_mode = "Minor" if minor_tones > major_tones else "Major"
        if char_mode != winner_mode:
            candidate = f"{winner_root_str} {char_mode}"
            logger.info(
                "_detect_key: char-tone mode flip %s → %s (min_tones=%.3f, maj_tones=%.3f, conf=%.2f)",
                best_key, candidate, minor_tones, major_tones, best_score,
            )
            best_key, best_score = candidate, all_scores.get(candidate, best_score)
            correction_fired = True

    # ── Post-hoc key correction ───────────────────────────────────────────────
    # Two targeted checks that catch failure modes the Krumhansl correlator
    # can't resolve on its own.  Both require the mean chroma vector.
    if all_chroma_vecs:
        avg_cv = np.mean(all_chroma_vecs, axis=0)
        pk_root_str, pk_mode = best_key.split()
        pk_root_idx = PITCH_CLASSES.index(pk_root_str)

        if pk_mode == "Major":
            # Check A — absent characteristic-tone override
            # If EITHER the major 3rd (4 semitones) OR the major 7th / leading
            # tone (11 semitones) of the detected key has strongly negative
            # z-scored chroma energy, the detected root is almost certainly
            # acting as the subdominant (IV) of the true minor key.
            # Both checks use the same -0.5 threshold; firing on either is
            # enough because a genuine major key requires both tones to be present.
            major_third_energy = float(avg_cv[(pk_root_idx + 4) % 12])
            major_seventh_energy = float(avg_cv[(pk_root_idx + 11) % 12])
            if major_third_energy < -0.5 or major_seventh_energy < -0.5:
                candidate_root = (pk_root_idx + 7) % 12
                candidate = f"{PITCH_CLASSES[candidate_root]} Minor"
                c_score = all_scores.get(candidate, -np.inf)
                if c_score > 0.5:
                    logger.info(
                        "_detect_key: absent-char-tone %s (3rd=%.2f 7th=%.2f) → %s (%.3f)",
                        best_key, major_third_energy, major_seventh_energy, candidate, c_score,
                    )
                    best_key, best_score = candidate, c_score
                    correction_fired = True

        elif pk_mode == "Minor":
            # Check B — 5th-alias distinguishing-tone check
            # When the bass pedals on the 5th degree, the chroma may treat the
            # 5th as the root, producing an alias minor key a P5 above the true
            # key.  Resolve by comparing the one note that distinguishes the two
            # keys: the alias key contains (root+1) semitone; the detected key
            # contains (root+2) semitone.  If the alias's distinctive note has
            # more energy, prefer the alias (the true key).
            alias_root = (pk_root_idx + 5) % 12
            alias_key = f"{PITCH_CLASSES[alias_root]} Minor"
            alias_score = all_scores.get(alias_key, -np.inf)
            if alias_score > best_score * 0.30:
                alias_dist = float(avg_cv[(pk_root_idx + 1) % 12])
                detected_dist = float(avg_cv[(pk_root_idx + 2) % 12])
                if alias_dist > detected_dist:
                    logger.info(
                        "_detect_key: 5th-alias %s → %s "
                        "(alias_dist=%.2f > det_dist=%.2f, score=%.3f)",
                        best_key, alias_key, alias_dist, detected_dist, alias_score,
                    )
                    best_key, best_score = alias_key, alias_score
                    correction_fired = True

    confidence = round(float(np.clip(best_score, 0.0, 1.0)), 2)

    # Ambiguity flag: when no domain-specific correction fired, check whether
    # the raw Krumhansl scores are too close to call.  A gap < 0.08 between
    # rank-1 and rank-2 means the chromagram fits two keys almost equally well.
    # When a correction did fire we trust the musical evidence and stay quiet.
    AMBIGUITY_THRESHOLD = 0.08
    if not correction_fired and len(all_scores) >= 2:
        sorted_scores = sorted(all_scores.values(), reverse=True)
        gap = sorted_scores[0] - sorted_scores[1]
        key_ambiguous = gap < AMBIGUITY_THRESHOLD
    else:
        key_ambiguous = False

    return best_key, confidence, key_ambiguous


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
