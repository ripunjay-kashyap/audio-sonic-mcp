"""
Shared analysis window selection for the pipeline.

All three audio stages (stem separation, analyzer, vectorizer) read a 60s
window. For long tracks the window start is chosen via a Self-Similarity
Matrix (SSM) that finds the most structurally repeated 30s section — usually
the first chorus — so the analysis lands in harmonically representative
material even when the intro sits in a different key.

For shorter songs the window adapts:
  - track < 30s:        read the whole track from 0
  - 30s ≤ track < 75s:  read 0–60s (or 0 → end if shorter)
  - track ≥ 75s:        SSM chorus window (60s centred on best 30s block)
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DURATION_SEC = 60.0
SHORT_TRACK_THRESHOLD_SEC = 30.0
OFFSET_VIABLE_THRESHOLD_SEC = 75.0
_SSM_SR = 11025   # low-res SR for cheap SSM scan
_SSM_HOP = 512
_CHORUS_BLOCK_SEC = 30.0  # the repeating block we search for


def find_ssm_window(wav_path: Path, total_sec: float) -> float:
    """Return the start-second of the most tonally representative 60s window.

    Divides the track into 15s blocks, computes mean CENS chroma per block,
    filters out low-energy blocks (silence, very quiet sections) and the first
    30 s (intro), then finds the block whose chroma is closest to the overall
    median — i.e. the block most representative of the song's tonal centre.
    The 60s analysis window is centred on that block.

    This replaces a pure SSM approach which is susceptible to edge effects
    (frame-0 bias from affinity column sums) that caused it to always return 0 s.
    Falls back to the fixed 30 s offset on any error.
    """
    import librosa
    import numpy as np

    BLOCK_SEC = _CHORUS_BLOCK_SEC / 2  # 15 s blocks
    MIN_INTRO_SEC = 30.0               # always skip the first 30 s

    try:
        y, _ = librosa.load(str(wav_path), sr=_SSM_SR, mono=True)
        chroma = librosa.feature.chroma_cens(y=y, sr=_SSM_SR, hop_length=_SSM_HOP)
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=_SSM_HOP)[0]

        block_frames = int(BLOCK_SEC * _SSM_SR / _SSM_HOP)
        n_frames = chroma.shape[1]
        n_blocks = n_frames // block_frames

        if n_blocks < 4:
            return 30.0

        block_chromas = []
        block_energies = []
        for i in range(n_blocks):
            s = i * block_frames
            e = min(s + block_frames, n_frames)
            block_chromas.append(chroma[:, s:e].mean(axis=1))
            block_energies.append(float(rms[s:e].mean()))

        block_chromas = np.array(block_chromas)    # (n_blocks, 12)
        block_energies = np.array(block_energies)  # (n_blocks,)

        # Exclude intro and low-energy blocks from the search
        min_block = int(np.ceil(MIN_INTRO_SEC / BLOCK_SEC))
        energy_floor = block_energies.max() * 0.20
        valid = np.array([
            i >= min_block and block_energies[i] >= energy_floor
            for i in range(n_blocks)
        ])

        if not valid.any():
            logger.warning("find_ssm_window: no valid blocks — falling back to 30 s offset")
            return 30.0

        # Median chroma of valid blocks = the song's tonal centre
        median_chroma = np.median(block_chromas[valid], axis=0)

        # Score each valid block by correlation with the median
        correlations = np.full(n_blocks, -np.inf)
        for i in range(n_blocks):
            if valid[i]:
                r = float(np.corrcoef(block_chromas[i], median_chroma)[0, 1])
                correlations[i] = r if np.isfinite(r) else -np.inf

        best_block = int(np.argmax(correlations))
        center_sec = (best_block + 0.5) * BLOCK_SEC
        start_sec = max(0.0, center_sec - DEFAULT_DURATION_SEC / 2)
        start_sec = min(start_sec, max(0.0, total_sec - DEFAULT_DURATION_SEC))

        logger.info(
            "find_ssm_window: best_block=%d center=%.1fs → window_start=%.1fs",
            best_block, center_sec, start_sec,
        )
        return start_sec
    except Exception as exc:
        logger.warning("find_ssm_window: failed (%s) — falling back to 30 s offset", exc)
        return 30.0


def pick_window(
    total_frames: int,
    sr: int,
    chorus_start_sec: "float | None" = None,
    duration_sec: float = DEFAULT_DURATION_SEC,
) -> tuple[int, int]:
    """Return ``(start_frame, frames_to_read)`` for the analysis window.

    When ``chorus_start_sec`` is provided (from ``find_ssm_window``), it is
    used as the window start for long tracks instead of the fixed 30 s offset.
    """
    total_seconds = total_frames / sr
    if total_seconds < SHORT_TRACK_THRESHOLD_SEC:
        return 0, total_frames
    if total_seconds < OFFSET_VIABLE_THRESHOLD_SEC:
        return 0, min(int(duration_sec * sr), total_frames)

    offset_sec = chorus_start_sec if chorus_start_sec is not None else 30.0
    offset_frames = int(offset_sec * sr)
    return offset_frames, min(int(duration_sec * sr), total_frames - offset_frames)
