"""
Shared analysis window selection for the pipeline.

All three audio stages (stem separation, analyzer, vectorizer) read a 60s
window starting at a 30s offset, so the analysis lands past the intro in
verse 1 / pre-chorus where BPM and harmony are most representative.

For shorter songs the window adapts:
  - track < 30s:        read the whole track from 0
  - 30s ≤ track < 75s:  read 0–60s (or 0 → end if shorter)
  - track ≥ 75s:        read 30s–90s (the normal window)
"""

DEFAULT_OFFSET_SEC = 30.0
DEFAULT_DURATION_SEC = 60.0
SHORT_TRACK_THRESHOLD_SEC = 30.0   # below this, analyze the whole track
OFFSET_VIABLE_THRESHOLD_SEC = 75.0  # below this, skip the offset


def pick_window(
    total_frames: int,
    sr: int,
    offset_sec: float = DEFAULT_OFFSET_SEC,
    duration_sec: float = DEFAULT_DURATION_SEC,
) -> tuple[int, int]:
    """Return ``(start_frame, frames_to_read)`` for the analysis window.

    The caller is expected to ``snd.seek(start_frame)`` then
    ``snd.read(frames=frames_to_read, ...)``.
    """
    total_seconds = total_frames / sr
    if total_seconds < SHORT_TRACK_THRESHOLD_SEC:
        return 0, total_frames
    if total_seconds < OFFSET_VIABLE_THRESHOLD_SEC:
        return 0, min(int(duration_sec * sr), total_frames)
    offset_frames = int(offset_sec * sr)
    return offset_frames, min(int(duration_sec * sr), total_frames - offset_frames)
