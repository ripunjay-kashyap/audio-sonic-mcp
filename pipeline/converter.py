"""
Stage 3 — WAV Conversion
Standardizes raw audio to 44.1kHz / 16-bit stereo WAV — the format
htdemucs expects for best-quality separation.
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 44100
TARGET_CHANNELS = 2   # stereo
TARGET_BIT_DEPTH = "s16"  # signed 16-bit PCM


def convert_to_wav(raw_audio_path: Path) -> Path:
    """
    Converts the raw downloaded file to a normalized WAV.
    Uses FFmpeg with high-quality resampling (soxr engine).

    Returns the path to the .wav output file.
    """
    if not raw_audio_path.exists():
        raise FileNotFoundError(f"Raw audio not found: {raw_audio_path}")

    wav_path = raw_audio_path.parent / "input.wav"

    if wav_path.exists():
        logger.info("WAV already exists, skipping conversion: %s", wav_path)
        return wav_path

    cmd = [
        "ffmpeg",
        "-y",                          # overwrite output
        "-i", str(raw_audio_path),     # input file
        "-vn",                         # strip any video stream
        "-acodec", "pcm_s16le",        # PCM 16-bit little-endian
        "-ar", str(TARGET_SAMPLE_RATE),# 44.1kHz
        "-ac", str(TARGET_CHANNELS),   # stereo
        "-af", "aresample=resampler=soxr",  # high-quality resampler
        "-map_metadata", "-1",         # strip metadata (clean slate)
        str(wav_path),
    ]

    logger.info(
        "Converting %s → %s @ %dHz stereo …",
        raw_audio_path.name, wav_path.name, TARGET_SAMPLE_RATE,
    )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg conversion failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr: {result.stderr[-2000:]}"
        )

    size_mb = wav_path.stat().st_size / 1e6
    logger.info("WAV ready: %s (%.1f MB)", wav_path, size_mb)
    return wav_path
