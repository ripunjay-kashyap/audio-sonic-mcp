"""
Stage 2 — Streaming Audio Download
Uses yt-dlp to pull only the audio stream (no video), saving ~70% bandwidth.
"""

import logging
import subprocess
from subprocess import DEVNULL
from pathlib import Path

logger = logging.getLogger(__name__)


def download_audio(url: str, job_id: str, jobs_root: Path) -> Path:
    """
    Downloads the best available audio stream from the given URL.
    Returns the path to the downloaded raw audio file.

    Strategy:
    - bestaudio: grabs the highest-quality audio-only stream
    - No video mux: skips the heavy video track entirely
    - Output is saved to a per-job temp directory
    """
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    output_template = str(job_dir / "raw_audio.%(ext)s")

    cmd = [
        "yt-dlp",
        # Audio-only, best quality available
        "--format",
        "bestaudio/best",
        # Avoid video download entirely
        "--no-playlist",
        # Do not embed metadata (faster)
        "--no-embed-metadata",
        # Progress to stderr
        "--newline",
        "--progress",
        # Output path
        "--output",
        output_template,
        # No sponsorblock or chapters needed for audio analysis
        "--no-sponsorblock",
        url,
    ]

    logger.info("Downloading: %s", url)
    result = subprocess.run(
        cmd,
        stdin=DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,  # 5-minute cap
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"yt-dlp download failed for '{url}'.\nstderr: {stderr}")

    # Find the downloaded file (extension varies: webm, m4a, opus, etc.)
    candidates = list(job_dir.glob("raw_audio.*"))
    if not candidates:
        raise FileNotFoundError(
            f"yt-dlp reported success but no file found in: {job_dir}"
        )

    raw_path = max(candidates, key=lambda p: p.stat().st_size)
    logger.info("Downloaded → %s (%.1f MB)", raw_path, raw_path.stat().st_size / 1e6)
    return raw_path
