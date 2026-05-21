"""
Stage 2 — Streaming Audio Download
Pulls only the audio stream (no video) via yt-dlp, saving ~70% bandwidth.
"""

import logging
import os
from pathlib import Path

import yt_dlp

logger = logging.getLogger(__name__)


class _YtdlpLogger:
    """Routes yt-dlp output through Python logger — keeps MCP stdio pipe clean."""

    def debug(self, msg):
        # yt-dlp routes progress lines and verbose internals both through debug();
        # the "[debug] " prefix marks the internals, which we drop as noise.
        if not msg.startswith("[debug] "):
            logger.debug(msg)

    def info(self, msg):
        logger.info(msg)

    def warning(self, msg):
        logger.warning(msg)

    def error(self, msg):
        logger.error(msg)


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

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        # Correct key is "noplaylist" — yt-dlp silently ignores the misspelled
        # "no_playlist". Ensures a radio/playlist URL downloads only the video.
        "noplaylist": True,
        "geo_bypass": True,
        "logger": _YtdlpLogger(),
        "noprogress": False,
        "newline": True,
    }

    proxy_url = os.environ.get("YTDLP_PROXY")
    if proxy_url:
        ydl_opts["proxy"] = proxy_url

    logger.info("Downloading: %s", url)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError(f"yt-dlp download failed for '{url}': {exc}") from exc

    candidates = list(job_dir.glob("raw_audio.*"))
    if not candidates:
        raise FileNotFoundError(
            f"yt-dlp reported success but no file found in: {job_dir}"
        )

    raw_path = max(candidates, key=lambda p: p.stat().st_size)
    logger.info("Downloaded → %s (%.1f MB)", raw_path, raw_path.stat().st_size / 1e6)
    return raw_path
