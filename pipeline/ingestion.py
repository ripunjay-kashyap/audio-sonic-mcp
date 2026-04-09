"""
Stage 1 — Ingestion & Validation
Validates that the URL is a supported source and reachable.
"""

import re
import subprocess
import json
from urllib.parse import urlparse


SUPPORTED_HOSTS = {
    "youtube.com", "www.youtube.com", "youtu.be",
    "music.youtube.com", "m.youtube.com",
}


def validate_source(url: str) -> dict:
    """
    Validates the URL and probes metadata via yt-dlp.
    Returns a dict with title, duration, uploader, and thumbnail.
    Raises ValueError for unsupported or unreachable sources.
    """
    parsed = urlparse(url)

    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL format: {url!r}")

    host = parsed.netloc.lstrip("www.")
    if host not in SUPPORTED_HOSTS and not _is_direct_audio(url):
        raise ValueError(
            f"Unsupported source: '{parsed.netloc}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_HOSTS))}"
        )

    # Probe metadata without downloading
    result = subprocess.run(
        [
            "yt-dlp",
            "--dump-json",
            "--no-playlist",
            "--quiet",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"yt-dlp failed to probe '{url}'. "
            f"Reason: {stderr or 'unknown error'}"
        )

    try:
        meta = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse yt-dlp metadata: {exc}") from exc

    duration = meta.get("duration", 0)
    if duration and duration > 3600:
        raise ValueError(
            f"Track duration {duration}s exceeds 60-minute limit. "
            "Please use a shorter clip."
        )

    return {
        "title": meta.get("title", "Unknown"),
        "duration_sec": duration,
        "uploader": meta.get("uploader", "Unknown"),
        "thumbnail": meta.get("thumbnail"),
        "webpage_url": meta.get("webpage_url", url),
        "extractor": meta.get("extractor", "unknown"),
    }


def _is_direct_audio(url: str) -> bool:
    """Allow direct .mp3/.wav/.flac/.ogg URLs."""
    return bool(re.search(r"\.(mp3|wav|flac|ogg|m4a|aac)(\?.*)?$", url, re.IGNORECASE))
