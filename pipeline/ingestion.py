"""
Stage 1 — Ingestion & Validation
Validates that the URL is a supported source and reachable.
"""

import os
import re
import subprocess
import json
from pathlib import Path
from subprocess import DEVNULL
from urllib.parse import urlparse


SUPPORTED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "youtu.be",
    "music.youtube.com",
    "m.youtube.com",
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

    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-playlist",
        "--quiet",
        "--geo-bypass",
        "--extractor-args", "youtube:player_client=android",
    ]
    
    proxy_url = os.environ.get("YTDLP_PROXY")
    if proxy_url:
        cmd.extend(["--proxy", proxy_url])
        
    cmd.append(url)

    # Probe metadata without downloading
    result = subprocess.run(
        cmd,
        stdin=DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"yt-dlp failed to probe '{url}'. Reason: {stderr or 'unknown error'}"
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

    categories = meta.get("categories") or []
    genre_hint = categories[0] if categories else None

    return {
        "title": meta.get("title", "Unknown"),
        "duration_sec": duration,
        "uploader": meta.get("uploader", "Unknown"),
        "thumbnail": meta.get("thumbnail"),
        "webpage_url": meta.get("webpage_url", url),
        "extractor": meta.get("extractor", "unknown"),
        "genre_hint": genre_hint,
    }


def save_metadata(job_dir: Path, info: dict) -> None:
    """
    Atomically persists source metadata to disk alongside the stems.
    Uses a .tmp → rename pattern to prevent partial-write corruption.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    target = job_dir / "metadata.json"
    tmp = job_dir / "metadata.json.tmp"
    try:
        tmp.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, target)  # atomic on POSIX; near-atomic on Windows NTFS
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def load_metadata(job_dir: Path) -> dict | None:
    """
    Loads cached source metadata from disk.
    Returns None if the file does not exist or is malformed.
    """
    path = job_dir / "metadata.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _is_direct_audio(url: str) -> bool:
    """Allow direct .mp3/.wav/.flac/.ogg URLs."""
    return bool(re.search(r"\.(mp3|wav|flac|ogg|m4a|aac)(\?.*)?$", url, re.IGNORECASE))
