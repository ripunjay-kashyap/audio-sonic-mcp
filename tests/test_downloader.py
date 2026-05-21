"""Tests for pipeline/downloader.py — yt-dlp-backed audio download."""

from unittest.mock import patch

from pipeline.downloader import download_audio
from tests.conftest import make_ydl_mock


def test_download_audio_does_not_pin_android_player_client(tmp_path):
    """Regression: pinning player_client=android limits YouTube to ~5 fragile
    formats and risks HTTP 403. yt-dlp's default client selection must be used so
    the MCP works out-of-the-box on a user's local PC with no PO-token setup."""
    job_id = "job1"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)
    # Pre-create the artefact so the post-download glob succeeds (mock download is a no-op).
    (job_dir / "raw_audio.mp4").write_bytes(b"\x00")

    with patch(
        "pipeline.downloader.yt_dlp.YoutubeDL", return_value=make_ydl_mock()
    ) as mock_cls:
        result = download_audio("https://www.youtube.com/watch?v=test", job_id, tmp_path)

    opts = mock_cls.call_args.args[0]
    clients = opts.get("extractor_args", {}).get("youtube", {}).get("player_client", [])
    assert "android" not in clients
    assert result.name == "raw_audio.mp4"


def test_download_audio_disables_playlist(tmp_path):
    """Regression: a radio/playlist URL must download only the single video, not
    the whole radio. Requires noplaylist=True — note yt-dlp silently ignores the
    misspelled 'no_playlist', so the exact key matters."""
    job_id = "job_np"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "raw_audio.mp4").write_bytes(b"\x00")

    with patch(
        "pipeline.downloader.yt_dlp.YoutubeDL", return_value=make_ydl_mock()
    ) as mock_cls:
        download_audio("https://www.youtube.com/watch?v=test&list=RDtest", job_id, tmp_path)

    opts = mock_cls.call_args.args[0]
    assert opts.get("noplaylist") is True
