"""Tests for pipeline/ingestion.py — URL validation and yt-dlp metadata probe."""

from unittest.mock import patch

import pytest
import yt_dlp

from pipeline.ingestion import _is_direct_audio, validate_source
from tests.conftest import make_ydl_mock


# ── _is_direct_audio ──────────────────────────────────────────────────────────


class TestIsDirectAudio:
    @pytest.mark.parametrize(
        "url",
        [
            "https://cdn.example.com/track.mp3",
            "https://cdn.example.com/track.wav",
            "https://cdn.example.com/track.flac",
            "https://cdn.example.com/track.ogg",
            "https://cdn.example.com/track.m4a",
            "https://cdn.example.com/track.aac",
            "https://cdn.example.com/track.MP3",  # case-insensitive
            "https://cdn.example.com/track.mp3?token=abc",  # query string
        ],
    )
    def test_recognised_audio_extensions(self, url):
        assert _is_direct_audio(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=abc123",
            "https://cdn.example.com/page.html",
            "https://cdn.example.com/image.jpg",
            "https://cdn.example.com/audio",  # no extension
        ],
    )
    def test_non_audio_urls(self, url):
        assert _is_direct_audio(url) is False


# ── validate_source ───────────────────────────────────────────────────────────


class TestValidateSource:
    def test_invalid_url_format_raises(self):
        with pytest.raises(ValueError, match="Invalid URL format"):
            validate_source("not-a-url")

    def test_missing_scheme_raises(self):
        with pytest.raises(ValueError, match="Invalid URL format"):
            validate_source("www.youtube.com/watch?v=abc")

    def test_unsupported_host_raises(self):
        with pytest.raises(ValueError, match="Unsupported source"):
            validate_source("https://soundcloud.com/artist/track")

    def test_valid_youtube_url_returns_metadata(self):
        meta = {
            "title": "Test Track",
            "duration": 180,
            "uploader": "Test Artist",
            "thumbnail": "https://img.example.com/thumb.jpg",
            "webpage_url": "https://www.youtube.com/watch?v=test123",
            "extractor": "youtube",
        }
        with patch("pipeline.ingestion.yt_dlp.YoutubeDL", return_value=make_ydl_mock(meta)):
            result = validate_source("https://www.youtube.com/watch?v=test123")

        assert result["title"] == "Test Track"
        assert result["duration_sec"] == 180
        assert result["uploader"] == "Test Artist"
        assert result["extractor"] == "youtube"

    @pytest.mark.parametrize(
        "host",
        [
            "https://youtu.be/abc123",
            "https://music.youtube.com/watch?v=abc",
            "https://m.youtube.com/watch?v=abc",
        ],
    )
    def test_all_supported_youtube_hosts_accepted(self, host):
        meta = {"title": "T", "duration": 60, "uploader": "U"}
        with patch("pipeline.ingestion.yt_dlp.YoutubeDL", return_value=make_ydl_mock(meta)):
            result = validate_source(host)
        assert result["duration_sec"] == 60

    def test_yt_dlp_failure_raises_runtime_error(self):
        error = yt_dlp.utils.DownloadError("Video unavailable")
        with patch("pipeline.ingestion.yt_dlp.YoutubeDL", return_value=make_ydl_mock(error=error)):
            with pytest.raises(RuntimeError, match="yt-dlp failed"):
                validate_source("https://www.youtube.com/watch?v=invalid")

    def test_duration_over_limit_raises(self):
        meta = {"title": "Long", "duration": 3700, "uploader": "Artist"}
        with patch("pipeline.ingestion.yt_dlp.YoutubeDL", return_value=make_ydl_mock(meta)):
            with pytest.raises(ValueError, match="60-minute limit"):
                validate_source("https://www.youtube.com/watch?v=long")

    def test_duration_exactly_at_limit_is_allowed(self):
        meta = {"title": "Edge", "duration": 3600, "uploader": "Artist"}
        with patch("pipeline.ingestion.yt_dlp.YoutubeDL", return_value=make_ydl_mock(meta)):
            result = validate_source("https://www.youtube.com/watch?v=edge")
        assert result["duration_sec"] == 3600

    def test_direct_audio_url_bypasses_host_check(self):
        meta = {"title": "Direct", "duration": 120, "uploader": "Unknown"}
        with patch("pipeline.ingestion.yt_dlp.YoutubeDL", return_value=make_ydl_mock(meta)):
            result = validate_source("https://cdn.example.com/track.mp3")
        assert result["title"] == "Direct"

    def test_missing_duration_does_not_raise(self):
        meta = {"title": "No Duration", "uploader": "Artist"}  # no "duration" key
        with patch("pipeline.ingestion.yt_dlp.YoutubeDL", return_value=make_ydl_mock(meta)):
            result = validate_source("https://www.youtube.com/watch?v=nodur")
        assert result["duration_sec"] == 0

    def test_does_not_pin_android_player_client(self):
        """Regression: forcing player_client=android limits YouTube to ~5 fragile
        formats and triggers GVS PO-token / HTTP 403 failures. The probe must let
        yt-dlp pick its default clients so the MCP works out-of-the-box locally."""
        meta = {"title": "T", "duration": 60, "uploader": "U"}
        with patch(
            "pipeline.ingestion.yt_dlp.YoutubeDL", return_value=make_ydl_mock(meta)
        ) as mock_cls:
            validate_source("https://www.youtube.com/watch?v=test")
        opts = mock_cls.call_args.args[0]
        clients = opts.get("extractor_args", {}).get("youtube", {}).get("player_client", [])
        assert "android" not in clients


# ── validate_url_format ───────────────────────────────────────────────────────


from pipeline.ingestion import validate_url_format


class TestValidateUrlFormat:
    def test_accepts_youtube(self):
        validate_url_format("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_accepts_youtu_be(self):
        validate_url_format("https://youtu.be/dQw4w9WgXcQ")

    def test_accepts_direct_audio(self):
        validate_url_format("https://example.com/track.mp3")

    def test_rejects_no_scheme(self):
        with pytest.raises(ValueError, match="Invalid URL format"):
            validate_url_format("www.youtube.com/watch?v=abc")

    def test_rejects_unsupported_domain(self):
        with pytest.raises(ValueError, match="Unsupported source"):
            validate_url_format("https://vimeo.com/123456")

    def test_rejects_random_string(self):
        with pytest.raises(ValueError, match="Invalid URL format"):
            validate_url_format("not a url at all")
