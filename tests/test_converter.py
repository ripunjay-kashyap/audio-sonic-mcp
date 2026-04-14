"""Tests for pipeline/converter.py — FFmpeg WAV conversion."""

from unittest.mock import MagicMock, patch

import pytest

from pipeline.converter import convert_to_wav


class TestConvertToWav:
    def test_missing_input_raises_file_not_found(self, tmp_path):
        fake = tmp_path / "nonexistent.m4a"
        with pytest.raises(FileNotFoundError, match="Raw audio not found"):
            convert_to_wav(fake)

    def test_existing_wav_skips_ffmpeg(self, tmp_path, synthetic_stereo_wav):
        # Pre-create input.wav next to the raw file
        existing_wav = synthetic_stereo_wav.parent / "input.wav"
        existing_wav.write_bytes(b"already converted")

        with patch("pipeline.converter.subprocess.run") as mock_run:
            result = convert_to_wav(synthetic_stereo_wav)

        mock_run.assert_not_called()
        assert result == existing_wav

    def test_successful_conversion_returns_wav_path(
        self, tmp_path, synthetic_stereo_wav
    ):
        wav_output = synthetic_stereo_wav.parent / "input.wav"

        mock_result = MagicMock()
        mock_result.returncode = 0

        def fake_ffmpeg(*args, **kwargs):
            wav_output.write_bytes(b"fake pcm data")
            return mock_result

        with patch("pipeline.converter.subprocess.run", side_effect=fake_ffmpeg):
            result = convert_to_wav(synthetic_stereo_wav)

        assert result == wav_output
        assert result.exists()

    def test_ffmpeg_failure_raises_runtime_error(self, tmp_path, synthetic_stereo_wav):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Error: codec not found\n" * 10

        with patch("pipeline.converter.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="FFmpeg conversion failed"):
                convert_to_wav(synthetic_stereo_wav)

    def test_ffmpeg_command_contains_expected_flags(
        self, tmp_path, synthetic_stereo_wav
    ):
        wav_output = synthetic_stereo_wav.parent / "input.wav"
        mock_result = MagicMock()
        mock_result.returncode = 0
        captured = {}

        def capture_cmd(*args, **kwargs):
            captured["cmd"] = args[0]
            wav_output.write_bytes(b"data")
            return mock_result

        with patch("pipeline.converter.subprocess.run", side_effect=capture_cmd):
            convert_to_wav(synthetic_stereo_wav)

        cmd = captured["cmd"]
        assert "ffmpeg" in cmd
        assert "pcm_s16le" in cmd  # 16-bit PCM codec
        assert "44100" in cmd  # target sample rate
        assert "soxr" in " ".join(cmd)  # high-quality resampler
