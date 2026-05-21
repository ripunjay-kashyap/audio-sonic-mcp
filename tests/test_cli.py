import os
import numpy as np
import sys
import json
import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

# Setup path bootstrap for tests
VENV_SCRIPTS = str(Path(__file__).parent.parent / ".venv" / "Scripts")
FFMPEG_BIN = (
    r"C:\Users\ROOP\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1-full_build\bin"
)
os.environ["PATH"] = (
    VENV_SCRIPTS + os.pathsep + FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")
)

from pipeline.ingestion import validate_file_path, validate_file_source
from pipeline.converter import convert_to_wav
from pipeline.separator import separate_stems
from pipeline.analyzer import analyze_audio
from pipeline.vectorizer import generate_vibe_vector
from pipeline.assembler import assemble_payload


class TestCLIValidation:
    def test_validate_file_path_valid(self, synthetic_stereo_wav):
        # Should not raise any error
        validate_file_path(str(synthetic_stereo_wav))

    def test_validate_file_path_nonexistent(self):
        with pytest.raises(ValueError, match="File does not exist"):
            validate_file_path("non_existent_file.mp3")

    def test_validate_file_path_directory(self, tmp_path):
        with pytest.raises(ValueError, match="Not a regular file"):
            validate_file_path(str(tmp_path))

    def test_validate_file_path_unsupported_ext(self, tmp_path):
        bad_file = tmp_path / "song.txt"
        bad_file.write_text("not a song")
        with pytest.raises(ValueError, match="Unsupported audio extension"):
            validate_file_path(str(bad_file))

    def test_validate_file_source_success(self, synthetic_stereo_wav):
        info = validate_file_source(str(synthetic_stereo_wav))
        assert info["title"] == synthetic_stereo_wav.stem
        assert info["uploader"] == "local file"
        assert info["duration_sec"] > 0
        assert info["source_type"] == "file"
        assert info["source_path"] == str(synthetic_stereo_wav.resolve())

    def test_validate_file_source_duration_limit(self, synthetic_stereo_wav):
        with patch.dict(os.environ, {"FILE_MAX_DURATION_SEC": "1"}):
            # The synthetic file is 4s, which exceeds 1s limit
            with pytest.raises(ValueError, match="exceeds 1.0s limit"):
                validate_file_source(str(synthetic_stereo_wav))


class TestCLIConverter:
    def test_convert_to_wav_with_out_dir(self, synthetic_stereo_wav, tmp_path):
        out_dir = tmp_path / "job_dir"
        wav_path = convert_to_wav(synthetic_stereo_wav, out_dir=out_dir)
        assert wav_path == out_dir / "input.wav"
        assert wav_path.exists()
        # Verify original file is untouched and not deleted
        assert synthetic_stereo_wav.exists()


class TestCLIFullSongPipeline:
    def test_full_song_stages(self, audio_wav):
        # Tests that analyzer runs over full song under HPSS fallback
        # (Demucs omitted/None for speed)
        features = analyze_audio(audio_wav, stems_dir=None, full_song=True)
        assert "bpm" in features
        assert "key" in features
        assert features["bpm"] > 0
        
        vibe = generate_vibe_vector(audio_wav, full_song=True)
        assert len(vibe) == 512

        payload = assemble_payload(
            job_id="test_full_song",
            features=features,
            vibe_vector=vibe,
            inference_time=0.5,
            cpu_samples=[],
            source_info={
                "title": audio_wav.stem,
                "uploader": "local file",
                "duration_sec": 4.0,
                "source_type": "file",
                "source_path": str(audio_wav.resolve())
            }
        )
        assert payload["header"]["status"] == "success"
        assert payload["header"]["source_metadata"]["source_type"] == "file"
        assert payload["header"]["source_metadata"]["source_path"] == str(audio_wav.resolve())


class TestCLISubprocess:
    def test_cli_execution_success(self, synthetic_stereo_wav, tmp_path):
        # Run analyze_file.py as a subprocess on synthetic file
        cli_script = Path(__file__).parent.parent / "analyze_file.py"
        out_json = tmp_path / "result.json"
        
        cmd = [
            sys.executable,
            str(cli_script),
            str(synthetic_stereo_wav),
            "--out", str(out_json),
            "--job-id", "test_cli_sub"
        ]
        
        # Override KEEP_JOB_FILES to keep cleanup active and tested
        env = os.environ.copy()
        if "KEEP_JOB_FILES" in env:
            del env["KEEP_JOB_FILES"]

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        
        # Check success
        assert res.returncode == 0, f"CLI failed: {res.stderr}"
        
        # Print output logic
        stdout_content = res.stdout.strip()
        parsed = json.loads(stdout_content)
        assert parsed["header"]["job_id"] == "test_cli_sub"
        assert parsed["header"]["source_metadata"]["source_type"] == "file"
        
        # Check out file was written
        assert out_json.exists()
        assert json.loads(out_json.read_text(encoding="utf-8")) == parsed


class TestRankTags:
    def test_orders_by_similarity_and_applies_floor(self):
        from pipeline.vectorizer import _rank_tags
        text = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        labels = ["a", "b", "c"]
        audio = np.array([0.3, 0.9, -0.2], dtype=float)  # sims: a=.31 b=.92 c=-.21
        out = _rank_tags(audio, text, labels, top_n=5, floor=0.0)
        assert out == ["b", "a"]  # c dropped by floor 0.0 (negative sim)

    def test_returns_top1_when_all_below_floor(self):
        from pipeline.vectorizer import _rank_tags
        text = np.eye(3)
        labels = ["a", "b", "c"]
        audio = np.array([0.1, 0.2, 0.05], dtype=float)
        out = _rank_tags(audio, text, labels, top_n=5, floor=0.9)
        assert out == ["b"]  # nothing clears 0.9 -> top-1 returned

    def test_respects_top_n_cap(self):
        from pipeline.vectorizer import _rank_tags
        text = np.eye(4)
        labels = ["a", "b", "c", "d"]
        audio = np.array([0.4, 0.3, 0.2, 0.1], dtype=float)
        out = _rank_tags(audio, text, labels, top_n=2, floor=0.0)
        assert out == ["a", "b"]

