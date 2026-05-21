import os
import numpy as np
import sys
import json
import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

# FFmpeg is expected on PATH; honor an optional FFMPEG_BIN override like the CLI.
_VENV_SCRIPTS = str(Path(__file__).parent.parent / ".venv" / "Scripts")
_extra = [_VENV_SCRIPTS]
if os.environ.get("FFMPEG_BIN"):
    _extra.append(os.environ["FFMPEG_BIN"])
os.environ["PATH"] = os.pathsep.join(_extra + [os.environ.get("PATH", "")])

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
        assert features["bpm"] >= 0
        
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


class TestSelectTagsPerAxis:
    def test_spans_axes_in_order(self):
        from pipeline.vectorizer import _select_tags_per_axis
        vocab = [
            ("m1", "mood"), ("m2", "mood"), ("m3", "mood"),
            ("e1", "energy"), ("e2", "energy"),
            ("g1", "genre"), ("g2", "genre"),
            ("t1", "texture"), ("t2", "texture"),
        ]
        text = np.eye(len(vocab))
        audio = np.zeros(len(vocab))
        # mood: m2>m1>m3 ; energy: e1>e2 ; genre: g2>g1 ; texture: t1>t2
        audio[1], audio[0], audio[2] = 0.9, 0.8, 0.1
        audio[3], audio[4] = 0.7, 0.2
        audio[6], audio[5] = 0.6, 0.3
        audio[7], audio[8] = 0.5, 0.2
        counts = {"mood": 2, "energy": 1, "genre": 1, "texture": 1}
        out = _select_tags_per_axis(audio, text, vocab, counts, floor=0.0)
        # 2 mood + 1 energy + 1 genre + 1 texture, in axis order, within-axis by sim
        assert out == ["m2", "m1", "e1", "g2", "t1"]

    def test_returns_top1_when_all_below_floor(self):
        from pipeline.vectorizer import _select_tags_per_axis
        vocab = [("m1", "mood"), ("g1", "genre")]
        text = np.eye(2)
        audio = np.array([0.1, 0.2], dtype=float)  # g1 highest overall
        out = _select_tags_per_axis(audio, text, vocab, {"mood": 1, "genre": 1}, floor=0.9)
        assert out == ["g1"]  # nothing clears 0.9 -> single global top word

    def test_floor_drops_weak_axis_pick(self):
        from pipeline.vectorizer import _select_tags_per_axis
        vocab = [("m1", "mood"), ("g1", "genre")]
        text = np.eye(2)
        audio = np.array([0.05, 0.8], dtype=float)  # mood weak, genre strong
        out = _select_tags_per_axis(audio, text, vocab, {"mood": 1, "genre": 1}, floor=0.5)
        assert out == ["g1"]  # mood pick dropped by floor; genre kept


class TestGenerateVibeTags:
    def test_returns_none_when_clap_unavailable(self, monkeypatch, audio_wav):
        from pipeline import vectorizer
        monkeypatch.setattr(vectorizer, "_clap_tag_embeddings", lambda *a, **k: None)
        assert vectorizer.generate_vibe_tags(audio_wav, full_song=True) is None

    def test_returns_tags_spanning_axes_when_clap_available(self, monkeypatch, audio_wav):
        from pipeline import vectorizer
        vocab = vectorizer.VIBE_TAG_VOCAB
        n = len(vocab)
        text = np.eye(n)
        audio = np.zeros(n)
        mood_idx = next(i for i, (w, ax) in enumerate(vocab) if ax == "mood")
        genre_idx = next(i for i, (w, ax) in enumerate(vocab) if ax == "genre")
        audio[mood_idx] = 1.0   # boost one mood word
        audio[genre_idx] = 0.9  # and one genre word
        monkeypatch.setattr(
            vectorizer, "_clap_tag_embeddings", lambda *a, **k: (audio, text)
        )
        out = vectorizer.generate_vibe_tags(audio_wav, full_song=True)
        assert isinstance(out, list) and len(out) >= 1
        assert out[0] == vocab[mood_idx][0]          # mood comes first (axis order)
        assert vocab[genre_idx][0] in out            # genre present -> spans axes
        assert len(out) <= sum(vectorizer.VIBE_TAG_AXIS_COUNTS.values())


class TestAssembleVibeTags:
    def _features(self):
        return {
            "bpm": 120.0, "key": "C Major", "mode_confidence": 0.7,
            "vocal_presence_label": "present", "transient_punch": 0.4,
            "stereo_width_label": "wide", "freq_peaks_hz": {"harmonic": [100.0]},
        }

    def test_includes_vibe_tags_when_provided(self):
        from pipeline.assembler import assemble_payload
        p = assemble_payload("j", self._features(), [0.0] * 512, 1.0, [],
                             {"title": "t"}, vibe_tags=["dark", "jazz"])
        assert p["sonic_signature"]["vibe_tags"] == ["dark", "jazz"]

    def test_omits_vibe_tags_when_not_provided(self):
        from pipeline.assembler import assemble_payload
        p = assemble_payload("j", self._features(), [0.0] * 512, 1.0, [],
                             {"title": "t"})
        assert "vibe_tags" not in p["sonic_signature"]
        assert p["sonic_signature"]["bpm"] == 120.0  # rest of payload intact


class TestPrintSummary:
    def _payload(self, vibe_tags=["aggressive", "dark", "hip-hop"]):
        return {
            "header": {
                "confidence_score": 0.78,
                "source_metadata": {"title": "input", "duration_sec": 102.34},
            },
            "sonic_signature": {
                "bpm": 153.85, "bpm_variable": False, "bpm_range": None,
                "key": "G Major", "mode_confidence": 0.63, "key_variable": True,
                "key_map": [
                    {"start_sec": 0.0, "end_sec": 30.0, "key": "G Major"},
                    {"start_sec": 30.0, "end_sec": 90.0, "key": "G Phrygian"},
                ],
                "vibe_vector": [0.0] * 512,
                "vibe_tags": vibe_tags,
                "production_profile": {
                    "vocal_presence": "forward", "transient_punch": 0.325,
                    "stereo_width": "narrow",
                    "dominant_freq_peaks_hz": {"harmonic": [49.7, 49.4], "percussive": []},
                },
            },
            "telemetry": {"inference_time_sec": 322.81},
        }

    def test_renders_core_fields(self, capsys):
        from analyze_file import _print_summary
        _print_summary(self._payload())
        out = capsys.readouterr().out
        assert "SONIC SIGNATURE" in out
        assert "153" in out and "BPM" in out
        assert "G Major" in out and "shifts to G Phrygian" in out
        assert "aggressive · dark · hip-hop" in out
        assert "forward" in out
        assert "78%" in out
        assert "0.0," not in out  # the 512 array is NOT printed

    def test_unavailable_vibe_when_tags_none(self, capsys):
        from analyze_file import _print_summary
        _print_summary(self._payload(vibe_tags=None))
        out = capsys.readouterr().out
        assert "unavailable" in out


class TestCLIFlags:
    def _run(self, wav, tmp_path, *extra, out=None):
        cli = Path(__file__).parent.parent / "analyze_file.py"
        cmd = [sys.executable, str(cli), str(wav), *extra]
        if out:
            cmd += ["--out", str(out)]
        env = os.environ.copy()
        env.pop("KEEP_JOB_FILES", None)
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, env=env)

    def test_summary_prints_digest_not_json(self, synthetic_stereo_wav, tmp_path):
        out_json = tmp_path / "r.json"
        res = self._run(synthetic_stereo_wav, tmp_path, "--summary",
                        "--job-id", "test_cli_sum", out=out_json)
        assert res.returncode == 0, res.stderr
        assert "SONIC SIGNATURE" in res.stdout
        assert "BPM" in res.stdout
        assert '"vibe_vector"' not in res.stdout  # no JSON / no big array
        # --out still has the COMPLETE JSON including the 512-vector
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert len(data["sonic_signature"]["vibe_vector"]) == 512

    def test_no_vector_strips_array_from_json(self, synthetic_stereo_wav, tmp_path):
        res = self._run(synthetic_stereo_wav, tmp_path, "--no-vector",
                        "--job-id", "test_cli_nv")
        assert res.returncode == 0, res.stderr
        parsed = json.loads(res.stdout)
        assert "vibe_vector" not in parsed["sonic_signature"]
        assert "bpm" in parsed["sonic_signature"]
