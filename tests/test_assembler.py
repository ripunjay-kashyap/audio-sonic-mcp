"""Tests for pipeline/assembler.py — payload assembly and confidence scoring."""

from pathlib import Path

from pipeline.assembler import _confidence_score, _cpu_avg, assemble_payload


# ── _cpu_avg ──────────────────────────────────────────────────────────────────


class TestCpuAvg:
    def test_empty_returns_zero(self):
        assert _cpu_avg([]) == 0.0

    def test_single_value(self):
        assert _cpu_avg([55.0]) == 55.0

    def test_average_of_multiple(self):
        assert _cpu_avg([10.0, 20.0, 30.0]) == 20.0


# ── _confidence_score ─────────────────────────────────────────────────────────


class TestConfidenceScore:
    def _full_features(self):
        return {"bpm": 128.0, "key": "A Minor"}

    def test_high_sdr_full_features_gives_high_score(self):
        assert _confidence_score(12.0, self._full_features()) > 0.8

    def test_low_sdr_penalizes_score(self):
        high = _confidence_score(12.0, self._full_features())
        low = _confidence_score(1.0, self._full_features())
        assert low < high

    def test_missing_bpm_penalizes(self):
        base = _confidence_score(8.0, {"bpm": 120.0, "key": "C Major"})
        no_bpm = _confidence_score(8.0, {"bpm": 0, "key": "C Major"})
        assert no_bpm < base

    def test_unknown_key_penalizes(self):
        base = _confidence_score(8.0, {"bpm": 120.0, "key": "C Major"})
        unknown = _confidence_score(8.0, {"bpm": 120.0, "key": "Unknown"})
        assert unknown < base

    def test_none_key_penalizes(self):
        base = _confidence_score(8.0, {"bpm": 120.0, "key": "C Major"})
        none = _confidence_score(8.0, {"bpm": 120.0, "key": None})
        assert none < base

    def test_score_always_between_0_and_1(self):
        for sdr, features in [
            (999.0, {"bpm": 999.0, "key": "C Major"}),  # absurdly high
            (-999.0, {"bpm": 0, "key": None}),  # worst case
        ]:
            score = _confidence_score(sdr, features)
            assert 0.0 <= score <= 1.0


# ── assemble_payload ──────────────────────────────────────────────────────────


class TestAssemblePayload:
    def _args(self, tmp_path: Path) -> dict:
        return dict(
            job_id="job_test_001",
            stems_dir=tmp_path / "stems",
            stem_files=["vocals.wav", "drums.wav", "bass.wav", "other.wav"],
            sdr=8.5,
            features={
                "bpm": 128.0,
                "key": "A Minor",
                "mode_confidence": 0.76,
                "transient_punch": 0.75,
                "freq_peaks_hz": {"vocals": [440.0], "drums": [80.0]},
                "stereo_width_label": "wide",
                "vocal_presence_label": "balanced",
            },
            vibe_vector=[0.1] * 512,
            inference_time=45.3,
            cpu_samples=[60.0, 72.0, 68.0],
            source_info={
                "title": "Test Track",
                "uploader": "Test Artist",
                "duration_sec": 210,
                "webpage_url": "https://www.youtube.com/watch?v=test",
            },
        )

    def test_top_level_keys_present(self, tmp_path):
        payload = assemble_payload(**self._args(tmp_path))
        assert set(payload.keys()) == {
            "header",
            "stems_metadata",
            "sonic_signature",
            "telemetry",
        }

    def test_header_job_id_and_status(self, tmp_path):
        payload = assemble_payload(**self._args(tmp_path))
        assert payload["header"]["job_id"] == "job_test_001"
        assert payload["header"]["status"] == "success"

    def test_header_source_fields(self, tmp_path):
        payload = assemble_payload(**self._args(tmp_path))
        src = payload["header"]["source_metadata"]
        assert src["title"] == "Test Track"
        assert src["uploader"] == "Test Artist"
        assert src["duration_sec"] == 210

    def test_genre_hint_present(self, tmp_path):
        args = self._args(tmp_path)
        args["source_info"]["genre_hint"] = "Electronic"
        payload = assemble_payload(**args)
        assert payload["header"]["source_metadata"]["genre_hint"] == "Electronic"

    def test_genre_hint_none_when_absent(self, tmp_path):
        payload = assemble_payload(**self._args(tmp_path))
        assert payload["header"]["source_metadata"]["genre_hint"] is None

    def test_stems_metadata_fields(self, tmp_path):
        payload = assemble_payload(**self._args(tmp_path))
        meta = payload["stems_metadata"]
        assert meta["sdr_ratio"] == 8.5
        assert set(meta["files"]) == {
            "vocals.wav",
            "drums.wav",
            "bass.wav",
            "other.wav",
        }
        assert "stems" in meta["local_root"]

    def test_sonic_signature_bpm_and_key(self, tmp_path):
        payload = assemble_payload(**self._args(tmp_path))
        sig = payload["sonic_signature"]
        assert sig["bpm"] == 128.0
        assert sig["key"] == "A Minor"
        assert sig["mode_confidence"] == 0.76

    def test_vibe_vector_length(self, tmp_path):
        payload = assemble_payload(**self._args(tmp_path))
        assert len(payload["sonic_signature"]["vibe_vector"]) == 512

    def test_production_profile_keys(self, tmp_path):
        payload = assemble_payload(**self._args(tmp_path))
        profile = payload["sonic_signature"]["production_profile"]
        assert profile["vocal_presence"] == "balanced"
        assert profile["drum_transient_punch"] == 0.75
        assert profile["stereo_width"] == "wide"

    def test_telemetry_cpu_average(self, tmp_path):
        payload = assemble_payload(**self._args(tmp_path))
        assert payload["telemetry"]["cpu_usage_avg"] == "67%"  # avg(60,72,68)
        assert payload["telemetry"]["inference_time_sec"] == 45.3

    def test_telemetry_no_cpu_samples(self, tmp_path):
        args = self._args(tmp_path)
        args["cpu_samples"] = []
        payload = assemble_payload(**args)
        assert payload["telemetry"]["cpu_usage_avg"] == "n/a"

    def test_confidence_score_is_float_in_range(self, tmp_path):
        payload = assemble_payload(**self._args(tmp_path))
        score = payload["header"]["confidence_score"]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
