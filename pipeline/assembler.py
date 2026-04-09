"""
Final Stage — Context Assembly
Bundles all pipeline outputs into the canonical JSON-RPC payload.
"""

import os
import time
import uuid
from pathlib import Path
from typing import Any


def assemble_payload(
    job_id: str,
    stems_dir: Path,
    stem_files: list[str],
    sdr: float,
    features: dict[str, Any],
    vibe_vector: list[float],
    inference_time: float,
    cpu_samples: list[float],
    source_info: dict,
) -> dict:
    """
    Assembles the canonical JSON response payload.
    """
    cpu_avg = _cpu_avg(cpu_samples)
    confidence = _confidence_score(sdr, features)

    return {
        "header": {
            "job_id": job_id,
            "status": "success",
            "confidence_score": confidence,
            "source": {
                "title": source_info.get("title"),
                "uploader": source_info.get("uploader"),
                "duration_sec": source_info.get("duration_sec"),
                "url": source_info.get("webpage_url"),
            },
        },
        "stems_metadata": {
            "local_root": str(stems_dir) + "/",
            "files": stem_files,
            "sdr_ratio": sdr,
        },
        "sonic_signature": {
            "bpm": features["bpm"],
            "key": features["key"],
            "vibe_vector": vibe_vector,
            "production_profile": {
                "vocal_presence": features["vocal_presence_label"],
                "drum_transient_punch": features["transient_punch"],
                "stereo_width": features["stereo_width_label"],
                "dominant_freq_peaks_hz": features.get("freq_peaks_hz", {}),
            },
        },
        "telemetry": {
            "cpu_usage_avg": f"{cpu_avg:.0f}%" if cpu_samples else "n/a",
            "inference_time_sec": round(inference_time, 2),
        },
    }


def _cpu_avg(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return sum(samples) / len(samples)


def _confidence_score(sdr: float, features: dict) -> float:
    """
    Heuristic confidence based on SDR quality and feature extraction success.
    SDR of 8+ → high confidence; below 4 → low.
    """
    sdr_score = min(max((sdr - 2) / 10.0, 0.0), 1.0)

    feature_score = 1.0
    if features.get("bpm", 0) == 0:
        feature_score -= 0.15
    if features.get("key") in (None, "Unknown"):
        feature_score -= 0.10

    combined = (sdr_score * 0.6 + feature_score * 0.4)
    return round(min(max(combined, 0.0), 1.0), 2)
