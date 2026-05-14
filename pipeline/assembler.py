"""
Final Stage — Context Assembly
Bundles all pipeline outputs into the canonical JSON-RPC payload.
"""

from typing import Any


def assemble_payload(
    job_id: str,
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
    confidence = _confidence_score(features)

    return {
        "header": {
            "job_id": job_id,
            "status": "success",
            "confidence_score": confidence,
            "source_metadata": {
                "title": source_info.get("title"),
                "uploader": source_info.get("uploader"),
                "duration_sec": source_info.get("duration_sec"),
                "url": source_info.get("webpage_url"),
                "genre_hint": source_info.get("genre_hint"),
            },
        },
        "sonic_signature": {
            "bpm": features["bpm"],
            "bpm_variable": features.get("bpm_variable", False),
            "bpm_range": features.get("bpm_range"),
            "key": features["key"],
            "mode_confidence": features.get("mode_confidence"),
            "key_ambiguous": features.get("key_ambiguous", False),
            "key_variable": features.get("key_variable", False),
            "key_map": features.get("key_map", []),
            "vibe_vector": vibe_vector,
            "production_profile": {
                "vocal_presence": features["vocal_presence_label"],
                "transient_punch": features["transient_punch"],
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


def _confidence_score(features: dict) -> float:
    """
    Heuristic confidence based on mode_confidence and feature extraction success.
    mode_confidence is the primary signal (CQT chroma correlation quality).
    """
    mode_conf = float(features.get("mode_confidence") or 0.0)

    feature_score = 1.0
    if features.get("bpm", 0) == 0:
        feature_score -= 0.15
    if features.get("key") in (None, "Unknown"):
        feature_score -= 0.10

    combined = mode_conf * 0.6 + feature_score * 0.4
    return round(min(max(combined, 0.0), 1.0), 2)
