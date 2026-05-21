#!/usr/bin/env python3
"""
Local File Analysis CLI for audio-sonic-mcp.
Analyzes a local audio file synchronously and prints the sonic-signature JSON.
"""

import sys
import os
import argparse
import time
import uuid
import json
import logging
import shutil
from pathlib import Path

# venv-installed tools (yt-dlp, demucs) resolve when run via the venv Python
# without an activated shell. FFmpeg is expected on PATH (see README); set the
# FFMPEG_BIN env var to its bin directory if it lives somewhere off PATH.
VENV_SCRIPTS = str(Path(__file__).parent / ".venv" / "Scripts")
_extra_paths = [VENV_SCRIPTS]
if os.environ.get("FFMPEG_BIN"):
    _extra_paths.append(os.environ["FFMPEG_BIN"])
os.environ["PATH"] = os.pathsep.join(_extra_paths + [os.environ.get("PATH", "")])

# Setup logging - redirecting logs to stderr to leave stdout clean for JSON
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("analyze_file")

from pipeline.ingestion import validate_file_path, validate_file_source, save_metadata
from pipeline.converter import convert_to_wav
from pipeline.separator import load_demucs_model, separate_stems
from pipeline.analyzer import analyze_audio
from pipeline.vectorizer import generate_vibe_vector, generate_vibe_tags
from pipeline.assembler import assemble_payload

def _cleanup_job_artifacts(job_dir: Path, keep_job_files: bool = False) -> None:
    if keep_job_files or os.environ.get("KEEP_JOB_FILES", "").lower() in ("1", "true", "yes"):
        logger.info("Cleanup: skipped (keep requested) for %s", job_dir.name)
        return

    import shutil
    removed_bytes = 0
    if not job_dir.exists():
        return
    for path in job_dir.iterdir():
        if path.name == "metadata.json":
            continue
        try:
            if path.is_dir():
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                shutil.rmtree(path)
                removed_bytes += size
            else:
                removed_bytes += path.stat().st_size
                path.unlink()
        except Exception as e:
            logger.warning("Cleanup failed on %s: %s", path, e)
    logger.info("Cleanup: freed %.1f MB from %s", removed_bytes / 1_048_576, job_dir.name)


def _fmt_mmss(seconds: float) -> str:
    s = int(round(seconds or 0))
    return f"{s // 60}:{s % 60:02d}"


def _punch_label(v: float) -> str:
    if v < 0.33:
        return "low"
    if v < 0.66:
        return "moderate"
    return "high"


def _print_summary(payload: dict) -> None:
    """Print a compact, musician-friendly digest (no JSON, no 512-float vector)."""
    def _safe_print(text: str) -> None:
        try:
            print(text)
        except UnicodeEncodeError:
            t = text.replace("\U0001F3B5", "").replace("—", "-").replace("·", ".").replace("–", "-")
            try:
                print(t)
            except UnicodeEncodeError:
                print(t.encode("ascii", errors="replace").decode("ascii"))

    sm = payload["header"]["source_metadata"]
    ss = payload["sonic_signature"]
    pp = ss["production_profile"]

    title = sm.get("title") or "audio"
    dur = _fmt_mmss(sm.get("duration_sec") or 0)

    if ss.get("bpm_variable") and ss.get("bpm_range"):
        lo, hi = ss["bpm_range"]
        tempo = f"{ss['bpm']:.1f} BPM  (variable {lo:.0f}–{hi:.0f})"
    else:
        tempo = f"{ss['bpm']:.1f} BPM  (steady)"

    base_key = ss.get("key", "Unknown")
    key_line = base_key
    if ss.get("key_variable") and ss.get("key_map"):
        shift = next((seg for seg in ss["key_map"] if seg.get("key") != base_key), None)
        if shift:
            key_line = f"{base_key}  ·  shifts to {shift['key']} @{_fmt_mmss(shift['start_sec'])}"
    conf = ss.get("mode_confidence")
    if conf is not None:
        key_line += f"   (confidence {round(conf * 100)}%)"

    tags = ss.get("vibe_tags")
    vibe = " · ".join(tags) if tags else "(unavailable — CLAP not installed)"

    punch = pp.get("transient_punch")
    punch_str = f"{punch:.2f}  ({_punch_label(punch)})" if punch is not None else "n/a"
    harm = (pp.get("dominant_freq_peaks_hz") or {}).get("harmonic") or []
    low_end = f"~{round(sum(harm) / len(harm))} Hz dominant" if harm else "n/a"

    overall = round((payload["header"].get("confidence_score") or 0) * 100)
    elapsed = _fmt_mmss(payload.get("telemetry", {}).get("inference_time_sec") or 0)

    _safe_print(f"\n\U0001F3B5 SONIC SIGNATURE — {title}  ({dur})\n")
    _safe_print(f"  TEMPO    {tempo}")
    _safe_print(f"  KEY      {key_line}")
    _safe_print(f"  VIBE     {vibe}\n")
    _safe_print("  PRODUCTION")
    _safe_print(f"    Vocals     {pp.get('vocal_presence', 'n/a')}")
    _safe_print(f"    Punch      {punch_str}")
    _safe_print(f"    Stereo     {pp.get('stereo_width', 'n/a')}")
    _safe_print(f"    Low end    {low_end}\n")
    _safe_print(f"  Overall confidence: {overall}%   ·   analyzed in {elapsed}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a local audio file and output its sonic signature."
    )
    parser.add_argument("path", help="Path to the local audio file to analyze.")
    parser.add_argument("--out", "-o", help="Optional path to write the output JSON result.")
    parser.add_argument("--keep", "-k", action="store_true", help="Keep intermediate files in job directory.")
    parser.add_argument("--job-id", "-j", help="Custom Job ID for the run.")
    parser.add_argument("--summary", "-s", action="store_true",
                        help="Print a human-readable digest instead of JSON.")
    parser.add_argument("--no-vector", dest="no_vector", action="store_true",
                        help="Print full JSON but omit the 512-float vibe_vector array.")
    
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.stderr.write(
            "Error: FFmpeg not found on PATH. Install FFmpeg, or set the "
            "FFMPEG_BIN environment variable to its bin directory.\n"
        )
        sys.exit(1)

    start_time = time.time()
    
    try:
        # 1. Validate file exists and is valid format
        validate_file_path(args.path)
        source_info = validate_file_source(args.path)
    except (ValueError, RuntimeError) as e:
        sys.stderr.write(f"Validation Error: {e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Error during validation: {e}\n")
        sys.exit(1)

    duration = source_info["duration_sec"]
    # Estimate time: ~4x CPU time, but faster if on GPU
    est_sec = int(duration * 4)
    logger.info("Starting local file analysis:")
    logger.info("  File: %s", args.path)
    logger.info("  Duration: %.2fs", duration)
    logger.info("  Estimated CPU processing time: %ds (faster on GPU)", est_sec)

    # 2. Setup job directory
    job_id = args.job_id or f"file_{uuid.uuid4().hex[:8]}"
    JOBS_ROOT = Path(os.environ.get("JOBS_ROOT", Path(__file__).parent / "jobs"))
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        save_metadata(job_dir, source_info)

        # 3. Convert to WAV in-place into job_dir
        logger.info("Converting raw audio to normalized WAV...")
        wav_path = convert_to_wav(Path(args.path), out_dir=job_dir)
        
        # 4. Attempt stem separation with Demucs (pre-warmed/loaded)
        logger.info("Loading Demucs model...")
        try:
            load_demucs_model()
        except Exception as e:
            logger.warning("Could not load Demucs model (GPU/CPU package missing or error): %s. Falling back to HPSS.", e)

        logger.info("Running stem separation (full-song)...")
        stems_dir = separate_stems(wav_path, full_song=True)
        if stems_dir is None:
            logger.warning("Stem separation bypassed or failed. Falling back to HPSS in analyzer.")

        # 5. Analyze audio (full-song)
        logger.info("Running audio feature analysis (full-song)...")
        features = analyze_audio(wav_path, stems_dir=stems_dir, full_song=True)

        # 6. Generate vibe vector (full-song)
        logger.info("Generating vibe vector (full-song)...")
        vibe_vector = generate_vibe_vector(wav_path, full_song=True)

        logger.info("Generating vibe tags (CLAP zero-shot)...")
        vibe_tags = generate_vibe_tags(wav_path, full_song=True)

        # 7. Assemble canonical payload
        elapsed = time.time() - start_time
        payload = assemble_payload(
            job_id=job_id,
            features=features,
            vibe_vector=vibe_vector,
            inference_time=elapsed,
            cpu_samples=[],
            source_info=source_info,
            vibe_tags=vibe_tags,
        )

        # 8. Output: --summary prints the digest; --no-vector strips the array;
        #    default prints full JSON. --out ALWAYS writes the complete payload.
        full_json = json.dumps(payload, indent=2)
        if args.summary:
            _print_summary(payload)
        elif args.no_vector:
            trimmed = {
                **payload,
                "sonic_signature": {
                    k: v for k, v in payload["sonic_signature"].items()
                    if k != "vibe_vector"
                },
            }
            print(json.dumps(trimmed, indent=2))
        else:
            print(full_json)

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(full_json, encoding="utf-8")
            logger.info("Result written to %s", out_path)

        logger.info("Analysis completed successfully in %.2fs.", elapsed)

    except Exception as e:
        sys.stderr.write(f"Processing Error: {e}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        # 9. Cleanup
        _cleanup_job_artifacts(job_dir, keep_job_files=args.keep)


if __name__ == "__main__":
    main()
