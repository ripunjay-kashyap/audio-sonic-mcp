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
from pathlib import Path

# Ensure ffmpeg and venv-installed tools (yt-dlp, demucs) are on PATH
VENV_SCRIPTS = str(Path(__file__).parent / ".venv" / "Scripts")
FFMPEG_BIN = (
    r"C:\Users\ROOP\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1-full_build\bin"
)
os.environ["PATH"] = (
    VENV_SCRIPTS + os.pathsep + FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")
)

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
from pipeline.vectorizer import generate_vibe_vector
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


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a local audio file and output its sonic signature."
    )
    parser.add_argument("path", help="Path to the local audio file to analyze.")
    parser.add_argument("--out", "-o", help="Optional path to write the output JSON result.")
    parser.add_argument("--keep", "-k", action="store_true", help="Keep intermediate files in job directory.")
    parser.add_argument("--job-id", "-j", help="Custom Job ID for the run.")
    
    args = parser.parse_args()

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

        # 7. Assemble canonical payload
        elapsed = time.time() - start_time
        payload = assemble_payload(
            job_id=job_id,
            features=features,
            vibe_vector=vibe_vector,
            inference_time=elapsed,
            cpu_samples=[],
            source_info=source_info
        )

        # 8. Print JSON payload to stdout
        payload_str = json.dumps(payload, indent=2)
        print(payload_str)

        # Write to --out if requested
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(payload_str, encoding="utf-8")
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
