"""
Audio Stem Splitter MCP Server
Converts YouTube URLs into separated stems + sonic signature JSON.
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from pipeline.ingestion import validate_source
from pipeline.downloader import download_audio
from pipeline.converter import convert_to_wav
from pipeline.splitter import split_stems
from pipeline.analyzer import analyze_stems
from pipeline.vectorizer import generate_vibe_vector
from pipeline.assembler import assemble_payload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("audio-stem-mcp")

# ── Config ────────────────────────────────────────────────────────────────────
STEMS_ROOT = Path(os.environ.get("STEMS_ROOT", Path(__file__).parent / "stems"))
STEMS_ROOT.mkdir(parents=True, exist_ok=True)

app = FastMCP("audio-stem-mcp")

# ── In-memory job store ───────────────────────────────────────────────────────
JOB_STORE: dict[str, dict] = {}


# ── Tool handlers ─────────────────────────────────────────────────────────────

@app.tool()
async def split_audio(url: str, job_id: str = None, model: str = "htdemucs") -> str:
    """
    Downloads audio from a YouTube URL, splits it into stems 
    (vocals, drums, bass, other) using htdemucs, then extracts 
    BPM, key, transients, frequency peaks, and a CLAP vibe vector. 
    Returns a structured JSON payload with all metadata and local file paths.
    
    Args:
        url: YouTube URL of the track to process.
        job_id: Optional custom job ID. Auto-generated if omitted.
        model: Demucs model variant to use for separation. 
               Options: "htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra".
    """
    job_id = job_id or f"sig_{uuid.uuid4().hex[:8]}"

    logger.info("▶ Job %s | Starting pipeline for: %s", job_id, url)

    JOB_STORE[job_id] = {"status": "running", "started_at": time.time()}
    cpu_samples: list[float] = []
    t_start = time.perf_counter()

    try:
        # ── Stage 1: Validate ─────────────────────────────────────────────────
        logger.info("[1/6] Validating source …")
        source_info = await asyncio.to_thread(validate_source, url)

        # ── Stage 2: Download ─────────────────────────────────────────────────
        logger.info("[2/6] Downloading audio stream …")
        raw_audio_path = await asyncio.to_thread(
            download_audio, url, job_id, STEMS_ROOT
        )

        # ── Stage 3: Convert ──────────────────────────────────────────────────
        logger.info("[3/6] Converting to 44.1kHz WAV …")
        wav_path = await asyncio.to_thread(convert_to_wav, raw_audio_path)

        # ── Stage 4: Split ────────────────────────────────────────────────────
        logger.info("[4/6] Running htdemucs stem separation …")
        stems_dir, stem_files, sdr = await asyncio.to_thread(
            split_stems, wav_path, job_id, STEMS_ROOT, model
        )

        # ── Stage 5: Analyze ──────────────────────────────────────────────────
        logger.info("[5/6] Extracting BPM, key, transients, frequencies …")
        features = await asyncio.to_thread(analyze_stems, stems_dir, stem_files)

        # ── Stage 6: Vectorize ────────────────────────────────────────────────
        logger.info("[6/6] Generating CLAP vibe vector …")
        vibe_vector = await asyncio.to_thread(generate_vibe_vector, stems_dir, stem_files)

        # ── Assemble payload ──────────────────────────────────────────────────
        elapsed = time.perf_counter() - t_start
        payload = assemble_payload(
            job_id=job_id,
            stems_dir=stems_dir,
            stem_files=stem_files,
            sdr=sdr,
            features=features,
            vibe_vector=vibe_vector,
            inference_time=elapsed,
            cpu_samples=cpu_samples,
            source_info=source_info,
        )

        JOB_STORE[job_id] = {"status": "success", "payload": payload}
        logger.info("✓ Job %s completed in %.1fs", job_id, elapsed)

        return json.dumps(payload, indent=2)

    except Exception as exc:
        elapsed = time.perf_counter() - t_start
        error_payload = {
            "header": {"job_id": job_id, "status": "error", "confidence_score": 0.0},
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "telemetry": {"inference_time_sec": round(elapsed, 2)},
        }
        JOB_STORE[job_id] = {"status": "error", "payload": error_payload}
        logger.error("✗ Job %s failed: %s", job_id, exc, exc_info=True)
        return json.dumps(error_payload, indent=2)


@app.tool()
async def get_job_status(job_id: str) -> str:
    """
    Returns the current status and payload of a previously submitted job.
    
    Args:
        job_id: The job ID returned by split_audio.
    """
    job = JOB_STORE.get(job_id)
    if not job:
        result = {"error": f"Job '{job_id}' not found."}
    else:
        result = job.get("payload", {"status": job["status"]})
    return json.dumps(result, indent=2)


@app.tool()
async def list_jobs() -> str:
    """Lists all completed and in-progress stem separation jobs."""
    summary = [
        {
            "job_id": jid,
            "status": info.get("status"),
            "started_at": info.get("started_at"),
        }
        for jid, info in JOB_STORE.items()
    ]
    return json.dumps(summary, indent=2)


@app.tool()
async def check_health() -> str:
    """Verification tool for server dependencies."""
    import subprocess
    import importlib.util

    results = {"status": "ok", "checks": []}
    all_ok = True

    # 1. Check CLI tools
    for cmd, name in [("ffmpeg", "FFmpeg"), ("yt-dlp", "yt-dlp")]:
        try:
            proc = await asyncio.create_subprocess_exec(
                cmd, "--version", stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            await proc.communicate()
            res = {"name": name, "status": "ok" if proc.returncode == 0 else "error"}
            if proc.returncode != 0: all_ok = False
        except Exception as e:
            res = {"name": name, "status": "not_found", "error": str(e)}
            all_ok = False
        results["checks"].append(res)

    # 2. Check Python packages
    packages = ["mcp", "librosa", "demucs", "torch"]
    for pkg in packages:
        found = importlib.util.find_spec(pkg) is not None
        results["checks"].append({"name": f"python:{pkg}", "status": "ok" if found else "missing"})
        if not found: all_ok = False

    # 3. Check paths
    results["checks"].append({
        "name": "stems_root",
        "status": "ok" if STEMS_ROOT.exists() else "error",
        "path": str(STEMS_ROOT)
    })

    if not all_ok:
        results["status"] = "degraded"

    return json.dumps(results, indent=2)


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run()

