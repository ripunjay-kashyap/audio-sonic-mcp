"""
Audio Stem Splitter MCP Server
Converts YouTube URLs into separated stems + sonic signature JSON.
"""

import asyncio
import anyio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from pipeline.ingestion import validate_source, save_metadata, load_metadata
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

# Ensure CLI tools installed alongside this Python (yt-dlp, demucs) are
# findable by subprocess calls even when the venv is not "activated".
_venv_bin = str(Path(sys.executable).parent)
if _venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")
    logger.info("Added venv bin to PATH: %s", _venv_bin)

# Prevent torchaudio from attempting to load FFmpeg/torchcodec DLLs
os.environ["TORCHAUDIO_USE_BACKEND_PREFERENCE"] = "soundfile"

# ── Config ────────────────────────────────────────────────────────────────────
STEMS_ROOT = Path(os.environ.get("STEMS_ROOT", Path(__file__).parent / "stems"))
STEMS_ROOT.mkdir(parents=True, exist_ok=True)

app = FastMCP("audio-stem-mcp")

# ── Concurrency Control ───────────────────────────────────────────────────────
# Prevents multiple Demucs/ML jobs from crashing local RAM
CONCURRENCY_LOCK = asyncio.Lock()

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

    logger.info("▶ split_audio | job=%s model=%s url=%s", job_id, model, url)

    JOB_STORE[job_id] = {"status": "running", "started_at": time.time()}
    t_start = time.perf_counter()

    try:
        async with CONCURRENCY_LOCK:
            # ── Stage 1: Validate + Metadata Cache ─────────────────────────────────
            logger.info("[1/6] Validating source …")
            job_dir = STEMS_ROOT / job_id
            cached_meta = load_metadata(job_dir)
            if cached_meta:
                logger.info("Fast-Resume: Loaded metadata from disk for job=%s", job_id)
                source_info = cached_meta
            else:
                source_info = await anyio.to_thread.run_sync(validate_source, url)
                try:
                    save_metadata(job_dir, source_info)
                    logger.info("Persisted metadata.json for job=%s", job_id)
                except Exception as e:
                    logger.warning(
                        "Could not persist metadata for job=%s: %s", job_id, e
                    )

            # ── Fast Resume Check ─────────────────────────────────────────────────
            expected_stems_dir = STEMS_ROOT / job_id / "stems"
            resume_stems = None
            if expected_stems_dir.exists():
                if all(
                    (expected_stems_dir / f"{s}.wav").exists()
                    for s in ["vocals", "drums", "bass", "other"]
                ):
                    resume_stems = expected_stems_dir
                else:
                    for nested in expected_stems_dir.rglob("vocals.wav"):
                        if all(
                            (nested.parent / f"{s}.wav").exists()
                            for s in ["vocals", "drums", "bass", "other"]
                        ):
                            resume_stems = nested.parent
                            break

            if resume_stems:
                logger.info(
                    "Fast-Resume: Found existing stems in %s. Skipping Download & Split stages.",
                    resume_stems,
                )
                stems_dir = resume_stems
                stem_files = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
                sdr = 8.0  # Proxy default for resumed jobs
            else:
                # ── Stage 2: Download ─────────────────────────────────────────────────
                logger.info("[2/6] Downloading audio stream …")
                raw_audio_path = await anyio.to_thread.run_sync(
                    download_audio, url, job_id, STEMS_ROOT
                )

                # ── Stage 3: Convert ──────────────────────────────────────────────────
                logger.info("[3/6] Converting to 44.1kHz WAV …")
                wav_path = await anyio.to_thread.run_sync(
                    convert_to_wav, raw_audio_path
                )

                # ── Stage 4: Split ────────────────────────────────────────────────────
                logger.info("[4/6] Running htdemucs stem separation …")
                stems_dir, stem_files, sdr = await anyio.to_thread.run_sync(
                    split_stems, wav_path, job_id, STEMS_ROOT, model
                )

            # ── Stage 5: Analyze ──────────────────────────────────────────────────
            # asyncio.to_thread is used here (not anyio.to_thread.run_sync) because
            # the MCP server's anyio task group can cancel tasks mid-flight during
            # long-running stages. asyncio.to_thread runs in the standard executor
            # and is not subject to anyio's cancellation machinery.
            logger.info("[5/6] Extracting BPM, key, transients, frequencies …")
            features = await asyncio.to_thread(analyze_stems, stems_dir, stem_files)
            logger.info(
                "[5/6] Analysis complete: bpm=%.1f key=%s confidence=%.2f",
                features["bpm"],
                features["key"],
                features.get("mode_confidence", 0),
            )

            # ── Stage 6: Vectorize ────────────────────────────────────────────────
            logger.info("[6/6] Generating CLAP vibe vector …")
            vibe_vector = await asyncio.to_thread(
                generate_vibe_vector, stems_dir, stem_files
            )
            logger.info("[6/6] Vectorize complete: dim=%d", len(vibe_vector))

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
                cpu_samples=[],
                source_info=source_info,
            )

            JOB_STORE[job_id] = {"status": "success", "payload": payload}
            result = json.dumps(payload, indent=2)
            logger.info(
                "✓ split_audio | job=%s elapsed=%.1fs response_bytes=%d",
                job_id,
                elapsed,
                len(result),
            )
            return result

    except BaseException as exc:
        elapsed = time.perf_counter() - t_start
        error_payload = {
            "header": {"job_id": job_id, "status": "error", "confidence_score": 0.0},
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "telemetry": {"inference_time_sec": round(elapsed, 2)},
        }
        JOB_STORE[job_id] = {"status": "error", "payload": error_payload}
        logger.error(
            "✗ split_audio | job=%s elapsed=%.1fs error=%s: %s",
            job_id,
            elapsed,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return json.dumps(error_payload, indent=2)


@app.tool()
async def get_job_status(job_id: str) -> str:
    """
    Returns the current status and payload of a previously submitted job.

    Args:
        job_id: The job ID returned by split_audio.
    """
    logger.info("▶ get_job_status | job=%s", job_id)
    job = JOB_STORE.get(job_id)
    if not job:
        result = {"error": f"Job '{job_id}' not found."}
        logger.warning("  get_job_status | job=%s not found", job_id)
    else:
        result = job.get("payload", {"status": job["status"]})
        logger.info("  get_job_status | job=%s status=%s", job_id, job.get("status"))
    return json.dumps(result, indent=2)


@app.tool()
async def list_jobs() -> str:
    """Lists all completed and in-progress stem separation jobs."""
    logger.info("▶ list_jobs | total=%d", len(JOB_STORE))
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

    # 1. Check CLI tools — use to_thread/subprocess.run, consistent with pipeline
    def _check_cmd(cmd):
        return subprocess.run(
            [cmd, "-version" if cmd == "ffmpeg" else "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=10,
        )

    for cmd, name in [("ffmpeg", "FFmpeg"), ("yt-dlp", "yt-dlp")]:
        try:
            r = await asyncio.to_thread(_check_cmd, cmd)
            res = {"name": name, "status": "ok" if r.returncode == 0 else "error"}
            if r.returncode != 0:
                all_ok = False
        except Exception as e:
            res = {"name": name, "status": "not_found", "error": str(e)}
            all_ok = False
        results["checks"].append(res)

    # 2. Check Python packages
    packages = ["mcp", "librosa", "demucs", "torch"]
    for pkg in packages:
        found = importlib.util.find_spec(pkg) is not None
        results["checks"].append(
            {"name": f"python:{pkg}", "status": "ok" if found else "missing"}
        )
        if not found:
            all_ok = False

    # 3. Check paths
    results["checks"].append(
        {
            "name": "stems_root",
            "status": "ok" if STEMS_ROOT.exists() else "error",
            "path": str(STEMS_ROOT),
        }
    )

    if not all_ok:
        results["status"] = "degraded"

    return json.dumps(results, indent=2)


# ── Entrypoint ────────────────────────────────────────────────────────────────


def main():
    app.run()


if __name__ == "__main__":
    main()
