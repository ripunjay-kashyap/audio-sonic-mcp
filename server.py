"""
Audio Sonic Signature MCP Server
Analyzes a YouTube URL and returns a structured sonic signature JSON.
"""

import asyncio
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
from pipeline.separator import separate_stems
from pipeline.analyzer import analyze_audio
from pipeline.vectorizer import generate_vibe_vector
from pipeline.assembler import assemble_payload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("audio-sonic-mcp")

# Ensure CLI tools installed alongside this Python (yt-dlp) are
# findable by subprocess calls even when the venv is not "activated".
_venv_bin = str(Path(sys.executable).parent)
if _venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")
    logger.info("Added venv bin to PATH: %s", _venv_bin)

# Prevent torchaudio from attempting to load FFmpeg/torchcodec DLLs
os.environ["TORCHAUDIO_USE_BACKEND_PREFERENCE"] = "soundfile"

# ── Config ────────────────────────────────────────────────────────────────────
JOBS_ROOT = Path(os.environ.get("JOBS_ROOT", Path(__file__).parent / "jobs"))
JOBS_ROOT.mkdir(parents=True, exist_ok=True)

app = FastMCP("audio-sonic-mcp")

# ── Concurrency Control ───────────────────────────────────────────────────────
# Serializes ML jobs to prevent OOM on constrained hardware
CONCURRENCY_LOCK = asyncio.Lock()

# ── In-memory job store ───────────────────────────────────────────────────────
JOB_STORE: dict[str, dict] = {}


def _cleanup_job_artifacts(job_dir: Path) -> None:
    """Remove downloaded audio and stem WAVs after a successful run.

    Keeps metadata.json for traceability. Disabled when KEEP_JOB_FILES=1
    (useful for debugging or to preserve fast-resume).
    """
    if os.environ.get("KEEP_JOB_FILES", "").lower() in ("1", "true", "yes"):
        logger.info("cleanup: skipped (KEEP_JOB_FILES set) for %s", job_dir.name)
        return

    import shutil

    removed_bytes = 0
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
            logger.warning("cleanup: failed on %s: %s", path, e)

    logger.info(
        "cleanup: freed %.1f MB from %s", removed_bytes / 1_048_576, job_dir.name
    )


# ── Tool handlers ─────────────────────────────────────────────────────────────


@app.tool()
async def get_sonic_signature(url: str, job_id: str = None) -> str:
    """
    Downloads audio from a YouTube URL and returns its sonic signature.
    Uses in-memory HPSS to extract BPM, key, stereo width, transient punch,
    frequency peaks, vocal presence, and a 512-dim CLAP vibe vector.

    Args:
        url: YouTube URL of the track to analyze.
        job_id: Optional custom job ID. Auto-generated if omitted.
    """
    job_id = job_id or f"sig_{uuid.uuid4().hex[:8]}"

    logger.info("▶ get_sonic_signature | job=%s url=%s", job_id, url)

    JOB_STORE[job_id] = {"status": "running", "started_at": time.time()}
    t_start = time.perf_counter()

    try:
        async with CONCURRENCY_LOCK:
            import anyio

            # ── Stage 1: Validate + Metadata Cache ─────────────────────────────────
            logger.info("[1/5] Validating source …")
            job_dir = JOBS_ROOT / job_id
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
            wav_path = job_dir / "input.wav"
            if wav_path.exists():
                logger.info(
                    "Fast-Resume: Found existing WAV at %s. Skipping download & convert.",
                    wav_path,
                )
            else:
                # ── Stage 2: Download ─────────────────────────────────────────────────
                logger.info("[2/5] Downloading audio stream …")
                raw_audio_path = await anyio.to_thread.run_sync(
                    download_audio, url, job_id, JOBS_ROOT
                )

                # ── Stage 3: Convert ──────────────────────────────────────────────────
                logger.info("[3/5] Converting to 44.1kHz WAV …")
                wav_path = await anyio.to_thread.run_sync(
                    convert_to_wav, raw_audio_path
                )

            # ── Stage 4: Stem Separation ──────────────────────────────────────────
            # asyncio.to_thread is used here (not anyio.to_thread.run_sync) because
            # the MCP server's anyio task group can cancel tasks mid-flight during
            # long-running stages. asyncio.to_thread runs in the standard executor
            # and is not subject to anyio's cancellation machinery.
            logger.info("[4/5] Separating stems (Demucs mdx_extra) …")
            stems_dir = await asyncio.to_thread(separate_stems, wav_path)
            if stems_dir:
                logger.info("[4/5] Stems ready at %s", stems_dir)
            else:
                logger.info("[4/5] Stem separation unavailable — will use HPSS")

            # ── Stage 5: Analyze + Vectorize ──────────────────────────────────────
            logger.info("[5/5] Running analysis and vibe vectorization …")
            features = await asyncio.to_thread(analyze_audio, wav_path, stems_dir)
            logger.info(
                "[5/5] Analysis complete: bpm=%.1f key=%s confidence=%.2f",
                features["bpm"],
                features["key"],
                features.get("mode_confidence", 0),
            )

            vibe_vector = await asyncio.to_thread(generate_vibe_vector, wav_path)
            logger.info("[5/5] Vectorize complete: dim=%d", len(vibe_vector))

            # ── Assemble payload ──────────────────────────────────────────────────
            elapsed = time.perf_counter() - t_start
            payload = assemble_payload(
                job_id=job_id,
                features=features,
                vibe_vector=vibe_vector,
                inference_time=elapsed,
                cpu_samples=[],
                source_info=source_info,
            )

            JOB_STORE[job_id] = {"status": "success", "payload": payload}
            result = json.dumps(payload, indent=2)
            logger.info(
                "✓ get_sonic_signature | job=%s elapsed=%.1fs response_bytes=%d",
                job_id,
                elapsed,
                len(result),
            )
            _cleanup_job_artifacts(job_dir)
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
            "✗ get_sonic_signature | job=%s elapsed=%.1fs error=%s: %s",
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
        job_id: The job ID returned by get_sonic_signature.
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
    """Lists all completed and in-progress analysis jobs."""
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

    packages = ["mcp", "librosa", "soundfile", "numpy"]
    for pkg in packages:
        found = importlib.util.find_spec(pkg) is not None
        results["checks"].append(
            {"name": f"python:{pkg}", "status": "ok" if found else "missing"}
        )
        if not found:
            all_ok = False

    results["checks"].append(
        {
            "name": "jobs_root",
            "status": "ok" if JOBS_ROOT.exists() else "error",
            "path": str(JOBS_ROOT),
        }
    )

    if not all_ok:
        results["status"] = "degraded"

    return json.dumps(results, indent=2)


# ── Entrypoint ────────────────────────────────────────────────────────────────


def _prewarm_librosa() -> None:
    """Trigger numba JIT compilation in the main thread before serving.

    Without this, the first librosa call from a worker thread (asyncio.to_thread)
    deadlocks under FastMCP's task group on Windows.
    """
    import numpy as np
    import librosa

    dummy_stereo = np.zeros((2, 4096), dtype=np.float32)
    dummy_mono = librosa.to_mono(dummy_stereo)
    librosa.resample(dummy_mono, orig_sr=44100, target_sr=22050)
    librosa.effects.hpss(dummy_mono)
    logger.info("librosa pre-warm complete")


def _prewarm_demucs() -> None:
    """Load Demucs model weights into memory before serving.
    Downloads ~400 MB of weights on first run; cached on disk thereafter.
    Download progress bars are redirected to stderr so they don't corrupt
    the MCP stdio JSON-RPC stream.
    """
    import sys
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        from pipeline.separator import load_demucs_model
        load_demucs_model()
    except Exception as exc:
        logger.warning("Demucs pre-warm skipped (stem separation will use HPSS): %s", exc)
    finally:
        sys.stdout = old_stdout


def main():
    _prewarm_librosa()
    _prewarm_demucs()
    app.run()


if __name__ == "__main__":
    main()
