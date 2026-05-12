# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An MCP (Model Context Protocol) server that accepts a YouTube URL and returns a structured **sonic signature** JSON — giving an LLM like Claude rich musical context (BPM, key, 512-dim CLAP embedding, production profile) from a single tool call.

## System Requirements

- Python 3.10+
- FFmpeg (must be on PATH)
- yt-dlp (installed via pip)
- Optional: `transformers`, `torch`, `torchaudio` for CLAP vibe vectors (~4 GB download)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: CLAP vibe vectors
pip install ".[clap]"
```

## Running the Server

```bash
python server.py

# Custom job directory
JOBS_ROOT=/data/jobs python server.py
```

The server communicates over stdio (MCP protocol) — running it directly prints nothing useful interactively. Use Claude Desktop or an MCP client to exercise the tools.

## Claude Desktop Integration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "audio-sonic-mcp": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/path/to/server.py"],
      "env": {
        "JOBS_ROOT": "/tmp/audio_jobs"
      }
    }
  }
}
```

On Windows use the `.venv\Scripts\python.exe` path.

## Testing

Run the unit test suite (synthetic audio, no real downloads needed):

```bash
pytest                   # runs tests/ suite
pytest -v tests/         # verbose
```

For a full end-to-end smoke test via the MCP client:

```bash
python smoke_test.py                    # uses a default YouTube URL
python smoke_test.py <url> <job_id>     # custom URL + job ID
```

### Accuracy regression tests (real song stems)

`tests/test_accuracy.py` validates BPM (within 5%) and key (exact match) against ground-truth values from Songstats for a curated set of songs. Tests load cached Demucs stems from `jobs/<slug>/stems/mdx_extra/input/` — no network calls, ~1s per song. If stems aren't cached, the test is skipped.

To populate the cache for all songs, run each URL once with `KEEP_JOB_FILES=1` and the canonical slug as the job ID:

```bash
$env:KEEP_JOB_FILES="1"  # Windows PowerShell; on bash: export KEEP_JOB_FILES=1
.venv\Scripts\python.exe smoke_test.py "https://www.youtube.com/watch?v=fXivMSJm_kA"  sig_yukon
.venv\Scripts\python.exe smoke_test.py "https://www.youtube.com/watch?v=tvTRZJ-4EyI"  sig_humble
.venv\Scripts\python.exe smoke_test.py "https://www.youtube.com/watch?v=jLQrk6rmX6w"  sig_somebody
.venv\Scripts\python.exe smoke_test.py "https://www.youtube.com/watch?v=iwd8N6K-sLk"  sig_exo_tempo
.venv\Scripts\python.exe smoke_test.py "https://www.youtube.com/watch?v=0PTU4kGj5JI"  sig_lauv_julia
.venv\Scripts\python.exe smoke_test.py "https://www.youtube.com/watch?v=kON9fn01rUQ"  sig_ballad
.venv\Scripts\python.exe smoke_test.py "https://www.youtube.com/watch?v=r_0JjYUe5jo"  sig_hiphop
```

Each run takes ~3-4 min (full Demucs separation). After populating, `pytest tests/test_accuracy.py -v` runs the full suite in seconds. The URL and ground-truth values are pinned in `GROUND_TRUTH` at the top of the test file — update both together when adding new songs.

## Container (Podman)

This project uses **Podman** (rootless, daemonless). The Dockerfile is standard OCI-compatible.

### Prerequisites (Windows)
```bash
podman machine start   # only needed if not already running
podman machine list    # verify status
```

### Build & Run
```bash
podman build -t audio-sonic-mcp .

podman run -it --rm \
  -v $(pwd)/jobs:/app/jobs:Z \
  -v $(pwd)/models:/app/models:Z \
  audio-sonic-mcp
```

> The `:Z` suffix sets the SELinux label. Drop it for Docker.

## Architecture

The pipeline is linear — each stage feeds into the next, all orchestrated by `server.py`:

| Stage | File | Responsibility |
|---|---|---|
| 1 — Validate | `ingestion.py` | URL validation + yt-dlp metadata probe; enforces 60-min limit |
| 2 — Download | `downloader.py` | yt-dlp audio-only download to `JOBS_ROOT/<job_id>/` |
| 3 — Convert | `converter.py` | FFmpeg → 44.1kHz WAV |
| 4 — Analyze | `analyzer.py` | HPSS in-memory separation; BPM from percussive signal, key from harmonic signal; transient punch, stereo width, dominant frequency peaks |
| 5 — Vectorize | `vectorizer.py` | CLAP 512-dim embedding via `laion/larger_clap_music_and_speech`; falls back to librosa mel+MFCC composite if CLAP unavailable |

**Analysis window:** All three audio stages (separator, analyzer, vectorizer) call `pipeline.window.pick_window()` to choose the same slice of the track:
- Track ≥ 75s → **30s–90s** (skips intro, lands in verse 1 / pre-chorus)
- Track 30s–75s → **0s–60s** (or 0 → end if shorter than 60s)
- Track < 30s → **the entire track** (snippet mode)

Tunable via constants in `pipeline/window.py`.
| Assemble | `assembler.py` | Merges all outputs into the canonical JSON payload |

All pipeline stages run via `asyncio.to_thread()` in `server.py` — they are synchronous functions wrapped for async execution.

## Job Store

Jobs are tracked in an in-memory dict `JOB_STORE` in `server.py`. It is not persisted — restarting the server clears all job history.

## MCP Tools Exposed

- `get_sonic_signature(url, job_id?)` — runs the full pipeline; returns sonic signature JSON
- `get_job_status(job_id)` — fetches result from `JOB_STORE`
- `list_jobs()` — lists all job statuses
- `check_health()` — verifies FFmpeg, yt-dlp, and Python package availability

## Sonic Signature Schema

```json
{
  "header": { "job_id", "status", "confidence_score", "source_metadata" },
  "sonic_signature": {
    "bpm": float,
    "key": "E Minor",
    "mode_confidence": float,
    "vibe_vector": [512 floats, L2-normalized],
    "production_profile": {
      "vocal_presence": "forward|present|background",
      "transient_punch": float,
      "stereo_width": "mono|narrow|wide|...",
      "dominant_freq_peaks_hz": { "harmonic": [...], "percussive": [...] }
    }
  },
  "telemetry": { "inference_time_sec": float }
}
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `JOBS_ROOT` | `./jobs` | Root directory for job working files (downloaded audio, WAV) |
| `KEEP_JOB_FILES` | unset | When `1`/`true`, skips post-run cleanup so audio + stems stay on disk (useful for debugging or fast-resume). |

## Gotchas

**Auto-cleanup:** After a successful run, the downloaded audio (`raw_audio.*`), converted WAV (`input.wav`), and `stems/` directory are deleted automatically. Only `metadata.json` is retained. This frees ~75 MB per job. Set `KEEP_JOB_FILES=1` to disable.

**Fast Resume:** If `JOBS_ROOT/<job_id>/input.wav` already exists, stages 2–3 (download, convert) are skipped automatically. Only effective when `KEEP_JOB_FILES=1` since auto-cleanup removes these files by default. Remove the job directory to force a full re-run.

**Concurrency:** A global `asyncio.Lock` (`CONCURRENCY_LOCK`) serializes all jobs — only one runs at a time.

**CLAP install shorthand:** `pip install ".[clap]"` (uses the extras group in `pyproject.toml`) installs torch, torchaudio, and transformers in one command.
