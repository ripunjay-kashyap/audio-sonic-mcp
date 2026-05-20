# Audio Stem Splitter & Analyzer — MCP Server

A local demonstration of an LLM-callable machine learning pipeline for audio source separation and Music Information Retrieval (MIR). Built as a Model Context Protocol (MCP) server, it makes state-of-the-art deep learning models programmatically accessible to AI agents and LLM orchestration frameworks.

---

## Project Status

This is a custom-mcp project running on consumer-grade hardware, not a hosted service.

**Hardware constraint:** Demucs and CLAP are computationally heavy. On a modern consumer CPU, a 5-minute track takes approximately 4 minutes end-to-end. However, the codebase is designed to scale dynamically: when deployed on a machine with an Nvidia GPU and CUDA configured (like a free Kaggle T4 notebook), the exact same pipeline completes in 20–30 seconds.

**What this project demonstrates:** ML pipeline architecture, integration of multiple deep learning models (Demucs, CLAP), async system design for high-latency inference workloads, and MCP server implementation. The bottleneck is hardware, not software design — the pipeline seamlessly leverages GPU acceleration when available.

---

## Problem Statement

Extracting high-quality isolated audio stems and semantic metadata from raw web audio is traditionally a manual, brittle, multi-tool process. ML workflows, automated music production, and audio dataset curation all require clean, structured, programmatic access to these features.

Compounding this is the operational challenge of deploying deep learning inference pipelines: managing conflicting system-level dependencies (CUDA, FFmpeg, tensor runtimes) in reproducible environments, and handling the latency characteristics of large model inference without blocking orchestrating agents.

This project addresses both by wrapping a 6-stage AI audio processing pipeline in a robust MCP server designed for asynchronous, high-latency inference workloads.

---

## Architecture

The pipeline is modular and idempotent, with built-in checkpointing for fault-tolerant execution across long-running batch jobs.


```text
  LLM Agent / Claude Desktop
           |
           |  MCP (stdio)
           v
+--------------------------------------------------------------------------+
|  FastMCP Server  (server.py)                                             |
|  get_sonic_signature · get_job_status · list_jobs · check_health        |
|  asyncio.Lock (one job at a time)  +  in-memory job store               |
+----------------------------------+---------------------------------------+
                                   |
                                   v
+--------------------------------------------------------------------------+
|  6-Stage Inference Pipeline                                              |
|                                                                          |
|  Stage 1 — Ingest      yt-dlp probe         -->  metadata.json          |
|  Stage 2 — Download    yt-dlp audio stream  -->  raw audio file         |
|  Stage 3 — Convert     FFmpeg               -->  44.1 kHz WAV           |
|  Stage 4 — Separate    Demucs (htdemucs)    -->  4 stem WAVs            |
|  Stage 5 — Analyze     librosa MIR          -->  BPM, key, peaks        |
|  Stage 6 — Embed       CLAP (laion)         -->  512-dim vector         |
+----------------------------------+---------------------------------------+
                                   |
                                   v
+--------------------------------------------------------------------------+
|  JSON Payload                                                            |
|  header:           job_id, status, confidence_score, source_meta        |
|  stems_metadata:   file paths, SDR quality metric                       |
|  sonic_signature:  BPM, key, vibe_vector, production_profile            |
|  telemetry:        inference_time_sec                                    |
+--------------------------------------------------------------------------+
```

**Stage 1 — Ingestion:** URL validation and metadata probing via `yt-dlp`. Enforces a 60-minute duration limit and persists source metadata to disk atomically (tmp-rename pattern) to avoid re-probing on resume.

**Stage 2–3 — Download & Normalization:** Pulls the highest-quality audio-only stream (no video mux), then standardizes to 44.1kHz mono WAV via `FFmpeg`, ensuring consistent tensor shapes for downstream model ingestion.

**Stage 4 — Source Separation (Inference):** Runs Facebook's **Demucs** (`mdx_extra` model) to decompose the audio mixture into four isolated stems: `vocals`, `drums`, `bass`, `other`. Computes a proxy SDR (Source-to-Distortion Ratio) as a separation quality metric used downstream in confidence scoring.

**Stage 5 — MIR Feature Extraction:** Uses `librosa` and `madmom` to extract deterministic musical features per stem:
- BPM via `madmom`'s Recurrent Neural Network (RNN) beat tracker (highly robust against syncopation and octave/half-time errors), with `librosa` as a fallback.
- Musical key via weighted chroma fusion (60% bass / 40% harmonic content), with HPSS pre-filtering to remove percussive leakage before CQT chroma extraction.
- Transient punch from onset strength envelope (97th-percentile peak-to-mean ratio)
- Dominant frequency peaks per stem (20Hz–16kHz masked FFT)
- Stereo width via L/R channel correlation
- Vocal presence via RMS ratio of vocals-to-mix

**Stage 6 — Semantic Embedding:** Passes the stem mix through **CLAP** (Contrastive Language-Audio Pretraining, `laion/larger_clap_music_and_speech`) to generate a 512-dimensional vector embedding. CLAP maps audio into the same latent space as text, enabling cross-modal similarity search ("find tracks that sound like this") without text labels. Falls back gracefully to a hand-engineered 512-dim librosa composite (mel-spectrogram, MFCC, chroma, spectral statistics, Tonnetz) if CLAP is unavailable, maintaining a consistent output schema for downstream consumers.

---

## System Design: Handling Inference Latency

### The Constraint

Demucs and CLAP are computationally heavy. This project can run on a local CPU — where processing a 5-minute track takes approximately 4 minutes end-to-end. However, when run on a GPU (e.g., Kaggle T4), it completes in 20-30 seconds. Even at 20 seconds, standard synchronous request-response patterns are risky for orchestration agents, which is why an async job architecture is used.

### The Solutions

**Asynchronous Job Orchestration:** The MCP server accepts a job submission and returns a `job_id` immediately, decoupling the orchestrating LLM's context window from the inference runtime. A `get_job_status` endpoint allows polling asynchronously.

**Idempotent Pipeline with Checkpointing:** Intermediate artifacts (downloaded audio, converted WAV, separated stem files) are persisted to a per-job directory. If a job is interrupted (e.g., OOM during model inference), the pipeline detects existing artifacts and resumes from the last successful stage rather than re-executing expensive upstream work.

**Concurrency Serialization:** A single `asyncio.Lock` serializes all Demucs/CLAP jobs. This prevents RAM exhaustion from overlapping inference runs on constrained hardware, trading throughput for stability — the correct trade-off for a single-node deployment.

**Thread Isolation for ML Workloads:** Long-running synchronous ML stages (librosa, CLAP) are dispatched via `asyncio.to_thread()` rather than `anyio.to_thread.run_sync()`. This avoids anyio's task cancellation machinery interrupting mid-flight tensor operations.

---

## Technical Stack

**Machine Learning and AI:**
- PyTorch and Hugging Face Transformers for model loading and tensor operations
- Demucs (Meta AI) — Hybrid Spectrogram/Waveform Transformer for music source separation
- LAION CLAP — contrastive audio/language model for multi-modal semantic embeddings

**Audio Processing and MIR:**
- librosa — algorithmic feature extraction (BPM, key, spectral analysis)
- FFmpeg — media normalization and stream conversion

**Infrastructure and MLOps:**
- FastMCP — exposes the pipeline as a standardized MCP tool server for LLM agents
- Docker/Podman — containerization for reproducible deployment of GPU drivers (CUDA/ROCm) and system dependencies
- anyio — async I/O abstraction layer for the MCP runtime

---

## Output Schema

```json
{
  "header": {
    "job_id": "sig_a3f9b2c1",
    "status": "success",
    "confidence_score": 0.91,
    "source_metadata": {
      "title": "Track Name",
      "uploader": "Artist",
      "duration_sec": 214,
      "genre_hint": "Music"
    }
  },
  "sonic_signature": {
    "bpm": 128.05,
    "key": "F# Minor",
    "mode_confidence": 0.74,
    "vibe_vector": [0.012, -0.034, 0.091, "... 512 dims ..."],
    "production_profile": {
      "vocal_presence": "forward",
      "transient_punch": 0.781,
      "stereo_width": "wide",
      "dominant_freq_peaks_hz": {
        "harmonic": [55.0, 110.2, 82.4],
        "percussive": [8372.0, 125.0, 250.1]
      }
    }
  },
  "telemetry": {
    "inference_time_sec": 47.3
  }
}
```

The `confidence_score` is a weighted heuristic: 60% SDR separation quality + 40% feature extraction success, giving downstream consumers a single trust signal without requiring knowledge of the internal pipeline state.

> **Note:** Stem WAV files are deleted automatically after each run to free ~75 MB per job. Set `KEEP_JOB_FILES=1` to retain them on disk.

---

## Quick Start

You can run this MCP server in three different ways depending on your hardware availability.

### Option 1: Local CPU (Standard Python)
Best for testing or if you don't mind waiting ~4 minutes per song.

```bash
# 1. Clone the repository
git clone https://github.com/ripunjay-kashyap/audio-sonic-mcp.git
cd audio-sonic-mcp

# 2. Set up virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Optional: Install CLAP for semantic embeddings (~4 GB download)
pip install ".[clap]"
```

### Option 2: Free Cloud GPU (Kaggle Notebook)
Best for processing many tracks quickly (20-30 seconds per track) without paying for cloud compute.

1. Create a new notebook on [Kaggle](https://www.kaggle.com).
2. In the right panel, under **Accelerator**, select **GPU T4 x2**.
3. Add the following to the first cell to install the project with CUDA acceleration:
```python
!git clone https://github.com/ripunjay-kashyap/audio-sonic-mcp.git
%cd audio-sonic-mcp

!apt-get update && apt-get install -y ffmpeg
!pip install demucs yt-dlp librosa soundfile madmom
!pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```
4. You can now import and run `server.py` or trigger inference scripts directly within the notebook!

### Option 3: Containerized Deployment (Docker)
Best for deploying to a cloud VM (e.g., AWS `g5`, GCP `n1` with T4) or a local machine with a dedicated GPU.

```bash
# Build the container
docker build -t audio-sonic-mcp .

# Create persistent directories for outputs and models
mkdir -p stems models

# Run with GPU support (Requires nvidia-container-toolkit)
docker run -it --rm --gpus all \
  -v $(pwd)/stems:/app/stems \
  -v $(pwd)/models:/app/models \
  audio-sonic-mcp
```

### Claude Desktop Integration

#### Step 1 — Open the config file

The easiest way is through Claude Desktop itself:

1. Open Claude Desktop
2. Click your **profile icon** (bottom-left)
3. Go to **Settings → Developer**
4. Click **Edit Config**

This opens `claude_desktop_config.json` in your default text editor.

> **Can't find it manually?**
> - **Windows (Store app):** `%LOCALAPPDATA%\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Roaming\Claude\claude_desktop_config.json`
> - **Windows (direct install):** `%APPDATA%\Claude\claude_desktop_config.json`
> - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

---

#### Step 2 — Add the server entry

Paste the block below into `claude_desktop_config.json`, replacing the path placeholders with your actual install paths.

**Windows**
```json
{
  "mcpServers": {
    "audio-sonic-mcp": {
      "command": "C:\\path\\to\\audio-sonic-mcp\\.venv\\Scripts\\python.exe",
      "args": [
        "C:\\path\\to\\audio-sonic-mcp\\server.py"
      ],
      "env": {
        "JOBS_ROOT": "C:\\path\\to\\audio-sonic-mcp\\jobs"
      }
    }
  }
}
```

**macOS / Linux**
```json
{
  "mcpServers": {
    "audio-sonic-mcp": {
      "command": "/path/to/audio-sonic-mcp/.venv/bin/python",
      "args": [
        "/path/to/audio-sonic-mcp/server.py"
      ],
      "env": {
        "JOBS_ROOT": "/path/to/audio-sonic-mcp/jobs"
      }
    }
  }
}
```

> **Note:** JSON requires double backslashes `\\` for Windows paths.

---

#### Step 3 — Restart Claude Desktop

Fully quit Claude Desktop (system tray → right-click → **Quit**, or Task Manager on Windows) and reopen it.

> **First launch takes 1–3 minutes.** On the very first run, the server downloads the Demucs model weights (~400 MB) before it starts accepting requests. This is a one-time download — subsequent startups take ~10 seconds. Claude Desktop may show the server as "connecting" during this time; that is normal.

---

#### Step 4 — Verify the connection

Once Claude Desktop reopens, look for the **hammer icon** (🔨) in the chat input bar — that confirms the MCP server is connected.

To run a health check, ask Claude:

> *"Check the health of the audio-sonic-mcp server"*

Then try a real analysis:

> *"Get the sonic signature for [YouTube URL]"*

**Option B: Docker (Requires Option 3 installation)**
```json
{
  "mcpServers": {
    "audio-sonic-mcp": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "/absolute/path/to/audio-sonic-mcp/jobs:/app/jobs",
        "-v", "/absolute/path/to/audio-sonic-mcp/models:/app/models",
        "audio-sonic-mcp"
      ]
    }
  }
}
```

---

## MCP Tools Exposed

| Tool | Description |
|------|-------------|
| `get_sonic_signature(url, job_id?)` | Submits a full pipeline job; returns structured JSON payload |
| `get_job_status(job_id)` | Polls the status and result of a submitted job |
| `list_jobs()` | Lists all jobs and their current status |
| `check_health()` | Verifies FFmpeg, yt-dlp, and Python package availability |

---

## System Requirements

- Python 3.10+
- FFmpeg (on system PATH)
- yt-dlp
- Demucs
- Optional: `transformers`, `torch`, `torchaudio` for CLAP embeddings
