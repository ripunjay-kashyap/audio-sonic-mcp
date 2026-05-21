# 🎵 Audio Sonic MCP

[![Tests](https://github.com/ripunjay-kashyap/audio-sonic-mcp/actions/workflows/test.yml/badge.svg)](https://github.com/ripunjay-kashyap/audio-sonic-mcp/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-server-8A2BE2.svg)](https://modelcontextprotocol.io/)

**Turn any song into a structured "sonic signature" — extracting tempo, musical key, a 512-dimension CLAP vibe embedding, human-readable vibe tags, and a production profile — from a single local call.**

Audio Sonic MCP runs entirely on your local machine (requiring no API keys, external servers, or cloud dependencies) and exposes two premium access points to the same underlying high-fidelity audio analysis engine:

| | Tailored For | Core Interface & Mechanics |
|---|---|---|
| 🤖 **MCP Server** | LLMs, AI agents, & IDEs (Claude, Cursor, Windsurf, Cline) | Asynchronous, fire-and-forget analysis of YouTube URLs. Avoids blocking client LLMs during heavy audio processing. |
| 🎚️ **Local CLI** | Musicians, sound producers, & audio engineers | Deep command-line tool targeting local files for full-song multi-window analysis and high-fidelity output. |

---

## 🎹 Quick Taste: What You Get

### 1. Musician-Friendly CLI Summary (`--summary` mode)
```text
🎵 SONIC SIGNATURE — my_demo.mp3  (3:24)

  TEMPO    153.8 BPM  (steady)
  KEY      G Major  ·  shifts to G Phrygian @0:30   (confidence 74%)
  VIBE     aggressive · dark · driving · hip-hop · gritty

  PRODUCTION
     Vocals     forward
     Punch      0.62  (moderate)
     Stereo     wide
     Low end    ~55 Hz dominant

  Overall confidence: 88%   ·   analyzed in 0:28 (GPU-accelerated)
```

### 2. Comprehensive JSON (Returned by MCP and CLI by default)
```json
{
  "header": {
    "job_id": "sig_a3f9b2c1",
    "status": "success",
    "confidence_score": 0.88,
    "source_metadata": {
      "title": "Acoustic Vibe Demo",
      "duration_sec": 204,
      "source_type": "file"
    }
  },
  "sonic_signature": {
    "bpm": 153.8,
    "bpm_variable": false,
    "key": "G Major",
    "key_variable": true,
    "key_map": [
      { "start_sec": 0.0,  "end_sec": 30.0, "key": "G Major" },
      { "start_sec": 30.0, "end_sec": 90.0, "key": "G Phrygian" }
    ],
    "mode_confidence": 0.74,
    "vibe_vector": [0.012, -0.034, "... 512 float dimensions ..."],
    "vibe_tags": ["aggressive", "dark", "driving", "hip-hop", "gritty"],
    "production_profile": {
      "vocal_presence": "forward",
      "transient_punch": 0.62,
      "stereo_width": "wide",
      "dominant_freq_peaks_hz": {
        "harmonic": [55.0, 110.2],
        "percussive": [125.0, 250.1]
      }
    }
  },
  "telemetry": {
    "inference_time_sec": 28.0
  }
}
```

---

## ⚡ Key Features

* 🥁 **Tempo & Beat Tracking** — Full BPM computation with variable-tempo drift detection and transient windowing.
* 🎹 **Key & Harmonic Mapping** — Computes structural musical key + mode, generating a detailed `key_map` tracking section-by-section modulations.
* 🌈 **Vibe & Style Embeddings** — Compiles a 512-dimensional CLAP embedding and human-readable style tags (covering energy, texture, mood, and genre) using zero-shot music vocab classification.
* 🎚️ **Production Analytics** — Measures vocal spatial presence, transient punch coefficients, stereo width, and dominant frequency peaks.
* 🤖 **MCP-Native System** — Fully exposes 4 standardized Model Context Protocol tools for instant integration into AI tools.
* 🪶 **Robust Graceful Degradation** — Automatically utilizes a CUDA GPU if present and falls back to CPU; gracefully degrades to HPSS and standard librosa feature arrays if heavy deep learning packages (`[clap]`) are omitted.
* 🔒 **100% Offline & Private** — All conversion, separation, and inference occur locally.

---

## 📦 Installation & Setup

### System Prerequisites
Ensure you have **Python 3.10+** and **FFmpeg** installed and accessible on your system `PATH`.

#### Installing FFmpeg:
* **macOS**: `brew install ffmpeg`
* **Linux (Debian/Ubuntu)**: `sudo apt update && sudo apt install -y ffmpeg`
* **Windows**: Run `winget install Gyan.FFmpeg` via PowerShell (Administrator), or download manually from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin` directory to your system environment variables.

---

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ripunjay-kashyap/audio-sonic-mcp.git
   cd audio-sonic-mcp
   ```

2. **Initialize Virtual Environment**
   ```bash
   python -m venv .venv
   # Activate on macOS/Linux:
   source .venv/bin/activate
   # Activate on Windows (PowerShell):
   .venv\Scripts\activate
   ```

3. **Install Dependencies**
   Choose between the lightweight core engine or the full high-fidelity ML suite:
   
   * **Option A: Full High-Fidelity ML Suite (Recommended)**
     Includes demixing stems (Demucs) and zero-shot vibe vectors (CLAP). Requires ~4 GB disk space.
     ```bash
     pip install -e ".[clap]"
     ```
   * **Option B: Core Lightweight Pipeline**
     Uses standard digital signal processing (HPSS/librosa). Rapid install and minimal footprint.
     ```bash
     pip install -e .
     ```

> [!NOTE]
> The optional `[clap]` stack installs `torch`, `torchaudio`, `transformers`, and `demucs`. Without these, the server automatically switches to light fallbacks (HPSS instead of Demucs, standard feature matrices instead of CLAP vectors, and leaves out `vibe_tags`).

---

## 🤖 MCP Client Configuration Guide

Audio Sonic MCP registers itself as a standard package script. This enables you to run it using the global executable name (`audio-sonic-mcp`) directly from your virtual environment's bin folder, or run the script file manually.

### 1. Claude Desktop Setup
Open your Claude configuration file:
* **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
* **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
* **Linux**: `~/.config/Claude/claude_desktop_config.json`

Add the server to your `mcpServers` object:

```json
{
  "mcpServers": {
    "audio-sonic-mcp": {
      "command": "C:\\path\\to\\audio-sonic-mcp\\.venv\\Scripts\\audio-sonic-mcp.exe",
      "args": [],
      "env": {
        "JOBS_ROOT": "C:\\path\\to\\audio-sonic-mcp\\jobs"
      }
    }
  }
}
```

> [!IMPORTANT]
> **Windows Users**: Always use **double backslashes** (`\\`) in JSON configuration paths. Point the executable directly to the `.exe` inside your `.venv\Scripts\` directory.

---

### 2. Cursor IDE Integration
To integrate Audio Sonic MCP into Cursor's AI pane:
1. Navigate to **Settings** ➔ **Features** ➔ **MCP**.
2. Click **+ Add New MCP Server**.
3. Fill in the parameters:
   * **Name**: `audio-sonic-mcp`
   * **Type**: `command`
   * **Command**: `/path/to/audio-sonic-mcp/.venv/bin/audio-sonic-mcp` (use `.exe` extension on Windows)

---

### 3. Windsurf Integration
Open your Windsurf MCP configurations file (typically found at `~/.codeium/windsurf/mcp_config.json`) and append the configuration:

```json
{
  "mcpServers": {
    "audio-sonic-mcp": {
      "command": "/path/to/audio-sonic-mcp/.venv/bin/python",
      "args": ["/path/to/audio-sonic-mcp/server.py"],
      "env": {
        "JOBS_ROOT": "/path/to/audio-sonic-mcp/jobs"
      }
    }
  }
}
```

---

### 4. Cline (VS Code Extension) Setup
Open Cline's MCP setting file (usually located at `%APPDATA%\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json` or equivalent platform storage) and add:

```json
{
  "mcpServers": {
    "audio-sonic-mcp": {
      "command": "/path/to/audio-sonic-mcp/.venv/bin/audio-sonic-mcp",
      "args": [],
      "env": {
        "JOBS_ROOT": "/path/to/audio-sonic-mcp/jobs"
      }
    }
  }
}
```

---

## 🤖 Interaction Flow for AI Agents & LLMs

LLMs automatically learn how to use this server by reading its exposed tool definitions. Because audio stem separation and CLAP embeddings are computationally demanding, Audio Sonic MCP uses an **Asynchronous Fire-and-Forget Job Pattern**.

### Automated LLM Workflow
```
  [User Prompts LLM]
          │
          ▼
1. Submit URL ──────────────► [Tool: get_sonic_signature]
                                      │ (Returns Job ID instantly)
                                      ▼
2. Notify User ◄───────────── [LLM acknowledges job is queued]
          │
          ├───► 3. Wait 10-15s (Or proceed with other tasks)
          │
          ▼
4. Check Progress ──────────► [Tool: get_job_status]
                                      │ (Checks status: running/success/error)
                                      ▼
5. Present Signature ◄─────── [LLM formats rich output for user]
```

### Natural Prompts to Try
* *"Check the health of my audio-sonic-mcp server to make sure all ML components are ready."*
* *"Submit this YouTube track for sonic analysis: `https://www.youtube.com/watch?v=XXXXXX`."*
* *"Check the progress of my sonic signature job `sig_a1b2c3d4` and summarize the BPM, production width, and vibe once complete."*

---

## 🎚️ CLI Usage (Local Files)

For musicians, engineers, and producers working directly in the terminal, you can analyze a full-length local file directly without running any background servers:

```bash
# Get a visual, musician-friendly sonic signature digest (recommended)
python analyze_file.py "my_demo.wav" --summary

# Print full raw JSON directly to the stdout stream
python analyze_file.py "my_demo.wav"

# Dump JSON payload to a file while keeping the stdout clean
python analyze_file.py "my_demo.wav" > signature.json
```

### CLI Command Options Reference

| Option | Shorthand | Description |
|---|---|---|
| `path` | *None* | Absolute or relative path to the local audio file (Required). |
| `--summary` | `-s` | Print a clean, formatted terminal summary instead of standard JSON. |
| `--no-vector` | *None* | Generate JSON signature but omit the heavy 512-dimension vibe float array. |
| `--out FILE` | `-o` | Output the final JSON signature directly to the specified file. |
| `--keep` | `-k` | Do not delete intermediate WAV files or separated stem files in `jobs/`. |
| `--job-id ID` | `-j` | Explicitly define the internal identifier (useful for batch scripts). |

**Supported File Formats**: `wav`, `mp3`, `flac`, `ogg`, `m4a`, `aac`.

---

## 🔧 Environment Variables Reference

Configure environment options by declaring these variables in your active terminal session, container environment, or the `env` block of your MCP configuration file:

| Variable | Default Value | Description / Practical Use |
|---|---|---|
| `JOBS_ROOT` | `./jobs` | Workspace directory where audio files, temporary converted WAVs, and stems are processed. |
| `KEEP_JOB_FILES` | *Unset* | Set to `1` or `true` to keep separated stem WAVs on disk (adds ~75MB per job, useful for troubleshooting). |
| `FILE_MAX_DURATION_SEC`| `600` | Safety ceiling for local file processing duration (YouTube downloads are capped at 60 minutes). |
| `FFMPEG_BIN` | *Unset* | Path to folder containing the `ffmpeg` binary if it is not present in your system `PATH`. |
| `YTDLP_PROXY` | *Unset* | HTTP/SOCKS proxy string passed directly to `yt-dlp` to bypass rate limits or network blocks. |

---

## 🐳 Docker / Podman Execution

If you prefer to avoid setting up local Python libraries, running via containers encapsulates FFmpeg, yt-dlp, and the core Python dependencies (CPU-based pipeline):

```bash
# Build the container image
docker build -t audio-sonic-mcp .

# Run the MCP server over stdio, mounting local folders for job persistence
docker run -i --rm \
  -v "$(pwd)/jobs:/app/jobs" \
  -v "$(pwd)/models:/app/models" \
  audio-sonic-mcp
```

To connect Claude Desktop to your Docker container, configure `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "audio-sonic-mcp-docker": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "/absolute/path/to/jobs:/app/jobs",
        "-v", "/absolute/path/to/models:/app/models",
        "audio-sonic-mcp"
      ]
    }
  }
}
```

---

## ⚙️ How it Works under the Hood

Audio Sonic MCP pipelines are constructed modularly, using transactional checkpoints to ensure reliability. 

```text
  LLM Agent / Claude Desktop                 Musician (Terminal)
            │                                          │
            │  MCP (stdio JSON-RPC)                    │  analyze_file.py
            ▼                                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Modular 6-Stage Analysis Pipeline                                       │
│                                                                          │
│  Stage 1: Ingestion   │ Pre-checks format, scans duration metadata       │
│  Stage 2: Download    │ Fetches audio tracks via yt-dlp (URLs only)      │
│  Stage 3: Conversion  │ normalizes sample formats to 44.1kHz WAV (FFmpeg)│
│  Stage 4: Separation  │ Splits stems: Vocals, Drums, Bass, Other (Demucs)│
│  Stage 5: Analysis    │ Computes BPM, modulations, key, punch (librosa)  │
│  Stage 6: Embeddings  │ Generates 512-dim zero-shot music vibe tags (CLAP)│
└─────────────────────────────────────┬────────────────────────────────────┘
                                      ▼
             Result Payload: (header · sonic_signature · telemetry)
```

1. **Stem Demixing**: Meta AI's **Demucs (`mdx_extra`)** separates the track into isolation stems (`vocals`, `drums`, `bass`, `other`). If missing, it gracefully drops back to **Harmonic-Percussive Source Separation (HPSS)**.
2. **Analysis engine**: **librosa** extracts rhythmic and tonal structures, matching chord patterns and sub-bass movements against Krumhansl-Schmuckler and Phrygian template engines.
3. **Semantic Vibe Tagging**: **LAION CLAP** (`laion/larger_clap_music_and_speech`) runs zero-shot inference against high-coverage aesthetic descriptors (moods, textures, genres), choosing top candidates across stylistic poles.

---

## 🩺 Resiliency & Troubleshooting

### 1. One-Time Setup Download Delays
Upon the **very first analysis job** utilizing the full ML pipeline, `demucs` and `transformers` will download their pre-trained model weights (approximately **400 MB** for Demucs, and **200 MB** for CLAP). 
* The server redirects download progress indicators to `stderr` so they **do not corrupt** the JSON-RPC standard stream.
* During this download, `get_job_status` will remain in `running`. Allow 1–3 minutes depending on your network speed. Subsequent startups take under **10 seconds**.

### 2. FastMCP Concurrency Controls
Model inference on multi-staged architectures is highly CPU/VRAM intensive. To protect consumer hardware and virtual environments from crashing (OutOfMemory exceptions), Audio Sonic MCP enforces a strict global serialization lock (`CONCURRENCY_LOCK`). 
* If you submit multiple URLs simultaneously, they will be processed **sequentially**. 
* Polling `get_job_status` for subsequent jobs will report `queued` or `running` while they wait in the pipeline queue.

### 3. Windows Librosa Deadlock Fix
FastMCP thread dispatching under Windows can cause Numba compilation deadlocks inside background worker threads. To prevent this, Audio Sonic MCP incorporates a **Pre-warming Routine** (`_prewarm_librosa()` and `_prewarm_demucs()`) on launch. It forces JIT compile of resampling, HPSS, and mono-mixing functions in the main thread before starting the RPC listener.

### 4. Diagnosing with `check_health`
If the server reports as `degraded` or tools are missing, call the `check_health` tool or check CLI warnings. It queries:
* Availability of `ffmpeg` on the execution path.
* Installation status of Python packages (`librosa`, `soundfile`, `mcp`, etc.).
* Access permissions to the `JOBS_ROOT` directory.

---

## 🛠️ Development & Testing

Run unit tests inside your virtual environment to verify the mathematical pipelines using synthesized audio waveforms:

```bash
# Install development test framework
pip install -e ".[dev]"

# Execute full suite (requires no network or model downloads)
pytest

# Test specifically CLI execution code paths
pytest tests/test_cli.py
```

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.

© 2026 Ripunjay Kashyap. All rights reserved.
