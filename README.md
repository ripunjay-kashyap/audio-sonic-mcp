# Audio Stem Splitter — MCP Server

Converts any YouTube URL into separated audio stems + a full sonic signature JSON, powered by **yt-dlp**, **FFmpeg**, **Demucs**, and **librosa**.

---

## What It Does

1. Downloads audio from a YouTube URL via `yt-dlp`
2. Converts to 44.1kHz WAV via FFmpeg
3. Separates into stems (vocals, drums, bass, other) via Demucs
4. Analyzes for BPM, musical key, transient punch, stereo width, dominant frequencies
5. Generates a 512-dim CLAP vibe vector (falls back to librosa mel+MFCC if CLAP unavailable)
6. Returns everything as a single JSON payload

---

## Architecture

```
YouTube URL
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1 — Ingestion         validate_source()          │
│  Stage 2 — Download          yt-dlp  (audio-only)       │
│  Stage 3 — Convert           FFmpeg → 44.1kHz WAV       │
│  Stage 4 — Split             Demucs htdemucs            │
│  Stage 5 — Analyze           librosa (BPM, key, etc.)   │
│  Stage 6 — Vectorize         CLAP or librosa fallback   │
└─────────────────────────────────────────────────────────┘
     │
     ▼
  JSON Payload (stems + sonic_signature + telemetry)
```

---

## System Requirements

| Dependency | Notes |
|-----------|-------|
| Python 3.10+ | |
| FFmpeg | Must be on PATH for local installs; bundled in container |
| yt-dlp | Installed via pip |
| Demucs | Installed via pip |
| torch / transformers / torchaudio | Optional — enables CLAP vibe vectors (~4 GB download) |

---

## Installation

### Option A — Local (Python)

```bash
# Create virtual environment
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: CLAP vibe vectors
pip install transformers torch torchaudio
```

Run the server:
```bash
python server.py

# Custom output directory
STEMS_ROOT=/data/stems python server.py
```

### Option B — Container (Podman) — recommended on Windows

Podman runs rootless/daemonless and is fully OCI-compatible. On Windows it uses WSL.

```bash
# Start the Podman machine (first time or after a reboot)
podman machine start

# Verify it's running
podman machine list

# Build the image
podman build -t audio-stem-mcp .

# Create persistent directories, then run
mkdir -p stems models

podman run -it --rm \
  -v $(pwd)/stems:/app/stems:Z \
  -v $(pwd)/models:/app/models:Z \
  audio-stem-mcp
```

> **`:Z` flag** — sets the SELinux relabel on volume mounts so the container can write to host directories. Required on SELinux-enabled systems (including Podman's WSL VM). Omit it when using plain Docker.

### Option C — Container (Docker)

```bash
docker build -t audio-stem-mcp .

mkdir -p stems models

docker run -it --rm \
  -v $(pwd)/stems:/app/stems \
  -v $(pwd)/models:/app/models \
  audio-stem-mcp
```

> **First run downloads ~4–6 GB of models** (Demucs + CLAP). Always mount `/app/models` so they persist across restarts.

---

## Claude Desktop Integration

Add to your Claude Desktop config:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "audio-stem-mcp": {
      "command": "/path/to/audio-stem-mcp/.venv/bin/python",
      "args": ["/path/to/audio-stem-mcp/server.py"],
      "env": {
        "STEMS_ROOT": "/tmp/audio_stems"
      }
    }
  }
}
```

On Windows use `.venv\Scripts\python.exe` as the command path.

---

## Smoke Testing

```bash
pip install "mcp[cli]"
mcp dev server.py
```

Opens a browser-based MCP inspector where you can call tools directly. Server logs go to stderr — watch that terminal for pipeline progress.

To test inside a container using Podman:
```bash
podman run -it --rm \
  -v $(pwd)/stems:/app/stems:Z \
  -v $(pwd)/models:/app/models:Z \
  -p 5173:5173 -p 3000:3000 \
  --entrypoint mcp \
  audio-stem-mcp dev server.py
```

---

## MCP Tools

### `split_audio`
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | string | Yes | YouTube URL |
| `job_id` | string | No | Custom ID (auto-generated if omitted) |
| `model` | enum | No | Demucs model (default: `htdemucs`) |

### `get_job_status`
| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | ID returned by `split_audio` |

### `list_jobs`
Lists all jobs and their statuses for the current server session.

---

## Response Payload

```json
{
  "header": {
    "job_id": "sig_a3f9b2c1",
    "status": "success",
    "confidence_score": 0.91,
    "source": {
      "title": "Track Name",
      "uploader": "Artist",
      "duration_sec": 214
    }
  },
  "stems_metadata": {
    "local_root": "/tmp/audio_stems/sig_a3f9b2c1/stems/",
    "files": ["vocals.wav", "drums.wav", "bass.wav", "other.wav"],
    "sdr_ratio": 8.6
  },
  "sonic_signature": {
    "bpm": 128.05,
    "key": "F# Minor",
    "vibe_vector": [0.12, -0.45, 0.88, "...512 dims..."],
    "production_profile": {
      "vocal_presence": "forward",
      "drum_transient_punch": 0.82,
      "stereo_width": "wide",
      "dominant_freq_peaks_hz": {
        "vocals": [261.5, 523.0, 880.0, 440.0, 196.0],
        "drums": [60.0, 120.0, 8000.0, 4000.0, 2000.0],
        "bass": [80.0, 160.0, 240.0, 55.0, 320.0],
        "other": [440.0, 880.0, 1760.0, 220.0, 660.0]
      }
    }
  },
  "telemetry": {
    "cpu_usage_avg": "72%",
    "inference_time_sec": 112.4
  }
}
```

---

## Demucs Models

| Model | Stems | Speed | Quality |
|-------|-------|-------|---------|
| `htdemucs` (default) | 4 | Fastest | Good |
| `htdemucs_ft` | 4 | Slowest | Best |
| `htdemucs_6s` | 6 (+ guitar, piano) | Slow | Good |
| `mdx_extra` | 4 | Medium | Good (alt architecture) |

---

## CLAP Vibe Vector

The vibe vector is a 512-dimension semantic audio embedding from the [LAION CLAP model](https://huggingface.co/laion/larger_clap_music_and_speech). It maps audio into a shared latent space with text, enabling:

- Semantic similarity search ("find tracks that sound like this")
- Genre/mood clustering
- Cross-modal retrieval (audio ↔ text)

If CLAP is not installed, the server falls back to a librosa-derived mel+MFCC composite.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STEMS_ROOT` | `/tmp/audio_stems` | Root directory for downloaded and separated audio |
| `TORCH_HOME` | `/app/models/torch` | Demucs model cache (container default) |
| `HF_HOME` | `/app/models/huggingface` | CLAP model cache (container default) |

---

## Notes

- Jobs are tracked **in memory only** — restarting the server clears all job history.
- The server communicates over **stdio** (MCP protocol) — running it directly produces no interactive output.
- The 60-minute duration limit is enforced at ingestion to prevent runaway jobs.
