# Audio Sonic Signature — MCP Server

An MCP (Model Context Protocol) server that accepts a YouTube URL and returns a structured **sonic signature** JSON — giving an LLM like Claude rich musical context about any song from a single tool call.

---

## What Is a Sonic Signature?

```json
{
  "header": {
    "job_id": "sig_a1b2c3d4",
    "status": "success",
    "confidence_score": 0.91,
    "source_metadata": {
      "title": "Song Title - Artist",
      "uploader": "ArtistChannel",
      "duration_sec": 214,
      "url": "https://www.youtube.com/watch?v=...",
      "genre_hint": "Music"
    }
  },
  "sonic_signature": {
    "bpm": 95.7,
    "key": "E Minor",
    "mode_confidence": 0.87,
    "vibe_vector": [0.012, -0.034, ...],
    "production_profile": {
      "vocal_presence": "forward",
      "transient_punch": 0.42,
      "stereo_width": "wide",
      "dominant_freq_peaks_hz": {
        "harmonic": [220.0, 440.0, 880.0],
        "percussive": [60.0, 120.0, 8000.0]
      }
    }
  },
  "telemetry": {
    "inference_time_sec": 38.2
  }
}
```

| Field | Description |
|---|---|
| `bpm` | Tempo via autocorrelation beat tracking on the percussive signal |
| `key` | Musical key via weighted chroma fusion on the harmonic signal |
| `mode_confidence` | Pearson correlation score for the detected key [0–1] |
| `vibe_vector` | 512-dim CLAP semantic embedding — enables cross-modal similarity search |
| `transient_punch` | Onset strength 97th-percentile peak-to-mean ratio |
| `stereo_width` | L/R channel correlation |
| `dominant_freq_peaks_hz` | Top frequency peaks per harmonic/percussive signal |

---

## Architecture

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
|  5-Stage Pipeline                                                        |
|                                                                          |
|  Stage 1 — Validate    yt-dlp probe         -->  metadata.json          |
|  Stage 2 — Download    yt-dlp audio stream  -->  raw audio file         |
|  Stage 3 — Convert     FFmpeg               -->  44.1 kHz WAV           |
|  Stage 4 — Analyze     librosa HPSS + MIR   -->  BPM, key, features    |
|  Stage 5 — Vectorize   CLAP (laion)         -->  512-dim vibe vector    |
+----------------------------------+---------------------------------------+
                                   |
                                   v
+--------------------------------------------------------------------------+
|  Sonic Signature JSON                                                    |
|  header:           job_id, status, confidence_score, source_metadata    |
|  sonic_signature:  BPM, key, vibe_vector, production_profile            |
|  telemetry:        inference_time_sec                                    |
+--------------------------------------------------------------------------+
```

**Stage 4 — Analysis:** Uses `librosa.effects.hpss()` to separate the audio into harmonic and percussive signals in memory. BPM is extracted from the percussive signal (cleaner beat tracking), key from the harmonic signal (chroma uncontaminated by drums). All other features run on the appropriate signal. No files written — analysis is entirely in-memory.

**Stage 5 — Vectorization:** Passes the full audio mix through **CLAP** (`laion/larger_clap_music_and_speech`) to produce a 512-dim embedding that maps audio into the same latent space as text — enabling semantic similarity search. Falls back to a librosa mel+MFCC composite if CLAP is unavailable.

---

## System Requirements

- Python 3.10+
- FFmpeg (must be on PATH)
- yt-dlp (installed via pip)

**Optional — CLAP vibe vectors (~4 GB download):**
```bash
pip install transformers torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# or: pip install ".[clap]"
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux
pip install -r requirements.txt
```

---

## Running the Server

```bash
python server.py

# Custom job directory
JOBS_ROOT=/data/jobs python server.py
```

The server communicates over stdio (MCP protocol). Use Claude Desktop or an MCP client to exercise the tools.

---

## Claude Desktop Integration

Add to `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "audio-sonic-mcp": {
      "command": "C:\\path\\to\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\server.py"],
      "env": {
        "JOBS_ROOT": "C:\\tmp\\audio_jobs"
      }
    }
  }
}
```

---

## MCP Tools

| Tool | Description |
|---|---|
| `get_sonic_signature(url, job_id?)` | Runs the full pipeline and returns the sonic signature JSON |
| `get_job_status(job_id)` | Fetches a completed or in-progress job result |
| `list_jobs()` | Lists all jobs in the current server session |
| `check_health()` | Verifies FFmpeg, yt-dlp, and Python package availability |

---

## Testing

```bash
pytest                 # unit test suite (synthetic audio, no downloads)
pytest -v tests/       # verbose

python smoke_test.py   # full end-to-end via MCP stdio
```

---

## Container (Podman)

```bash
# Build
podman build -t audio-sonic-mcp .

# Run
podman run -it --rm \
  -v $(pwd)/jobs:/app/jobs:Z \
  -v $(pwd)/models:/app/models:Z \
  audio-sonic-mcp
```

> The `:Z` suffix sets the SELinux label. Use `docker` and drop `:Z` for Docker compatibility.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `JOBS_ROOT` | `./jobs` | Root directory for job working files |
| `TORCH_HOME` | `~/.cache/torch` | Cache directory for CLAP model weights |
| `HF_HOME` | `~/.cache/huggingface` | Cache directory for HuggingFace model weights |

---

## Performance

On a modern CPU (no GPU required):

| Stage | Time |
|---|---|
| Validate + Download + Convert | ~8s |
| HPSS + Analyze | ~15–25s |
| CLAP Vectorize | ~12s |
| **Total** | **~35–45s** |
