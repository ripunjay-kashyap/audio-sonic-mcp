# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An MCP (Model Context Protocol) server that accepts a YouTube URL and returns separated audio stems (vocals, drums, bass, other) plus a structured JSON "sonic signature" containing BPM, musical key, stereo width, transient punch, dominant frequencies, and a 512-dim CLAP vibe vector.

## System Requirements

- Python 3.10+
- FFmpeg (must be on PATH)
- yt-dlp (installed via pip)
- Demucs (installed via pip)
- Optional: `transformers`, `torch`, `torchaudio` for CLAP vibe vectors (~4GB download)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: CLAP vibe vectors
pip install transformers torch torchaudio
```

## Running the Server

```bash
python server.py

# Custom stems output directory
STEMS_ROOT=/data/stems python server.py
```

The server communicates over stdio (MCP protocol) — running it directly prints nothing useful interactively. Use Claude Desktop or an MCP client to exercise the tools.


## Claude Desktop Integration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "audio-stem-mcp": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/path/to/audio-stem-mcp/server.py"],
      "env": {
        "STEMS_ROOT": "/tmp/audio_stems"
      }
    }
  }
}
```

On Windows use the `.venv\Scripts\python.exe` path.

## Testing

There are no automated tests. To smoke-test the server interactively:

```bash
# Requires mcp[cli] — install with: pip install "mcp[cli]"
mcp dev server.py
```

This opens a browser-based MCP inspector where you can call `split_audio`, `get_job_status`, and `list_jobs` directly. The server logs to stderr — watch that terminal for pipeline stage progress.

## Docker

A Dockerfile is provided to run the server in a containerized environment with a CPU-optimized `torch` build and statically compiled FFmpeg.

### Build
```bash
docker build -t audio-stem-mcp .
```

### Run
**Important:** Demucs and CLAP will download gigabytes of pre-trained models on their first run. To prevent models from being downloaded every time the container restarts, you **must** mount a volume to `/app/models`.

```bash
# Create local directories for persistence
mkdir -p stems models

# Run the container with volumes
docker run -it --rm \
  -v $(pwd)/stems:/app/stems \
  -v $(pwd)/models:/app/models \
  audio-stem-mcp
```

## Architecture

The pipeline is linear — each stage feeds into the next, all orchestrated by `server.py`:

| Stage | File | Responsibility |
|-------|------|---------------|
| 1 — Validate | `ingestion.py` | URL validation + yt-dlp metadata probe; enforces 60-min limit |
| 2 — Download | `downloader.py` | yt-dlp audio-only download to `STEMS_ROOT/<job_id>/` |
| 3 — Convert | `converter.py` | FFmpeg → 44.1kHz WAV |
| 4 — Split | `splitter.py` | Demucs stem separation; handles both flat and nested output layouts; computes proxy SDR |
| 5 — Analyze | `analyzer.py` | librosa BPM (from drums stem), key (from vocals+other+bass), transient punch, per-stem frequency peaks, stereo width, vocal presence |
| 6 — Vectorize | `vectorizer.py` | CLAP 512-dim embedding via `laion/larger_clap_music_and_speech`; falls back to a librosa mel+MFCC composite if CLAP is unavailable |
| Assemble | `assembler.py` | Merges all outputs into the canonical JSON payload |

All pipeline stages run via `asyncio.to_thread()` in `server.py` — they are synchronous functions wrapped for async execution.

## Job Store

Jobs are tracked in an in-memory dict `JOB_STORE` in `server.py`. It is not persisted — restarting the server clears all job history.

## MCP Tools Exposed

- `split_audio(url, job_id?, model?)` — runs the full pipeline
- `get_job_status(job_id)` — fetches result from `JOB_STORE`
- `list_jobs()` — lists all job statuses

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STEMS_ROOT` | `/tmp/audio_stems` | Root directory for all downloaded and separated audio files |

## Demucs Model Options

- `htdemucs` (default) — 4 stems, fastest
- `htdemucs_ft` — 4 stems, highest quality, slowest
- `htdemucs_6s` — 6 stems (adds guitar, piano)
- `mdx_extra` — 4 stems, alternative architecture
