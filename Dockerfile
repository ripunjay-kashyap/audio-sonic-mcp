# Stage 1: Get static FFmpeg binaries
FROM mwader/static-ffmpeg:6.1 AS ffmpeg

# Stage 2: Final runtime environment
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Set consolidated cache folders for downloaded models (Demucs & CLAP)
ENV TORCH_HOME=/app/models/torch
ENV HF_HOME=/app/models/huggingface
# Set output directory
ENV STEMS_ROOT=/app/stems

WORKDIR /app

# Create non-root user early so we can assign ownership
RUN useradd -m -u 1000 appuser

# Install system dependencies required by soundfile, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy FFmpeg binaries from Stage 1
COPY --from=ffmpeg /ffmpeg /usr/local/bin/
COPY --from=ffmpeg /ffprobe /usr/local/bin/

# Install Python dependencies
# 1. Install base requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Install CPU-only torch/transformers (large; kept separate for layer caching)
RUN pip install --no-cache-dir torch torchaudio transformers --index-url https://download.pytorch.org/whl/cpu

# Create mount-point directories and hand ownership to appuser
RUN mkdir -p /app/stems /app/models/torch /app/models/huggingface && \
    chown -R appuser:appuser /app

# Copy server code and fix ownership
COPY --chown=appuser:appuser . /app

# Drop to non-root
USER appuser

# Liveness probe: verify Python imports and FFmpeg are functional
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import librosa, demucs, mcp; import subprocess, sys; sys.exit(0 if subprocess.run(['ffmpeg','-version'], capture_output=True).returncode == 0 else 1)"

# The MCP server communicates over stdio.
ENTRYPOINT ["python", "server.py"]
