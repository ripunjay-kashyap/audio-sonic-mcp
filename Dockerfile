# Stage 1: Get static FFmpeg binaries
FROM mwader/static-ffmpeg:6.1 AS ffmpeg

# Stage 2: Final runtime environment
# 3.12 matches the dev/test env the deps were pinned against (requirements.txt
# pins numpy==2.4.4, which requires Python >= 3.11).
FROM python:3.12-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Set consolidated cache folders for downloaded models (Demucs & CLAP)
ENV TORCH_HOME=/app/models/torch
ENV HF_HOME=/app/models/huggingface
# Job working directory: downloaded audio, WAV, stems, metadata (mount a volume here)
ENV JOBS_ROOT=/app/jobs

WORKDIR /app

# Create non-root user early so we can assign ownership
RUN useradd -m -u 1000 appuser

# Install system dependencies required by soundfile
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

# 2. CPU-only torch/torchaudio from the PyTorch wheel index (Demucs + CLAP run on
#    these). Pinned to the versions the project is tested against.
RUN pip install --no-cache-dir torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
# 3. Demucs (stem separation) + transformers (CLAP) from PyPI — torch already
#    satisfied above. transformers is NOT on the PyTorch index, so it must come
#    from PyPI; a single --index-url command would fail to find it.
RUN pip install --no-cache-dir demucs==4.0.1 transformers==5.8.0

# Create mount-point directories and hand ownership to appuser
RUN mkdir -p /app/jobs /app/models/torch /app/models/huggingface && \
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
