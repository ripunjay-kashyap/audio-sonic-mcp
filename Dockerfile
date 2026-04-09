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

# Install system dependencies required by soundfile, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy FFmpeg binaries from Stage 1
COPY --from=ffmpeg /ffmpeg /usr/local/bin/
COPY --from=ffmpeg /ffprobe /usr/local/bin/

# Install Python dependencies
# 1. Install CPU-only torch to save space
RUN pip install --no-cache-dir torch torchaudio transformers --index-url https://download.pytorch.org/whl/cpu

# 2. Install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories for mounting
RUN mkdir -p /app/stems /app/models/torch /app/models/huggingface

# Copy server code
COPY . /app

# The MCP inspector typically uses stdio communication. 
# We run the FastMCP app directly or via MCP CLI.
ENTRYPOINT ["python", "server.py"]
