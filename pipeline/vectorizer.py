"""
Stage 6 — Vibe Vectorization
Passes stems through a local CLAP (Contrastive Language-Audio Pretraining)
model to generate a 512-dimension semantic embedding — the "Vibe Vector."

CLAP maps audio into the same latent space as text, enabling similarity
searches like "find tracks that sound like this."

Falls back gracefully to a librosa-derived feature vector if CLAP is unavailable.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"
VECTOR_DIM = 512


def generate_vibe_vector(stems_dir: Path, stem_files: list[str]) -> list[float]:
    """
    Generates a 512-dim vibe vector from the stem mix.

    Strategy:
    1. Try CLAP (laion/larger_clap_music_and_speech) via transformers
    2. Fallback: librosa mel-spectrogram embedding (mean-pooled, PCA-reduced)
    """
    try:
        return _clap_vector(stems_dir, stem_files)
    except Exception as exc:
        logger.warning("CLAP unavailable (%s). Falling back to librosa embedding.", exc)
        return _librosa_fallback_vector(stems_dir, stem_files)


# ── CLAP path ─────────────────────────────────────────────────────────────────


def _clap_vector(stems_dir: Path, stem_files: list[str]) -> list[float]:
    from transformers import ClapModel, ClapProcessor
    import torch

    logger.info("Loading CLAP model: %s", CLAP_MODEL_ID)
    processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
    model = ClapModel.from_pretrained(CLAP_MODEL_ID)
    model.eval()

    # Mix all stems into a single mono signal for CLAP
    mix = _load_mix(stems_dir, stem_files, sr=48000)

    inputs = processor(audios=[mix], sampling_rate=48000, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.get_audio_features(**inputs)

    vec = embeddings[0].numpy().tolist()
    logger.info("CLAP vector generated: %d dims", len(vec))
    return [round(v, 6) for v in vec]


# ── Librosa fallback ──────────────────────────────────────────────────────────


def _librosa_fallback_vector(stems_dir: Path, stem_files: list[str]) -> list[float]:
    """
    Generates a compact 512-dim representation using:
    - Mel-spectrogram statistics (mean + std across time for 128 bands = 256 dims)
    - MFCC statistics (mean + std for 40 coefficients = 80 dims)
    - Chroma statistics (mean + std for 12 bins = 24 dims)
    - Spectral stats: centroid, rolloff, bandwidth, flatness, ZCR (mean+std each = 10 dims)
    - Tonnetz (mean + std for 6 dims = 12 dims)
    Total: 256 + 80 + 24 + 10 + 12 = 382 → zero-padded to 512
    """
    import librosa

    mix = _load_mix(stems_dir, stem_files, sr=22050)
    y = mix.astype(np.float32)

    features: list[float] = []

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=22050, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.extend(mel_db.mean(axis=1).tolist())
    features.extend(mel_db.std(axis=1).tolist())  # 256

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=40)
    features.extend(mfccs.mean(axis=1).tolist())
    features.extend(mfccs.std(axis=1).tolist())  # 80

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=22050)
    features.extend(chroma.mean(axis=1).tolist())
    features.extend(chroma.std(axis=1).tolist())  # 24

    # Spectral features
    for feat in [
        librosa.feature.spectral_centroid(y=y, sr=22050),
        librosa.feature.spectral_rolloff(y=y, sr=22050),
        librosa.feature.spectral_bandwidth(y=y, sr=22050),
        librosa.feature.spectral_flatness(y=y),
        librosa.feature.zero_crossing_rate(y),
    ]:
        features.append(float(feat.mean()))
        features.append(float(feat.std()))  # 10

    # Tonnetz (uses the raw mix instead of expensive HPSS since Demucs is already handling separation in earlier stages)
    tonnetz = librosa.feature.tonnetz(y=y, sr=22050)
    features.extend(tonnetz.mean(axis=1).tolist())
    features.extend(tonnetz.std(axis=1).tolist())  # 12

    # Pad / trim to exactly 512 dims
    if len(features) < VECTOR_DIM:
        features.extend([0.0] * (VECTOR_DIM - len(features)))
    elif len(features) > VECTOR_DIM:
        features = features[:VECTOR_DIM]

    # L2 normalize
    arr = np.array(features, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr /= norm

    logger.info("Fallback librosa vector generated: %d dims", len(arr))
    return [round(float(v), 6) for v in arr]


# ── Shared utility ────────────────────────────────────────────────────────────


def _load_mix(stems_dir: Path, stem_files: list[str], sr: int) -> np.ndarray:
    """Loads and mixes all stems into a single mono array at given sample rate."""
    import librosa

    mix = None
    for fname in stem_files:
        path = stems_dir / fname
        y, _ = librosa.load(str(path), sr=sr, mono=True)
        mix = y if mix is None else mix[: len(y)] + y[: len(mix)]

    if mix is None:
        return np.zeros(sr * 10, dtype=np.float32)

    # Peak normalize
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix /= peak
    return mix
