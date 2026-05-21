"""
Stage 5 — Vibe Vectorization
Passes the WAV file through a local CLAP (Contrastive Language-Audio Pretraining)
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

# Vibe-tag vocabulary: (word, axis). Ranking is done PER AXIS (see
# VIBE_TAG_AXIS_COUNTS) so the output always spans mood/energy/genre/texture
# instead of collapsing to whichever prompt frame scores highest. Genre words
# use the "a {} track" prompt frame; all others use "this music sounds {}".
VIBE_TAG_VOCAB: list[tuple[str, str]] = [
    # Mood
    ("dark", "mood"), ("bright", "mood"), ("melancholic", "mood"),
    ("uplifting", "mood"), ("aggressive", "mood"), ("chill", "mood"),
    ("dreamy", "mood"), ("tense", "mood"), ("warm", "mood"),
    ("romantic", "mood"), ("melodic", "mood"),
    # Energy
    ("energetic", "energy"), ("mellow", "energy"), ("driving", "energy"),
    ("laid-back", "energy"), ("hard-hitting", "energy"), ("smooth", "energy"),
    # Genre
    ("hip-hop", "genre"), ("trap", "genre"), ("lo-fi", "genre"), ("jazz", "genre"),
    ("rock", "genre"), ("electronic", "genre"), ("ambient", "genre"),
    ("soul", "genre"), ("R&B", "genre"), ("pop", "genre"), ("funk", "genre"),
    ("classical", "genre"),
    # Texture
    ("gritty", "texture"), ("clean", "texture"), ("distorted", "texture"),
    ("acoustic", "texture"), ("synthetic", "texture"), ("lush", "texture"),
    ("sparse", "texture"),
]
# Tags taken from each axis (total = 5) — guarantees a spread across categories.
VIBE_TAG_AXIS_COUNTS: dict[str, int] = {"mood": 2, "energy": 1, "genre": 1, "texture": 1}
VIBE_TAG_FLOOR = 0.0  # cosine floor; calibrate upward to drop weak per-axis picks



def generate_vibe_vector(wav_path: Path, full_song: bool = False) -> list[float]:
    """
    Generates a 512-dim vibe vector from the input WAV file.

    Strategy:
    1. Try CLAP (laion/larger_clap_music_and_speech) via transformers
    2. Fallback: librosa mel-spectrogram embedding (mean-pooled, PCA-reduced)
    """
    try:
        return _clap_vector(wav_path, full_song=full_song)
    except Exception as exc:
        logger.warning("CLAP unavailable (%s). Falling back to librosa embedding.", exc)
        return _librosa_fallback_vector(wav_path, full_song=full_song)


# ── CLAP path ─────────────────────────────────────────────────────────────────


def _clap_vector(wav_path: Path, full_song: bool = False) -> list[float]:
    from transformers import ClapModel, ClapProcessor
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading CLAP model: %s (device=%s)", CLAP_MODEL_ID, device)
    processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
    model = ClapModel.from_pretrained(CLAP_MODEL_ID).to(device)
    model.eval()

    audio = _load_audio(wav_path, sr=48000, full_song=full_song)

    inputs = processor(audio=[audio], sampling_rate=48000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        raw = model.get_audio_features(**inputs)

    # Unwrap output objects (transformers ≥5.x may return a dataclass)
    if hasattr(raw, 'audio_embeds'):
        raw = raw.audio_embeds
    elif hasattr(raw, 'pooler_output'):
        raw = raw.pooler_output

    arr = raw.detach().cpu().numpy().reshape(-1)
    vec = arr.tolist()
    logger.info("CLAP vector generated: %d dims", len(vec))
    return [round(float(v), 6) for v in vec]


# ── Librosa fallback ──────────────────────────────────────────────────────────


def _librosa_fallback_vector(wav_path: Path, full_song: bool = False) -> list[float]:
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

    y = _load_audio(wav_path, sr=22050, full_song=full_song)
    y = y.astype(np.float32)

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

    # Tonnetz
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


# ── Vibe tags (CLAP zero-shot) ──────────────────────────────────────────────


def generate_vibe_tags(wav_path: Path, full_song: bool = False) -> "list[str] | None":
    """Top vibe/mood/genre words via CLAP zero-shot. Returns None only when
    CLAP is unavailable (the librosa fallback vector is not in CLAP space)."""
    embs = _clap_tag_embeddings(wav_path, full_song=full_song)
    if embs is None:
        logger.info("vibe_tags: CLAP unavailable — returning None")
        return None
    audio_emb, text_embs = embs
    tags = _select_tags_per_axis(
        audio_emb, text_embs, VIBE_TAG_VOCAB, VIBE_TAG_AXIS_COUNTS, VIBE_TAG_FLOOR
    )
    logger.info("vibe_tags: %s", ", ".join(tags))
    return tags


def _clap_tag_embeddings(
    wav_path: Path, full_song: bool = False
) -> "tuple[np.ndarray, np.ndarray] | None":
    """Return (audio_embedding (D,), text_embeddings (N, D)) from CLAP, or None
    if CLAP/transformers is unavailable or errors."""
    try:
        from transformers import ClapModel, ClapProcessor
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
        model = ClapModel.from_pretrained(CLAP_MODEL_ID).to(device)
        model.eval()

        audio = _load_audio(wav_path, sr=48000, full_song=full_song)
        prompts = [_vibe_prompt(word, axis) for word, axis in VIBE_TAG_VOCAB]

        a_in = processor(audio=[audio], sampling_rate=48000, return_tensors="pt")
        a_in = {k: v.to(device) for k, v in a_in.items()}
        t_in = processor(text=prompts, return_tensors="pt", padding=True)
        t_in = {k: v.to(device) for k, v in t_in.items()}

        with torch.no_grad():
            audio_feat = model.get_audio_features(**a_in)
            text_feat = model.get_text_features(**t_in)

        audio_np = _embed_to_numpy(audio_feat).reshape(-1)
        text_np = _embed_to_numpy(text_feat)
        return audio_np, text_np
    except Exception as exc:
        logger.warning("vibe_tags: CLAP path failed (%s)", exc)
        return None


def _embed_to_numpy(raw) -> np.ndarray:
    """Unwrap a transformers CLAP feature output to a numpy array."""
    if hasattr(raw, "audio_embeds"):
        raw = raw.audio_embeds
    elif hasattr(raw, "text_embeds"):
        raw = raw.text_embeds
    elif hasattr(raw, "pooler_output"):
        raw = raw.pooler_output
    return raw.detach().cpu().numpy()


def _vibe_prompt(word: str, axis: str) -> str:
    """CLAP text prompt for a vocab word. Genres read better as 'a X track';
    mood/energy/texture as 'this music sounds X'."""
    return f"a {word} track" if axis == "genre" else f"this music sounds {word}"


def _select_tags_per_axis(
    audio_emb: np.ndarray,
    text_embs: np.ndarray,
    vocab: list[tuple[str, str]],
    axis_counts: dict[str, int],
    floor: float,
) -> list[str]:
    """Pick the top tags *per axis* (mood/energy/genre/texture) by cosine
    similarity, so the result spans categories instead of collapsing to one.
    Each axis competes only against itself, so cross-frame cosine bias (genre
    "a X track" prompts scoring higher than mood "sounds X" prompts) can no
    longer crowd out an axis. Returns words in axis order, within-axis by
    similarity; falls back to the single global top word if nothing clears
    ``floor``."""
    a = audio_emb / (np.linalg.norm(audio_emb) or 1.0)
    t = text_embs / (np.linalg.norm(text_embs, axis=1, keepdims=True) + 1e-9)
    sims = t @ a  # (N,)

    by_axis: dict[str, list[tuple[float, str]]] = {}
    for i, (word, axis) in enumerate(vocab):
        by_axis.setdefault(axis, []).append((float(sims[i]), word))

    selected: list[str] = []
    for axis in ("mood", "energy", "genre", "texture"):
        ranked = sorted(by_axis.get(axis, []), reverse=True)
        for sim, word in ranked[: axis_counts.get(axis, 0)]:
            if sim >= floor:
                selected.append(word)

    if not selected:
        return [vocab[int(np.argmax(sims))][0]]
    return selected


# ── Shared utility ────────────────────────────────────────────────────────────


def _load_audio(wav_path: Path, sr: int, full_song: bool = False) -> np.ndarray:
    """Loads the analysis-window slice of a WAV as a mono float32 array.
    See ``pipeline.window`` for window selection (offset + duration vary
    with track length). Uses soundfile directly to avoid librosa's audioread
    fallback, which spawns FFmpeg and deadlocks when stdin is an MCP pipe.
    """
    import soundfile as sf
    import librosa
    from pipeline.window import pick_window

    with sf.SoundFile(str(wav_path)) as snd:
        native_sr = snd.samplerate
        if full_song:
            raw = snd.read(dtype="float32", always_2d=True)
        else:
            offset_frames, frames_to_read = pick_window(snd.frames, native_sr)
            snd.seek(offset_frames)
            raw = snd.read(frames=frames_to_read, dtype="float32", always_2d=True)

    # raw: (frames, channels) → mono
    y = librosa.to_mono(raw.T)
    if native_sr != sr:
        y = librosa.resample(y, orig_sr=native_sr, target_sr=sr)

    # Peak normalize
    peak = np.max(np.abs(y))
    if peak > 0:
        y /= peak
    return y
