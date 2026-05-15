"""
Stage 3.5 — Stem Separation
Runs Demucs mdx_extra via the Python API (not subprocess) to isolate
vocals, drums, bass, and other stems from the first 60s of the input WAV.

The model is loaded once at server startup and reused across calls.
Falls back gracefully (returns None) on any failure so the analyzer uses HPSS.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "mdx_extra"
STEMS = ["vocals", "drums", "bass", "other"]

# Module-level model cache — loaded once during pre-warm, reused per call
_demucs_model = None
_demucs_device: str = "cpu"


def load_demucs_model():
    """Download (if needed) and cache the Demucs model in memory. Call from main thread.

    Moves the model to CUDA when available so separation can run on GPU
    (10-20× faster than CPU for `mdx_extra`).  Falls back to CPU silently.
    """
    global _demucs_model, _demucs_device
    if _demucs_model is not None:
        return
    import torch
    from demucs.pretrained import get_model
    _demucs_device = "cuda" if torch.cuda.is_available() else "cpu"
    _demucs_model = get_model(MODEL_NAME)
    _demucs_model.to(_demucs_device)
    _demucs_model.eval()
    logger.info("separator: Demucs %s model loaded (device=%s)", MODEL_NAME, _demucs_device)


def separate_stems(wav_path: Path) -> "Path | None":
    """
    Separates the analysis window of wav_path (see pipeline.window) into 4 stems
    using the in-process Demucs model. Returns the stem directory, or None on failure.

    Output layout: {wav_path.parent}/stems/{MODEL_NAME}/{wav_path.stem}/vocals.wav …
    """
    stem_dir = wav_path.parent / "stems" / MODEL_NAME / wav_path.stem

    if all((stem_dir / f"{s}.wav").exists() for s in STEMS):
        logger.info("separator: stems cached at %s — skipping", stem_dir)
        return stem_dir

    if _demucs_model is None:
        logger.warning("separator: model not loaded — using HPSS fallback")
        return None

    try:
        import torch
        import soundfile as sf
        import librosa

        model = _demucs_model
        target_sr = model.samplerate  # mdx_extra expects 44100 Hz

        # Pick analysis window (see pipeline.window for tier logic).  For long
        # tracks we run the same SSM scan the analyzer uses, so the stems land
        # in the SSM chorus window — not the fixed 30 s default.  Both stages
        # are deterministic for the same WAV, so calling find_ssm_window
        # independently here is safe.
        from pipeline.window import pick_window, find_ssm_window
        with sf.SoundFile(str(wav_path)) as snd:
            native_sr = snd.samplerate
            total_sec = snd.frames / native_sr
            chorus_start = find_ssm_window(wav_path, total_sec) if total_sec >= 75.0 else None
            offset_frames, frames = pick_window(snd.frames, native_sr, chorus_start)
            snd.seek(offset_frames)
            data = snd.read(frames=frames, dtype="float32", always_2d=True)

        # data: [frames, channels] → [channels, frames]
        wav_np = data.T
        if wav_np.shape[0] == 1:
            wav_np = np.repeat(wav_np, 2, axis=0)
        elif wav_np.shape[0] > 2:
            wav_np = wav_np[:2]

        # Resample to model's required sample rate
        if native_sr != target_sr:
            wav_np = np.stack([
                librosa.resample(ch, orig_sr=native_sr, target_sr=target_sr)
                for ch in wav_np
            ])

        wav_tensor = torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0)  # [1, 2, T]
        wav_tensor = wav_tensor.to(_demucs_device)

        logger.info(
            "separator: separating %.1fs clip starting at %.1fs with Demucs %s (device=%s)",
            frames / native_sr, offset_frames / native_sr, MODEL_NAME, _demucs_device,
        )
        from demucs.apply import apply_model
        with torch.no_grad():
            sources = apply_model(model, wav_tensor, segment=10.0, overlap=0.1)

        sources = sources[0]  # [n_sources, 2, T]

        stem_dir.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(model.sources):
            stem_np = sources[i].cpu().numpy().T  # [samples, channels]
            sf.write(str(stem_dir / f"{name}.wav"), stem_np, target_sr, subtype="PCM_16")

        logger.info("separator: stems written to %s", stem_dir)
        return stem_dir

    except Exception as exc:
        logger.warning("separator: failed (%s) — analyzer will use HPSS fallback", exc)
        return None
