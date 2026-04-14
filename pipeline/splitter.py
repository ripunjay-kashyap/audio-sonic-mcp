"""
Stage 4 — Stem Separation
Runs Facebook's Demucs (htdemucs) to split the WAV into 4 stems:
vocals, drums, bass, other.

Also computes a simple proxy SDR score by measuring silence removed
vs. original energy — a heuristic that avoids needing ground-truth references.
"""

import logging
import subprocess
import sys
from subprocess import DEVNULL
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

STEM_NAMES = ["vocals", "drums", "bass", "other"]


def split_stems(
    wav_path: Path,
    job_id: str,
    stems_root: Path,
    model: str = "htdemucs",
) -> Tuple[Path, list[str], float]:
    """
    Runs Demucs on the given WAV file.

    Returns:
        stems_dir  — directory containing the 4 stem .wav files
        stem_files — list of filenames (e.g. ['vocals.wav', ...])
        sdr        — proxy Signal-to-Distortion ratio (dB)
    """
    output_root = stems_root / job_id / "stems"
    output_root.mkdir(parents=True, exist_ok=True)

    # Use demucs_runner.py wrapper which patches torchaudio.save() to use
    # soundfile instead of torchcodec (avoids FFmpeg shared-library requirement).
    runner = Path(__file__).with_name("demucs_runner.py")

    cmd = [
        sys.executable,
        str(runner),
        f"--name={model}",
        f"--out={output_root}",
        "--filename",
        "{stem}.wav",
        str(wav_path),
    ]

    logger.info("Running Demucs (%s) on %s …", model, wav_path.name)
    logger.info("Output → %s", output_root)

    result = subprocess.run(
        cmd,
        stdin=DEVNULL,
        capture_output=True,
        text=True,
        timeout=1200,  # 20-minute hard cap
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Demucs failed.\nModel: {model}\nstderr: {result.stderr[-3000:]}"
        )

    # ── Locate output stems ───────────────────────────────────────────────────
    # Demucs writes: <out>/<model>/<track_name>/{stem}.wav
    # With --filename flag it should be flat, but let's handle both layouts.
    stems_dir = _find_stems_dir(output_root, model)
    stem_files = _verify_stems(stems_dir)

    # ── Proxy SDR ─────────────────────────────────────────────────────────────
    sdr = _compute_proxy_sdr(wav_path, stems_dir, stem_files)
    logger.info("Stems ready in %s | proxy SDR: %.1f dB", stems_dir, sdr)

    return stems_dir, stem_files, sdr


# ── Helpers ───────────────────────────────────────────────────────────────────


def _find_stems_dir(output_root: Path, model: str) -> Path:
    """Finds wherever Demucs actually dropped the stem files."""
    # Flat layout (--filename flag)
    flat_wavs = list(output_root.glob("*.wav"))
    if len(flat_wavs) >= 4:
        return output_root

    # Nested layout: <out>/<model>/<trackname>/
    for candidate in output_root.rglob("vocals.wav"):
        return candidate.parent

    raise FileNotFoundError(
        f"Could not find stem WAV files under {output_root}. "
        f"Check Demucs output layout."
    )


def _verify_stems(stems_dir: Path) -> list[str]:
    """Confirms all 4 expected stems are present."""
    found = []
    missing = []
    for name in STEM_NAMES:
        f = stems_dir / f"{name}.wav"
        if f.exists():
            found.append(f"{name}.wav")
        else:
            missing.append(f"{name}.wav")

    if missing:
        raise FileNotFoundError(f"Missing expected stems in {stems_dir}: {missing}")

    return found


def _compute_proxy_sdr(
    original_wav: Path,
    stems_dir: Path,
    stem_files: list[str],
) -> float:
    """
    Proxy SDR: ratio of summed-stems energy to residual (difference).
    A clean separation gives ~8–12 dB.
    """
    try:
        import soundfile as sf

        orig, sr = sf.read(str(original_wav), always_2d=True)
        orig = orig.mean(axis=1)  # to mono for comparison

        mixed = np.zeros_like(orig)
        for fname in stem_files:
            stem_path = stems_dir / fname
            data, _ = sf.read(str(stem_path), always_2d=True)
            stem_mono = data.mean(axis=1)
            # Align lengths
            min_len = min(len(orig), len(stem_mono))
            mixed[:min_len] += stem_mono[:min_len]

        min_len = min(len(orig), len(mixed))
        residual = orig[:min_len] - mixed[:min_len]

        signal_power = np.mean(orig[:min_len] ** 2) + 1e-10
        noise_power = np.mean(residual**2) + 1e-10
        sdr = float(10 * np.log10(signal_power / noise_power))
        return round(sdr, 2)

    except Exception as exc:
        logger.warning("SDR calculation failed: %s — using default 8.0", exc)
        return 8.0
