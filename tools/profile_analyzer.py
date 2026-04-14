import time
from pathlib import Path
import librosa
import numpy as np


def to_mono(y):
    return librosa.to_mono(y) if y.ndim > 1 else y


def main():
    stems_dir = Path("stems/sig_970a1a4a/stems/htdemucs")
    stem_files = [f.name for f in stems_dir.glob("*.wav")]

    print("Loading audio...")
    t0 = time.time()
    stems = {}
    for f in stem_files:
        p = stems_dir / f
        stems[f.replace(".wav", "")], sr = librosa.load(str(p), sr=None, mono=False)
    print(f"Loaded in {time.time() - t0:.2f}s, sr={sr}")

    y_drums = to_mono(stems["drums"])

    print("Extracting BPM...")
    t0 = time.time()
    from pipeline.analyzer import _extract_bpm

    _extract_bpm(y_drums, sr)
    print(f"BPM done in {time.time() - t0:.2f}s")

    print("Preparing harmonic array...")
    t0 = time.time()
    harmonic_stems = ["vocals", "other", "bass"]
    # fast array construction
    harmonic_y = np.zeros_like(to_mono(stems["vocals"]))
    for k in harmonic_stems:
        harmonic_y += to_mono(stems[k])
    print(f"Harmonic array prepared in {time.time() - t0:.2f}s")

    print("Detecting key...")
    t0 = time.time()
    from pipeline.analyzer import _detect_key

    _detect_key(harmonic_y, sr)
    print(f"Key done in {time.time() - t0:.2f}s")

    print("Detecting transients...")
    t0 = time.time()
    from pipeline.analyzer import _transient_punch

    _transient_punch(y_drums, sr)
    print(f"Transients done in {time.time() - t0:.2f}s")

    print("Detecting freqs...")
    t0 = time.time()
    from pipeline.analyzer import _dominant_frequencies

    _dominant_frequencies(y_drums, sr)
    print(f"Freqs done in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
