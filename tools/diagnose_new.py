"""Diagnostic for the 3 new test songs."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.analyzer import _load_stem  # noqa: E402

SR = 22050
DUR = 60.0
JOBS = ROOT / "jobs"

SONGS = [
    ("sig_test_dna", 140, "A# Minor"),
    ("sig_test_hello", 79, "F Minor"),
    ("sig_test_levitating", 103, "F# Minor"),
]


def diagnose(slug, true_bpm, true_key):
    import librosa
    stems = JOBS / slug / "stems" / "mdx_extra" / "input"
    y_drums = _load_stem(stems, "drums", SR, DUR)
    y_bass = _load_stem(stems, "bass", SR, DUR)
    print(f"=== {slug}  true: {true_bpm} BPM, {true_key} ===")
    for start in [60, 80, 100, 110, 120, 140, 160, 180]:
        t, _ = librosa.beat.beat_track(y=y_drums, sr=SR, start_bpm=start)
        v = float(t[0]) if hasattr(t, "__len__") else float(t)
        print(f"  d{start:3d}={v:6.1f}")
    tb, _ = librosa.beat.beat_track(y=y_bass, sr=SR, start_bpm=90)
    bv = float(tb[0]) if hasattr(tb, "__len__") else float(tb)
    print(f"  bass={bv:6.1f}")
    onset = librosa.onset.onset_strength(y=y_drums, sr=SR)
    tempogram = librosa.feature.tempogram(onset_envelope=onset, sr=SR)
    tempi = librosa.tempo_frequencies(tempogram.shape[0], sr=SR)
    avg = tempogram.mean(axis=1)
    valid = (tempi > 50) & (tempi < 250)
    top5 = np.argsort(avg[valid])[-5:][::-1]
    vt, vs = tempi[valid], avg[valid]
    print("  top5 tempi: " + ", ".join(f"{vt[i]:.1f}({vs[i]:.2f})" for i in top5))
    print()


for slug, true_bpm, true_key in SONGS:
    diagnose(slug, true_bpm, true_key)
