"""Diagnostic dump for every cached song: BPM seeds, key scores, bass chroma stats."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.analyzer import (  # noqa: E402
    PITCH_CLASSES, MAJOR_TEMPLATE, MINOR_TEMPLATE, _load_stem,
)

SR = 22050
DUR = 60.0
JOBS = ROOT / "jobs"

SONGS = [
    ("sig_yukon",       96,  "G Minor"),
    ("sig_humble",      150, "C# Minor"),
    ("sig_somebody",    76,  "D Major"),
    ("sig_exo_tempo",   116, "C Major"),
    ("sig_lauv_julia",  113, "B Major"),
    ("sig_ballad",      73,  "C Major"),
    ("sig_hiphop",      166, "A# Minor"),
    ("sig_bieber_eta",  127, "E Minor"),
    ("sig_jcole_mc",    124, "G# Major"),
]


def analyze(slug, true_bpm, true_key):
    import librosa
    stems = JOBS / slug / "stems" / "mdx_extra" / "input"
    if not stems.exists():
        print(f"{slug}: NO STEMS")
        return
    y_drums = _load_stem(stems, "drums", SR, DUR)
    y_bass  = _load_stem(stems, "bass",  SR, DUR)
    y_other = _load_stem(stems, "other", SR, DUR)
    y_harm  = y_bass + y_other

    def track(sig, start):
        t, _ = librosa.beat.beat_track(y=sig, sr=SR, start_bpm=start)
        return float(t[0]) if hasattr(t, "__len__") else float(t)

    t80, t120, t160 = track(y_drums, 80), track(y_drums, 120), track(y_drums, 160)
    tb = track(y_bass, 90)

    # Key analysis
    chroma = librosa.feature.chroma_cens(y=y_harm, sr=SR, hop_length=2048)
    chroma_db = librosa.amplitude_to_db(chroma, ref=np.max)
    combined = chroma_db.mean(axis=1)
    if np.max(np.abs(combined)) > 0:
        combined = (combined - combined.mean()) / (combined.std() + 1e-6)

    scores = {}
    for i, root in enumerate(PITCH_CLASSES):
        rotated = np.roll(combined, -i)
        scores[f"{root} Major"] = float(np.corrcoef(rotated, MAJOR_TEMPLATE)[0, 1])
        scores[f"{root} Minor"] = float(np.corrcoef(rotated, MINOR_TEMPLATE)[0, 1])

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    chroma_winner, chroma_score = ranked[0]
    second_key, second_score = ranked[1]

    # Bass chroma
    bass_chroma = librosa.feature.chroma_cens(y=y_bass, sr=SR, hop_length=2048).mean(axis=1)
    bass_sorted = sorted(enumerate(bass_chroma), key=lambda kv: kv[1], reverse=True)
    bass_root_idx, bass_top = bass_sorted[0]
    bass_2nd_idx, bass_2nd = bass_sorted[1]
    bass_root = PITCH_CLASSES[bass_root_idx]
    bass_2nd_name = PITCH_CLASSES[bass_2nd_idx]
    bass_concentration = bass_top / (bass_2nd + 1e-6)

    # 3rd-interval test on the harmonic chroma at the TRUE key's root
    true_root_str = true_key.split()[0]
    true_root_idx = PITCH_CLASSES.index(true_root_str)
    raw_chroma = librosa.feature.chroma_cens(y=y_harm, sr=SR, hop_length=2048).mean(axis=1)
    maj_3rd = raw_chroma[(true_root_idx + 4) % 12]
    min_3rd = raw_chroma[(true_root_idx + 3) % 12]
    third_says = "Major" if maj_3rd > min_3rd else "Minor"

    # What the override would do
    winner_root = chroma_winner.split()[0]
    override_target = None
    if bass_root != winner_root:
        for mode in ("Minor", "Major"):
            cand = f"{bass_root} {mode}"
            if scores[cand] >= chroma_score * 0.75:
                override_target = cand
                break

    true_score = scores.get(true_key, None)

    print(f"\n=== {slug}  (true: {true_bpm} BPM, {true_key}) ===")
    print(f"  BPM seeds: d80={t80:6.1f}  d120={t120:6.1f}  d160={t160:6.1f}  bass={tb:6.1f}")
    print(f"  Chroma winner: {chroma_winner:10s} ({chroma_score:.3f})  2nd: {second_key:10s} ({second_score:.3f})")
    print(f"  True key score: {true_score:.3f}" if true_score is not None else "  True key not in scores!")
    print(f"  Bass: top={bass_root}({bass_top:.3f}) 2nd={bass_2nd_name}({bass_2nd:.3f}) concentration_ratio={bass_concentration:.2f}")
    print(f"  Override would pick: {override_target}")
    true_mode = true_key.split()[1]
    third_match = "OK" if third_says == true_mode else "WRONG"
    print(f"  3rd-interval test at {true_root_str}: maj3={maj_3rd:.3f} min3={min_3rd:.3f} says {third_says} (true: {true_mode}) [{third_match}]")


for slug, true_bpm, true_key in SONGS:
    try:
        analyze(slug, true_bpm, true_key)
    except Exception as e:
        print(f"{slug}: ERROR {e}")
