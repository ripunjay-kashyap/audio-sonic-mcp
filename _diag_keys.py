"""Diagnostic: mirror analyze_audio exactly and show full key evidence for any song."""
import logging
import sys
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(message)s")

import librosa
import soundfile as sf
from pipeline.window import find_ssm_window, pick_window
from pipeline.analyzer import (
    PITCH_CLASSES,
    _load_stem,
    _load_hpss_harmonic,
    _normalized_chroma_mean,
    _score_keys_for_chroma,
    detect_key_sections,
)

JOBS_ROOT = Path("jobs")
TARGET_SR  = 22050
WIN_SEC    = 30.0

# Accepts slug list from CLI: python _diag_keys.py sig_not_like_us sig_attention
# Falls back to the three historically-failing songs when no args given.
if len(sys.argv) > 1:
    SONGS = [(slug, "?") for slug in sys.argv[1:]]
else:
    SONGS = [
        ("sig_not_like_us",   "F Minor"),
        ("sig_shoulda_never", "E Minor"),
        ("sig_attention",     "D# Minor"),
    ]

for slug, true_key in SONGS:
    print(f"\n{'='*60}")
    print(f"  {slug}  (true key: {true_key})")
    print(f"{'='*60}")

    wav_path  = JOBS_ROOT / slug / "input.wav"
    stems_dir = JOBS_ROOT / slug / "stems" / "mdx_extra" / "input"

    if not wav_path.exists():
        print(f"  [SKIP] no input.wav found at {wav_path}")
        continue

    # ── SSM window (mirrors analyze_audio) ───────────────────────────────
    with sf.SoundFile(str(wav_path)) as snd:
        sr_orig   = snd.samplerate
        total_sec = snd.frames / sr_orig

    chorus_start = find_ssm_window(wav_path, total_sec) if total_sec >= 75.0 else None
    stem_offset  = chorus_start if chorus_start is not None else 0.0

    with sf.SoundFile(str(wav_path)) as snd:
        sr_orig = snd.samplerate
        off_frames, n_frames = pick_window(snd.frames, sr_orig, chorus_start)
        snd.seek(off_frames)
        raw = snd.read(frames=n_frames, dtype="float32", always_2d=True)
    y_mono = librosa.to_mono(raw.T)
    if sr_orig != TARGET_SR:
        y_mono = librosa.resample(y_mono, orig_sr=sr_orig, target_sr=TARGET_SR)
    offset_sec = off_frames / sr_orig
    print(f"  SSM chorus start: {chorus_start if chorus_start is not None else 'n/a'}"
          f"  pick_window offset: {offset_sec:.1f}s  duration: {n_frames/sr_orig:.1f}s")

    # ── stems ─────────────────────────────────────────────────────────────
    y_bass  = _load_stem(stems_dir, "bass",  TARGET_SR, 60.0, stem_offset)
    y_other = _load_stem(stems_dir, "other", TARGET_SR, 60.0, stem_offset)
    y_harm  = (y_bass + y_other) if (y_bass is not None and y_other is not None) else None

    # ── HPSS sections — match analyze_audio exactly ───────────────────────
    harmonics = [y_harm] if y_harm is not None else []
    if chorus_start is None:
        for sec in (90, 150):
            h = _load_hpss_harmonic(wav_path, sec, 60, TARGET_SR)
            if h is not None:
                harmonics.append(h)
    else:
        verse_offset = max(30.0, chorus_start - 45.0)
        h = _load_hpss_harmonic(wav_path, verse_offset, 60, TARGET_SR)
        if h is not None:
            harmonics.append(h)
    print(f"  harmonic sections: {len(harmonics)}"
          + (f"  (verse HPSS at {verse_offset:.0f}s)" if chorus_start is not None else ""))

    # ── accumulate key scores ─────────────────────────────────────────────
    win_samples = int(WIN_SEC * TARGET_SR)
    all_scores: dict[str, float] = {}
    for yh in harmonics:
        n = yh.size
        if n >= int(45 * TARGET_SR):
            starts = [0, (n - win_samples) // 2, n - win_samples]
            chromas = [_normalized_chroma_mean(yh[s:s + win_samples], TARGET_SR) for s in starts]
            scores  = {k: sum(ws[k] for ws in [_score_keys_for_chroma(c) for c in chromas]) / 3
                       for k in _score_keys_for_chroma(chromas[0])}
        else:
            scores = _score_keys_for_chroma(_normalized_chroma_mean(yh, TARGET_SR))
        for k, v in scores.items():
            all_scores[k] = all_scores.get(k, 0.0) + v

    ranked = sorted(all_scores.items(), key=lambda kv: kv[1], reverse=True)
    top8   = ranked[:8]
    true_rank   = next((i for i, (k, _) in enumerate(ranked) if k == true_key), -1)
    winner, winner_score = ranked[0]
    sorted_scores = [s for _, s in ranked]
    gap_top2 = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else 0.0

    print(f"\n  Top 8 key candidates:  (rank-1/rank-2 gap={gap_top2:.4f})")
    for rank, (k, s) in enumerate(top8):
        marker = " << detected" if rank == 0 else (" << TRUE" if k == true_key else "")
        print(f"    {rank+1:2d}. {k:<15}  score={s:.4f}{marker}")
    true_score = all_scores.get(true_key, 0.0)
    if true_key != "?":
        rank_str = str(true_rank + 1) if true_rank >= 0 else ">8"
        print(f"\n  True key rank: {rank_str}  score={true_score:.4f}  "
              f"winner={winner_score:.4f}  gap={winner_score - true_score:.4f}")

    # ── bass root analysis ────────────────────────────────────────────────
    if y_bass is not None:
        bass_chroma  = librosa.feature.chroma_cens(y=y_bass, sr=TARGET_SR, hop_length=2048).mean(axis=1)
        sorted_pcs   = np.sort(bass_chroma)[::-1]
        concentration = float(sorted_pcs[0] / (sorted_pcs[1] + 1e-6))
        bass_root    = PITCH_CLASSES[int(np.argmax(bass_chroma))]
        print(f"\n  Bass root: {bass_root}  concentration: {concentration:.2f}  (threshold 1.15)")
        top3 = sorted(zip(PITCH_CLASSES, bass_chroma), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Bass top-3: {[(p, f'{v:.3f}') for p, v in top3]}")

    # ── mean chroma ───────────────────────────────────────────────────────
    if harmonics:
        avg_chroma = np.mean([_normalized_chroma_mean(h, TARGET_SR) for h in harmonics], axis=0)
        print(f"\n  Mean chroma (z-scored):")
        for pc, val in sorted(zip(PITCH_CLASSES, avg_chroma), key=lambda x: x[1], reverse=True):
            bar = "#" * int(max(0, val + 3) * 4)
            print(f"    {pc:3s}  {val:+.3f}  {bar}")

    # ── progressive track detection ───────────────────────────────────────
    if total_sec >= 90.0:
        key_variable, key_map = detect_key_sections(wav_path, total_sec)
        print(f"\n  Progressive mode: key_variable={key_variable}")
        if key_map:
            for seg in key_map:
                print(f"    {seg['start_sec']:.0f}s -> {seg['end_sec']:.0f}s:  {seg['key']}")
        else:
            print("    (single key throughout)")
