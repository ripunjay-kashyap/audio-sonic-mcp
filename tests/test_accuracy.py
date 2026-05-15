"""
End-to-end accuracy regression tests.

Each test calls ``analyze_audio()`` directly against cached job artefacts
(input.wav + Demucs stems) and validates the full result dict.  The four
test types per song share a single ``analyze_audio()`` call via
``_ANALYSIS_CACHE`` so the suite stays fast after cache population.

Populating the cache (one-time, ~3-4 min per song):

    $env:KEEP_JOB_FILES="1"
    .venv\\Scripts\\python.exe smoke_test.py "<youtube_url>" sig_<slug>

After all songs are cached:

    pytest tests/test_accuracy.py -v -s

Ground-truth BPM and key values are sourced from Songstats (songstats.com).
5% BPM tolerance; exact-string key match.
"""

import time
from pathlib import Path

import pytest

from pipeline.analyzer import analyze_audio

JOBS_ROOT = Path(__file__).parent.parent / "jobs"
BPM_TOLERANCE_PCT = 5.0
ANALYSIS_TIME_MAX_SEC = 35.0  # SSM scan adds ~8 s on top of multi-section HPSS; observed max ~25s

# ---------------------------------------------------------------------------
# Ground truth
# (slug, url, true_bpm, true_key, bpm_xfail_reason, key_xfail_reason)
# ---------------------------------------------------------------------------
GROUND_TRUTH = [
    # Ground truth sourced from Songstats.  Flat keys converted to sharps to
    # match the pipeline's PITCH_CLASSES notation (Gb→F#, Ab→G#, Eb→D#).
    (
        "sig_million_dollar",
        "https://www.youtube.com/watch?v=Zf1d8SGuxfs",
        138, "F# Minor",   # Tommy Richman – MILLION DOLLAR BABY (Gb Minor)
        None, None,
    ),
    (
        "sig_not_like_us",
        "https://www.youtube.com/watch?v=H58vbez_m4E",
        101, "F Minor",    # Kendrick Lamar – Not Like Us; first ~30s intro is a different key; main body is F Minor
        None, "SSM window lands in a C-tonic section; detects C Minor (v of F Minor) — tonic centre off by a 4th",
    ),
    (
        "sig_get_it_sexyy",
        "https://www.youtube.com/watch?v=ru64eEvd6Ak",
        145, "F Minor",    # Sexyy Red – Get It Sexyy
        None, None,
    ),
    (
        "sig_shoulda_never",
        "https://www.youtube.com/watch?v=crWbG90dChw",
        137, "E Minor",    # Kehlani ft. Usher – Shoulda Never
        None, None,        # fixed: absent-major-3rd override (A Major C# = -0.86)
    ),
    (
        "sig_attention",
        "https://www.youtube.com/watch?v=nfs8NYg7yQM",
        100, "D# Minor",   # Charlie Puth – Attention (Eb Minor)
        None, None,        # fixed: 5th-alias distinguishing-tone check (B > C in chroma)
    ),
    (
        "sig_take_five",
        "https://www.youtube.com/watch?v=vmDDOFXSgAs",
        174, "D# Minor",   # Dave Brubeck – Take Five (Eb Minor); 5/4 time
        None,
        "modal jazz; Eb Dorian inflection may not surface as D# Minor via Krumhansl",
    ),
    (
        "sig_so_what",
        "https://www.youtube.com/watch?v=zqNTltOGh5c",
        136, "D Minor",    # Miles Davis – So What; D Dorian (closest diatonic = D Minor); Tunebat/human: 136-137 BPM
        "free-floating rubato intro has no strict BPM; tracker locks onto a different pulse than the 136 BPM main body",
        None,              # fixed: 5th-alias guard requires alias_dist genuinely present (z > 0), not just less-absent
    ),
    (
        "sig_dna",
        "https://www.youtube.com/watch?v=NLZRYQMLDW4",
        140, "B Minor",    # Kendrick Lamar – DNA.; original key B Phrygian → closest standard key is B Minor (only the 2nd degree differs)
        None,              # fixed: 5-window scan gives 3× 140 votes vs 2× 130 → majority overrides SSM primary
        "B Phrygian is modal; SSM window lands in Part 2 → detected D# Minor, not B Minor",
    ),
    (
        "sig_fein",
        "https://www.youtube.com/watch?v=U-l4ya3ejko",
        148, "D# Minor",   # Travis Scott – FE!N (Eb Minor); ground truth corrected from ChatGPT
        None, None,
    ),
    (
        "sig_power",
        "https://www.youtube.com/watch?v=L53gjP-TtGE",
        154, "C Minor",    # Kanye West – POWER; Tunebat/human verified 154 BPM; pipeline detects 152 = 1.3% ✓
        None,
        None,              # key correct: C Minor ✓
    ),
    (
        "sig_new_magic_wand",
        "https://www.youtube.com/watch?v=2w8KUgIkAu8",
        140, "F Minor",    # Tyler, The Creator – NEW MAGIC WAND; industrial distortion
        None,              # BPM correct: 136 within 5% of 140
        None,              # fixed: parallel-key tie-breaker — both 3rds absent, decide by 6th (min 6th present)
    ),
]


# ---------------------------------------------------------------------------
# Cache — analyze_audio() runs once per slug per session
# ---------------------------------------------------------------------------
_ANALYSIS_CACHE: dict = {}


def _get_or_run(slug: str) -> dict:
    """Return cached {result, elapsed} for slug, running analyze_audio() if needed."""
    if slug in _ANALYSIS_CACHE:
        return _ANALYSIS_CACHE[slug]

    wav_path  = JOBS_ROOT / slug / "input.wav"
    stems_dir = JOBS_ROOT / slug / "stems" / "mdx_extra" / "input"

    if not wav_path.exists():
        pytest.skip(
            f"No cached input.wav for {slug!r}. Populate with:\n"
            f"  $env:KEEP_JOB_FILES='1'\n"
            f"  .venv\\Scripts\\python.exe smoke_test.py <url> {slug}"
        )

    t0 = time.perf_counter()
    result = analyze_audio(wav_path, stems_dir if stems_dir.exists() else None)
    elapsed = time.perf_counter() - t0

    _ANALYSIS_CACHE[slug] = {"result": result, "elapsed": elapsed}
    return _ANALYSIS_CACHE[slug]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "slug,url,true_bpm,true_key,bpm_xfail,key_xfail",
    GROUND_TRUTH,
    ids=[g[0] for g in GROUND_TRUTH],
)
def test_bpm_within_tolerance(
    request, slug, url, true_bpm, true_key, bpm_xfail, key_xfail
):
    if bpm_xfail:
        request.node.add_marker(pytest.mark.xfail(reason=bpm_xfail, strict=False))

    data   = _get_or_run(slug)
    bpm    = data["result"]["bpm"]
    err    = abs(bpm - true_bpm) / true_bpm * 100
    print(f"\n  {slug}: BPM={bpm:.1f}  GT={true_bpm}  err={err:.1f}%  t={data['elapsed']:.2f}s")
    assert err <= BPM_TOLERANCE_PCT, (
        f"{slug}: detected BPM {bpm:.2f} differs from ground truth {true_bpm} "
        f"by {err:.1f}% (tolerance {BPM_TOLERANCE_PCT}%)"
    )


@pytest.mark.parametrize(
    "slug,url,true_bpm,true_key,bpm_xfail,key_xfail",
    GROUND_TRUTH,
    ids=[g[0] for g in GROUND_TRUTH],
)
def test_key_matches_ground_truth(
    request, slug, url, true_bpm, true_key, bpm_xfail, key_xfail
):
    if key_xfail:
        request.node.add_marker(pytest.mark.xfail(reason=key_xfail, strict=False))

    data = _get_or_run(slug)
    key  = data["result"]["key"]
    conf = data["result"]["mode_confidence"]
    print(f"\n  {slug}: key={key!r}  GT={true_key!r}  conf={conf:.2f}  t={data['elapsed']:.2f}s")
    assert key == true_key, (
        f"{slug}: detected key '{key}' (conf={conf:.2f}) "
        f"does not match ground truth '{true_key}'"
    )


@pytest.mark.parametrize(
    "slug,url,true_bpm,true_key,bpm_xfail,key_xfail",
    GROUND_TRUTH,
    ids=[g[0] for g in GROUND_TRUTH],
)
def test_analysis_timing(slug, url, true_bpm, true_key, bpm_xfail, key_xfail):
    """analyze_audio() on cached stems must finish within ANALYSIS_TIME_MAX_SEC."""
    data    = _get_or_run(slug)
    elapsed = data["elapsed"]
    print(f"\n  {slug}: analysis took {elapsed:.2f}s")
    assert elapsed <= ANALYSIS_TIME_MAX_SEC, (
        f"{slug}: analyze_audio took {elapsed:.2f}s "
        f"(limit {ANALYSIS_TIME_MAX_SEC}s)"
    )


@pytest.mark.parametrize(
    "slug,url,true_bpm,true_key,bpm_xfail,key_xfail",
    GROUND_TRUTH,
    ids=[g[0] for g in GROUND_TRUTH],
)
def test_production_profile_sanity(slug, url, true_bpm, true_key, bpm_xfail, key_xfail):
    """Production profile fields must be in valid ranges and value sets."""
    r = _get_or_run(slug)["result"]
    assert 0.0 <= r["mode_confidence"] <= 1.0
    assert isinstance(r["key_ambiguous"], bool)
    assert isinstance(r["key_variable"], bool)
    assert isinstance(r["key_map"], list)
    assert 0.0 <= r["transient_punch"] <= 1.0
    assert r["stereo_width_label"]    in {"mono", "narrow", "medium", "wide"}
    assert r["vocal_presence_label"]  in {"forward", "present", "background"}
    assert isinstance(r["freq_peaks_hz"]["harmonic"],   list)
    assert isinstance(r["freq_peaks_hz"]["percussive"], list)
