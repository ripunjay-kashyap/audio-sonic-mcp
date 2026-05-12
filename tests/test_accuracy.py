"""
End-to-end accuracy regression tests against real-song stems with known
ground-truth BPM and key (sourced from Songstats).

These tests load Demucs stems from ``jobs/<slug>/stems/mdx_extra/input/`` and
exercise ``_extract_bpm`` and ``_detect_key`` directly — no network calls,
no Demucs run, ~1 second per song.

If stems are missing the test is skipped with a hint on how to populate.
To populate fixtures, run with ``KEEP_JOB_FILES=1`` and the slug as job_id::

    KEEP_JOB_FILES=1 .venv/Scripts/python.exe smoke_test.py <URL> <slug>

Known-failing cases are marked ``xfail`` so a future fix surfaces as
``XPASS`` rather than silently flipping behavior elsewhere.
"""

from pathlib import Path

import pytest

from pipeline.analyzer import _detect_key, _extract_bpm, _load_stem


SR = 22050
DURATION = 60.0
JOBS_ROOT = Path(__file__).parent.parent / "jobs"
BPM_TOLERANCE_PCT = 5.0  # within 5% of ground-truth BPM counts as correct


# (slug, url, true_bpm, true_key, bpm_xfail_reason, key_xfail_reason)
GROUND_TRUTH = [
    (
        "sig_yukon",
        "https://www.youtube.com/watch?v=fXivMSJm_kA",
        96, "G Minor", None, None,
    ),
    (
        "sig_humble",
        "https://www.youtube.com/watch?v=tvTRZJ-4EyI",
        150, "C# Minor",
        None,
        "chromagram shows stronger F (major 3rd of C#) than E (minor 3rd) across all "
        "windows — first 60s emphasizes a major-mode tonality despite the song's "
        "overall C# Minor label",
    ),
    (
        "sig_somebody",
        "https://www.youtube.com/watch?v=jLQrk6rmX6w",
        76, "D Major",
        None,
        "chromagram shows stronger F (minor 3rd of D) than F# (major 3rd) across all "
        "windows — verse/intro modal content reads as D Minor even when overall song "
        "is labeled D Major",
    ),
    (
        "sig_exo_tempo",
        "https://www.youtube.com/watch?v=iwd8N6K-sLk",
        116, "C Major",
        None,
        "complex modulating harmony; analyzed 30-90s window is D-rooted; chorus "
        "where C Major emerges is not captured by single 60s segment",
    ),
    (
        "sig_lauv_julia",
        "https://www.youtube.com/watch?v=0PTU4kGj5JI",
        113, "B Major",
        "true tempo 113 BPM not present in librosa tempogram for any start_bpm seed; "
        "kick pattern peaks at 83/152/161 BPM (4/3 and 7/5 ratios of true) but never "
        "at 113 itself — fundamental beat-tracker limitation for this song's rhythm",
        None,
    ),
    (
        "sig_ballad",
        "https://www.youtube.com/watch?v=kON9fn01rUQ",
        73, "C Major", None, None,
    ),
    (
        "sig_hiphop",
        "https://www.youtube.com/watch?v=r_0JjYUe5jo",
        166, "A# Minor",
        None,
        "mode flip — chromagram correctly identifies A# root but shows stronger "
        "D (major 3rd) than C# (minor 3rd) across all windows",
    ),
    (
        "sig_bieber_eta",
        "https://www.youtube.com/watch?v=TiK7QtIHy_Y",
        127, "E Minor", None, None,
    ),
    (
        "sig_jcole_mc",
        "https://www.youtube.com/watch?v=WILNIXZr2oc",
        124, "G# Major", None, None,
    ),
    (
        "sig_test_dna",
        "https://www.youtube.com/watch?v=NLZRYQMLDW4",
        140, "A# Minor",
        "drum seeds split 70/144 with bass at 92 (triplet relation to both); no "
        "clean rule distinguishes this from sig_somebody where lower drum is correct",
        "chroma picks B Minor (one semitone off A# Minor); harmonic content of "
        "first 60s is ambiguous between adjacent keys",
    ),
    (
        "sig_test_hello",
        "https://www.youtube.com/watch?v=YQHsXMglC9A",
        79, "F Minor",
        "true 79 BPM not in tempogram top peaks; closest derivable is bass=83.4 "
        "(5.5% error, just outside tolerance) — librosa beat tracker limitation",
        None,
    ),
    (
        "sig_test_levitating",
        "https://www.youtube.com/watch?v=TUVcZfQe-Kw",
        103, "F# Minor",
        None,
        "chroma picks B Major; bass-root override gated correctly but underlying "
        "chromagram favors B's pattern over F#'s",
    ),
]


def _stems_path(slug: str) -> Path:
    return JOBS_ROOT / slug / "stems" / "mdx_extra" / "input"


def _load_song(slug: str):
    """Load drums + bass + (bass+other harmonic mix) from cached stems."""
    stems = _stems_path(slug)
    if not stems.exists():
        pytest.skip(
            f"No cached stems for {slug} at {stems}. To populate, run:\n"
            f"  KEEP_JOB_FILES=1 .venv/Scripts/python.exe smoke_test.py "
            f"<url> {slug}"
        )
    y_drums = _load_stem(stems, "drums", SR, DURATION)
    y_bass = _load_stem(stems, "bass", SR, DURATION)
    y_other = _load_stem(stems, "other", SR, DURATION)
    return y_drums, y_bass, y_bass + y_other


@pytest.mark.parametrize(
    "slug,url,true_bpm,true_key,bpm_xfail_reason,key_xfail_reason",
    GROUND_TRUTH,
    ids=[g[0] for g in GROUND_TRUTH],
)
def test_bpm_within_tolerance(request, slug, url, true_bpm, true_key, bpm_xfail_reason, key_xfail_reason):
    if bpm_xfail_reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=bpm_xfail_reason, strict=False))
    y_drums, y_bass, _ = _load_song(slug)
    bpm = _extract_bpm(y_drums, SR, y_bass=y_bass)
    error_pct = abs(bpm - true_bpm) / true_bpm * 100
    assert error_pct <= BPM_TOLERANCE_PCT, (
        f"{slug}: detected BPM {bpm:.2f} differs from ground truth {true_bpm} "
        f"by {error_pct:.1f}% (tolerance {BPM_TOLERANCE_PCT}%)"
    )


@pytest.mark.parametrize(
    "slug,url,true_bpm,true_key,bpm_xfail_reason,key_xfail_reason",
    GROUND_TRUTH,
    ids=[g[0] for g in GROUND_TRUTH],
)
def test_key_matches_ground_truth(request, slug, url, true_bpm, true_key, bpm_xfail_reason, key_xfail_reason):
    if key_xfail_reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=key_xfail_reason, strict=False))
    _, y_bass, y_harm = _load_song(slug)
    key, conf = _detect_key(y_harm, SR, y_bass=y_bass)
    assert key == true_key, (
        f"{slug}: detected key '{key}' (conf={conf:.2f}) "
        f"does not match ground truth '{true_key}'"
    )
