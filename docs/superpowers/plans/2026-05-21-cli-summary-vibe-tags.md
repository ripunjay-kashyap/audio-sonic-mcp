# CLI Summary View + CLAP Vibe Tags Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add musician-facing `--summary` / `--no-vector` flags to `analyze_file.py` and a CLAP zero-shot `vibe_tags` field (CLI-computed only), without touching the MCP/URL path.

**Architecture:** A pure ranking helper (`_rank_tags`) + a CLAP-loading seam (`_clap_tag_embeddings`) in `vectorizer.py` produce `vibe_tags`; `assemble_payload` gains an additive optional `vibe_tags=None` param; `analyze_file.py` computes the tags, threads them into the payload, and a pure `_print_summary` formatter renders the digest. `vibe_tags` is `None` only when CLAP is unavailable.

**Tech Stack:** Python 3.12, transformers (CLAP), torch, numpy, pytest, argparse, `unittest.mock`/`monkeypatch`.

---

## Current State (as of 2026-05-21)

| File | Status |
|------|--------|
| `analyze_file.py` | Exists; prints full JSON to stdout (`:144-145`); no summary/vector flags |
| `pipeline/vectorizer.py` | Has `generate_vibe_vector` / `_clap_vector` / `_load_audio`; no tags fn |
| `pipeline/assembler.py` | `assemble_payload(...)` builds `sonic_signature` inline; no `vibe_tags` |
| `tests/test_cli.py` | 9 passing tests; no summary/tags coverage |

Spec: `docs/superpowers/specs/2026-05-21-cli-summary-vibe-tags-design.md`

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `pipeline/vectorizer.py` | Modify | Add `VIBE_TAG_PROMPTS`, `VIBE_TAG_TOP_N`, `VIBE_TAG_FLOOR`, `_rank_tags`, `_clap_tag_embeddings`, `generate_vibe_tags` |
| `pipeline/assembler.py` | Modify | Add optional `vibe_tags=None`; emit `sonic_signature.vibe_tags` only when provided |
| `analyze_file.py` | Modify | Import `generate_vibe_tags`; add `--summary`/`--no-vector`; compute tags; `_fmt_mmss`, `_punch_label`, `_print_summary`; branch stdout |
| `README.md` | Modify | Document the two flags + `vibe_tags` |
| `tests/test_cli.py` | Modify | Add ranking, tags, assembler, formatter, and flag tests |

---

### Task 1: Pure tag-ranking helper `_rank_tags`

**Files:**
- Modify: `pipeline/vectorizer.py` (add helper near bottom, before `_load_audio`)
- Test: `tests/test_cli.py` (new `TestRankTags` class)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py`:

```python
import numpy as np


class TestRankTags:
    def test_orders_by_similarity_and_applies_floor(self):
        from pipeline.vectorizer import _rank_tags
        text = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        labels = ["a", "b", "c"]
        audio = np.array([0.3, 0.9, -0.2], dtype=float)  # sims: a=.31 b=.92 c=-.21
        out = _rank_tags(audio, text, labels, top_n=5, floor=0.0)
        assert out == ["b", "a"]  # c dropped by floor 0.0 (negative sim)

    def test_returns_top1_when_all_below_floor(self):
        from pipeline.vectorizer import _rank_tags
        text = np.eye(3)
        labels = ["a", "b", "c"]
        audio = np.array([0.1, 0.2, 0.05], dtype=float)
        out = _rank_tags(audio, text, labels, top_n=5, floor=0.9)
        assert out == ["b"]  # nothing clears 0.9 -> top-1 returned

    def test_respects_top_n_cap(self):
        from pipeline.vectorizer import _rank_tags
        text = np.eye(4)
        labels = ["a", "b", "c", "d"]
        audio = np.array([0.4, 0.3, 0.2, 0.1], dtype=float)
        out = _rank_tags(audio, text, labels, top_n=2, floor=0.0)
        assert out == ["a", "b"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py::TestRankTags -v`
Expected: FAIL — `ImportError: cannot import name '_rank_tags'`

- [ ] **Step 3: Implement `_rank_tags`**

In `pipeline/vectorizer.py`, add immediately above the `# ── Shared utility ──` comment:

```python
# ── Vibe tags (CLAP zero-shot) ──────────────────────────────────────────────


def _rank_tags(
    audio_emb: np.ndarray,
    text_embs: np.ndarray,
    labels: list[str],
    top_n: int,
    floor: float,
) -> list[str]:
    """Rank labels by cosine similarity of their text embeddings to the audio
    embedding. Keep those at/above ``floor``, capped at ``top_n``. If nothing
    clears the floor, return the single highest-ranked label (CLAP still ran)."""
    a = audio_emb / (np.linalg.norm(audio_emb) or 1.0)
    t = text_embs / (np.linalg.norm(text_embs, axis=1, keepdims=True) + 1e-9)
    sims = t @ a  # (N,)
    order = np.argsort(sims)[::-1]
    ranked = [(labels[i], float(sims[i])) for i in order]
    kept = [w for w, s in ranked if s >= floor][:top_n]
    if not kept:
        kept = [ranked[0][0]]
    return kept
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py::TestRankTags -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```powershell
git add pipeline/vectorizer.py tests/test_cli.py
git commit -m "feat(vectorizer): add _rank_tags pure cosine ranking helper"
```

---

### Task 2: `generate_vibe_tags` + CLAP seam + constants

**Files:**
- Modify: `pipeline/vectorizer.py` (constants near top; `_clap_tag_embeddings` + `generate_vibe_tags` in the vibe-tags section)
- Test: `tests/test_cli.py` (new `TestGenerateVibeTags` class)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py`:

```python
class TestGenerateVibeTags:
    def test_returns_none_when_clap_unavailable(self, monkeypatch, audio_wav):
        from pipeline import vectorizer
        monkeypatch.setattr(vectorizer, "_clap_tag_embeddings", lambda *a, **k: None)
        assert vectorizer.generate_vibe_tags(audio_wav, full_song=True) is None

    def test_returns_ranked_words_when_clap_available(self, monkeypatch, audio_wav):
        from pipeline import vectorizer
        labels = [w for w, _ in vectorizer.VIBE_TAG_PROMPTS]
        n = len(labels)
        text = np.eye(n)
        audio = np.zeros(n)
        audio[2] = 1.0   # strongest match = labels[2]
        audio[5] = 0.5
        monkeypatch.setattr(
            vectorizer, "_clap_tag_embeddings", lambda *a, **k: (audio, text)
        )
        out = vectorizer.generate_vibe_tags(audio_wav, full_song=True)
        assert isinstance(out, list) and len(out) >= 1
        assert out[0] == labels[2]
        assert len(out) <= vectorizer.VIBE_TAG_TOP_N
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py::TestGenerateVibeTags -v`
Expected: FAIL — `AttributeError: module 'pipeline.vectorizer' has no attribute 'VIBE_TAG_PROMPTS'` / `generate_vibe_tags`

- [ ] **Step 3: Add constants near the top of `pipeline/vectorizer.py`**

Insert after `VECTOR_DIM = 512` (line 20):

```python
# Vibe-tag vocabulary: (word, prompt-template). Mood/energy/texture use the
# "sounds {}" frame; genres use the "a {} track" frame (better CLAP text signal).
VIBE_TAG_PROMPTS: list[tuple[str, str]] = [
    # Mood
    ("dark", "this music sounds {}"), ("bright", "this music sounds {}"),
    ("melancholic", "this music sounds {}"), ("uplifting", "this music sounds {}"),
    ("aggressive", "this music sounds {}"), ("chill", "this music sounds {}"),
    ("dreamy", "this music sounds {}"), ("tense", "this music sounds {}"),
    ("warm", "this music sounds {}"), ("romantic", "this music sounds {}"),
    ("melodic", "this music sounds {}"),
    # Energy
    ("energetic", "this music sounds {}"), ("mellow", "this music sounds {}"),
    ("driving", "this music sounds {}"), ("laid-back", "this music sounds {}"),
    ("hard-hitting", "this music sounds {}"), ("smooth", "this music sounds {}"),
    # Genre
    ("hip-hop", "a {} track"), ("trap", "a {} track"), ("lo-fi", "a {} track"),
    ("jazz", "a {} track"), ("rock", "a {} track"), ("electronic", "a {} track"),
    ("ambient", "a {} track"), ("soul", "a {} track"), ("R&B", "a {} track"),
    ("pop", "a {} track"), ("funk", "a {} track"), ("classical", "a {} track"),
    # Texture
    ("gritty", "this music sounds {}"), ("clean", "this music sounds {}"),
    ("distorted", "this music sounds {}"), ("acoustic", "this music sounds {}"),
    ("synthetic", "this music sounds {}"), ("lush", "this music sounds {}"),
    ("sparse", "this music sounds {}"),
]
VIBE_TAG_TOP_N = 5
VIBE_TAG_FLOOR = 0.0  # cosine floor; calibrate upward after observing real runs
```

- [ ] **Step 4: Add `_clap_tag_embeddings` and `generate_vibe_tags`**

In the `# ── Vibe tags (CLAP zero-shot) ──` section (above `_rank_tags`), add:

```python
def generate_vibe_tags(wav_path: Path, full_song: bool = False) -> "list[str] | None":
    """Top vibe/mood/genre words via CLAP zero-shot. Returns None only when
    CLAP is unavailable (the librosa fallback vector is not in CLAP space)."""
    embs = _clap_tag_embeddings(wav_path, full_song=full_song)
    if embs is None:
        logger.info("vibe_tags: CLAP unavailable — returning None")
        return None
    audio_emb, text_embs = embs
    labels = [w for w, _ in VIBE_TAG_PROMPTS]
    tags = _rank_tags(audio_emb, text_embs, labels, VIBE_TAG_TOP_N, VIBE_TAG_FLOOR)
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
        prompts = [tmpl.format(w) for w, tmpl in VIBE_TAG_PROMPTS]

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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py::TestGenerateVibeTags -v`
Expected: 2 PASSED

- [ ] **Step 6: Commit**

```powershell
git add pipeline/vectorizer.py tests/test_cli.py
git commit -m "feat(vectorizer): add generate_vibe_tags CLAP zero-shot tagging"
```

---

### Task 3: Assembler `vibe_tags` optional field

**Files:**
- Modify: `pipeline/assembler.py:9-63`
- Test: `tests/test_cli.py` (new `TestAssembleVibeTags` class)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py`:

```python
class TestAssembleVibeTags:
    def _features(self):
        return {
            "bpm": 120.0, "key": "C Major", "mode_confidence": 0.7,
            "vocal_presence_label": "present", "transient_punch": 0.4,
            "stereo_width_label": "wide", "freq_peaks_hz": {"harmonic": [100.0]},
        }

    def test_includes_vibe_tags_when_provided(self):
        from pipeline.assembler import assemble_payload
        p = assemble_payload("j", self._features(), [0.0] * 512, 1.0, [],
                             {"title": "t"}, vibe_tags=["dark", "jazz"])
        assert p["sonic_signature"]["vibe_tags"] == ["dark", "jazz"]

    def test_omits_vibe_tags_when_not_provided(self):
        from pipeline.assembler import assemble_payload
        p = assemble_payload("j", self._features(), [0.0] * 512, 1.0, [],
                             {"title": "t"})
        assert "vibe_tags" not in p["sonic_signature"]
        assert p["sonic_signature"]["bpm"] == 120.0  # rest of payload intact
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py::TestAssembleVibeTags -v`
Expected: FAIL — `test_includes...` raises `TypeError: assemble_payload() got an unexpected keyword argument 'vibe_tags'`

- [ ] **Step 3: Add the `vibe_tags` param**

In `pipeline/assembler.py`, change the signature (line 9-16) to add the param after `source_info`:

```python
def assemble_payload(
    job_id: str,
    features: dict[str, Any],
    vibe_vector: list[float],
    inference_time: float,
    cpu_samples: list[float],
    source_info: dict,
    vibe_tags: "list[str] | None" = None,
) -> dict:
```

Then replace the `return { ... }` block (lines 35-63) with a version that builds `sonic_signature` first, conditionally adds `vibe_tags`, and returns:

```python
    sonic_signature = {
        "bpm": features["bpm"],
        "bpm_variable": features.get("bpm_variable", False),
        "bpm_range": features.get("bpm_range"),
        "key": features["key"],
        "mode_confidence": features.get("mode_confidence"),
        "key_ambiguous": features.get("key_ambiguous", False),
        "key_variable": features.get("key_variable", False),
        "key_map": features.get("key_map", []),
        "vibe_vector": vibe_vector,
        "production_profile": {
            "vocal_presence": features["vocal_presence_label"],
            "transient_punch": features["transient_punch"],
            "stereo_width": features["stereo_width_label"],
            "dominant_freq_peaks_hz": features.get("freq_peaks_hz", {}),
        },
    }
    if vibe_tags is not None:
        sonic_signature["vibe_tags"] = vibe_tags

    return {
        "header": {
            "job_id": job_id,
            "status": "success",
            "confidence_score": confidence,
            "source_metadata": source_metadata,
        },
        "sonic_signature": sonic_signature,
        "telemetry": {
            "cpu_usage_avg": f"{cpu_avg:.0f}%" if cpu_samples else "n/a",
            "inference_time_sec": round(inference_time, 2),
        },
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py::TestAssembleVibeTags tests/test_assembler.py -v`
Expected: new tests PASS; existing `test_assembler.py` still PASS (MCP-safety regression)

- [ ] **Step 5: Commit**

```powershell
git add pipeline/assembler.py tests/test_cli.py
git commit -m "feat(assembler): add optional vibe_tags field (omitted when None)"
```

---

### Task 4: Summary formatter in `analyze_file.py`

**Files:**
- Modify: `analyze_file.py` (add `_fmt_mmss`, `_punch_label`, `_print_summary` above `main()`)
- Test: `tests/test_cli.py` (new `TestPrintSummary` class)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py`:

```python
class TestPrintSummary:
    def _payload(self, vibe_tags=["aggressive", "dark", "hip-hop"]):
        return {
            "header": {
                "confidence_score": 0.78,
                "source_metadata": {"title": "input", "duration_sec": 102.34},
            },
            "sonic_signature": {
                "bpm": 153.85, "bpm_variable": False, "bpm_range": None,
                "key": "G Major", "mode_confidence": 0.63, "key_variable": True,
                "key_map": [
                    {"start_sec": 0.0, "end_sec": 30.0, "key": "G Major"},
                    {"start_sec": 30.0, "end_sec": 90.0, "key": "G Phrygian"},
                ],
                "vibe_vector": [0.0] * 512,
                "vibe_tags": vibe_tags,
                "production_profile": {
                    "vocal_presence": "forward", "transient_punch": 0.325,
                    "stereo_width": "narrow",
                    "dominant_freq_peaks_hz": {"harmonic": [49.7, 49.4], "percussive": []},
                },
            },
            "telemetry": {"inference_time_sec": 322.81},
        }

    def test_renders_core_fields(self, capsys):
        from analyze_file import _print_summary
        _print_summary(self._payload())
        out = capsys.readouterr().out
        assert "SONIC SIGNATURE" in out
        assert "153" in out and "BPM" in out
        assert "G Major" in out and "shifts to G Phrygian" in out
        assert "aggressive · dark · hip-hop" in out
        assert "forward" in out
        assert "78%" in out
        assert "0.0," not in out  # the 512 array is NOT printed

    def test_unavailable_vibe_when_tags_none(self, capsys):
        from analyze_file import _print_summary
        _print_summary(self._payload(vibe_tags=None))
        out = capsys.readouterr().out
        assert "unavailable" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py::TestPrintSummary -v`
Expected: FAIL — `ImportError: cannot import name '_print_summary' from 'analyze_file'`

- [ ] **Step 3: Implement the formatter**

In `analyze_file.py`, add these three functions immediately above `def main():` (line 67):

```python
def _fmt_mmss(seconds: float) -> str:
    s = int(round(seconds or 0))
    return f"{s // 60}:{s % 60:02d}"


def _punch_label(v: float) -> str:
    if v < 0.33:
        return "low"
    if v < 0.66:
        return "moderate"
    return "high"


def _print_summary(payload: dict) -> None:
    """Print a compact, musician-friendly digest (no JSON, no 512-float vector)."""
    sm = payload["header"]["source_metadata"]
    ss = payload["sonic_signature"]
    pp = ss["production_profile"]

    title = sm.get("title") or "audio"
    dur = _fmt_mmss(sm.get("duration_sec") or 0)

    if ss.get("bpm_variable") and ss.get("bpm_range"):
        lo, hi = ss["bpm_range"]
        tempo = f"{ss['bpm']:.1f} BPM  (variable {lo:.0f}–{hi:.0f})"
    else:
        tempo = f"{ss['bpm']:.1f} BPM  (steady)"

    base_key = ss.get("key", "Unknown")
    key_line = base_key
    if ss.get("key_variable") and ss.get("key_map"):
        shift = next((seg for seg in ss["key_map"] if seg.get("key") != base_key), None)
        if shift:
            key_line = f"{base_key}  ·  shifts to {shift['key']} @{_fmt_mmss(shift['start_sec'])}"
    conf = ss.get("mode_confidence")
    if conf is not None:
        key_line += f"   (confidence {round(conf * 100)}%)"

    tags = ss.get("vibe_tags")
    vibe = " · ".join(tags) if tags else "(unavailable — CLAP not installed)"

    punch = pp.get("transient_punch")
    punch_str = f"{punch:.2f}  ({_punch_label(punch)})" if punch is not None else "n/a"
    harm = (pp.get("dominant_freq_peaks_hz") or {}).get("harmonic") or []
    low_end = f"~{round(sum(harm) / len(harm))} Hz dominant" if harm else "n/a"

    overall = round((payload["header"].get("confidence_score") or 0) * 100)
    elapsed = _fmt_mmss(payload.get("telemetry", {}).get("inference_time_sec") or 0)

    print(f"\n\U0001F3B5 SONIC SIGNATURE — {title}  ({dur})\n")
    print(f"  TEMPO    {tempo}")
    print(f"  KEY      {key_line}")
    print(f"  VIBE     {vibe}\n")
    print("  PRODUCTION")
    print(f"    Vocals     {pp.get('vocal_presence', 'n/a')}")
    print(f"    Punch      {punch_str}")
    print(f"    Stereo     {pp.get('stereo_width', 'n/a')}")
    print(f"    Low end    {low_end}\n")
    print(f"  Overall confidence: {overall}%   ·   analyzed in {elapsed}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py::TestPrintSummary -v`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```powershell
git add analyze_file.py tests/test_cli.py
git commit -m "feat(cli): add _print_summary musician digest formatter"
```

---

### Task 5: Wire flags + tags into `main()`

**Files:**
- Modify: `analyze_file.py:39` (import), `:71-74` (argparse), `:130` (tags), `:134-141` (assemble call), `:143-152` (output branch)
- Test: `tests/test_cli.py` (new `TestCLIFlags` class)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py`:

```python
class TestCLIFlags:
    def _run(self, wav, tmp_path, *extra, out=None):
        cli = Path(__file__).parent.parent / "analyze_file.py"
        cmd = [sys.executable, str(cli), str(wav), *extra]
        if out:
            cmd += ["--out", str(out)]
        env = os.environ.copy()
        env.pop("KEEP_JOB_FILES", None)
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, env=env)

    def test_summary_prints_digest_not_json(self, synthetic_stereo_wav, tmp_path):
        out_json = tmp_path / "r.json"
        res = self._run(synthetic_stereo_wav, tmp_path, "--summary",
                        "--job-id", "test_cli_sum", out=out_json)
        assert res.returncode == 0, res.stderr
        assert "SONIC SIGNATURE" in res.stdout
        assert "BPM" in res.stdout
        assert '"vibe_vector"' not in res.stdout  # no JSON / no big array
        # --out still has the COMPLETE JSON including the 512-vector
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert len(data["sonic_signature"]["vibe_vector"]) == 512

    def test_no_vector_strips_array_from_json(self, synthetic_stereo_wav, tmp_path):
        res = self._run(synthetic_stereo_wav, tmp_path, "--no-vector",
                        "--job-id", "test_cli_nv")
        assert res.returncode == 0, res.stderr
        parsed = json.loads(res.stdout)
        assert "vibe_vector" not in parsed["sonic_signature"]
        assert "bpm" in parsed["sonic_signature"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py::TestCLIFlags -v`
Expected: FAIL — `--summary`/`--no-vector` are unrecognized args (exit code 2), or JSON still contains `vibe_vector`

- [ ] **Step 3: Update the import (line 39)**

```python
from pipeline.vectorizer import generate_vibe_vector, generate_vibe_tags
```

- [ ] **Step 4: Add the two flags (after line 74, the `--job-id` arg)**

```python
    parser.add_argument("--summary", "-s", action="store_true",
                        help="Print a human-readable digest instead of JSON.")
    parser.add_argument("--no-vector", dest="no_vector", action="store_true",
                        help="Print full JSON but omit the 512-float vibe_vector array.")
```

- [ ] **Step 5: Compute tags (immediately after line 130, the `vibe_vector = ...` line)**

```python
        logger.info("Generating vibe tags (CLAP zero-shot)...")
        vibe_tags = generate_vibe_tags(wav_path, full_song=True)
```

- [ ] **Step 6: Pass `vibe_tags` into `assemble_payload` (the call at lines 134-141)**

Change the call to include the new kwarg:

```python
        payload = assemble_payload(
            job_id=job_id,
            features=features,
            vibe_vector=vibe_vector,
            inference_time=elapsed,
            cpu_samples=[],
            source_info=source_info,
            vibe_tags=vibe_tags,
        )
```

- [ ] **Step 7: Replace the output block (lines 143-152) with flag-aware output**

```python
        # 8. Output: --summary prints the digest; --no-vector strips the array;
        #    default prints full JSON. --out ALWAYS writes the complete payload.
        full_json = json.dumps(payload, indent=2)
        if args.summary:
            _print_summary(payload)
        elif args.no_vector:
            trimmed = {
                **payload,
                "sonic_signature": {
                    k: v for k, v in payload["sonic_signature"].items()
                    if k != "vibe_vector"
                },
            }
            print(json.dumps(trimmed, indent=2))
        else:
            print(full_json)

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(full_json, encoding="utf-8")
            logger.info("Result written to %s", out_path)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py::TestCLIFlags -v`
Expected: 2 PASSED

- [ ] **Step 9: Commit**

```powershell
git add analyze_file.py tests/test_cli.py
git commit -m "feat(cli): wire --summary / --no-vector flags and vibe_tags into main"
```

---

### Task 6: README documentation

**Files:**
- Modify: `README.md` (the `## Local File Analysis (CLI)` arguments table + a vibe-tags note)

- [ ] **Step 1: Add the two flags to the Arguments table**

In the Arguments table under `## Local File Analysis (CLI)`, add two rows after the `--job-id` row:

```markdown
| `--summary` | `-s` | No | Print a human-readable digest (KEY, BPM, vibe tags, production profile) instead of the JSON |
| `--no-vector` | — | No | Print full JSON but omit the 512-float `vibe_vector` array |
```

- [ ] **Step 2: Add a vibe-tags note after the "Accepted file types" line**

```markdown
**Vibe tags:** the JSON includes a `vibe_tags` field — human-readable mood/genre/texture words (e.g. `aggressive · dark · hip-hop`) derived from the CLAP embedding via zero-shot matching. Requires the CLAP extra (`pip install ".[clap]"`); without it the field is `null` and the summary shows `(unavailable)`.
```

- [ ] **Step 3: Add a `--summary` usage example to the Usage code block**

After the `> result.json` example, add:

```powershell
# Human-readable digest only (no JSON) — quick read for musicians
.venv\Scripts\python.exe analyze_file.py "C:\Music\my_demo.mp3" --summary
```

- [ ] **Step 4: Commit**

```powershell
git add README.md
git commit -m "docs: document --summary / --no-vector flags and vibe_tags"
```

---

### Task 7: Full regression + verification

- [ ] **Step 1: Run the CLI test file**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py -v`
Expected: all tests PASS (original 9 + new ranking/tags/assembler/formatter/flag tests)

- [ ] **Step 2: Run the broad suite (excludes slow accuracy fixtures)**

Run: `.venv\Scripts\python.exe -m pytest --ignore=tests/test_accuracy.py -v 2>&1 | Select-Object -Last 15`
Expected: all PASS — in particular `test_server.py` (11) and `test_assembler.py` unchanged (MCP-safety).

- [ ] **Step 3: Live smoke on the real cached song (verifies tags + summary end-to-end)**

Run: `.venv\Scripts\python.exe analyze_file.py "jobs/sig_power/input.wav" --summary --job-id cli_demo_power2`
Expected: the digest prints with a real `VIBE` line (CLAP tags) — or `(unavailable)` if CLAP isn't installed; exit 0.

---

## Self-Review

**Spec coverage:**
- `--summary` / `--no-vector` flags, composable, `--out` always full → Task 5 ✓
- Summary layout + formatting rules → Task 4 ✓
- `generate_vibe_tags` (CLAP zero-shot, vocab, floor, top-N, None-only-when-unavailable) → Tasks 1–2 ✓
- `assemble_payload(vibe_tags=None)` additive, MCP-safe → Task 3 ✓
- `generate_vibe_vector` signature untouched, no `server.py` change → confirmed (no task touches them) ✓
- Testing: pure formatter, mocked-embedding selection, subprocess flag wiring, regression → Tasks 1,2,3,4,5,7 ✓
- README → Task 6 ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; commands concrete with expected output.

**Type consistency:**
- `_rank_tags(audio_emb, text_embs, labels, top_n, floor)` — defined Task 1, called in `generate_vibe_tags` Task 2 ✓
- `_clap_tag_embeddings(wav_path, full_song)` returns `(np.ndarray, np.ndarray) | None` — defined Task 2, monkeypatched in tests Task 2 ✓
- `generate_vibe_tags(wav_path, full_song)` returns `list[str] | None` — Task 2, imported Task 5 ✓
- `assemble_payload(..., vibe_tags=None)` — Task 3, called with `vibe_tags=vibe_tags` Task 5 ✓
- `_print_summary(payload: dict)` — Task 4, called in `main()` Task 5 ✓
- `VIBE_TAG_PROMPTS` / `VIBE_TAG_TOP_N` / `VIBE_TAG_FLOOR` — defined Task 2, referenced in Tasks 2 tests ✓
