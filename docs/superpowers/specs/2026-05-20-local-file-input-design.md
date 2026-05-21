# Local File Analysis CLI — Design Spec

- **Date:** 2026-05-20
- **Status:** Approved (pending written-spec review)
- **Topic:** A standalone command-line script that analyzes a user's local audio file (full-song) and prints the sonic-signature JSON. The MCP server is **not touched**.

## Problem & Motivation

The MCP server analyzes audio from a YouTube URL, fire-and-forget, for **LLM** callers. Musicians want to point the analyzer at **their own tracks or rough demos** (e.g. a phone recording of table-smash percussion + guitar + keys) to get a **sonic signature** — mood (vibe vector), **BPM**, and **key** — to know roughly where to set a session's tempo/key.

That audience is a **human at a terminal**, not an LLM. So local-file analysis ships as a **CLI script** (`analyze_file.py`) that runs synchronously and prints JSON — no MCP tool, no queued/poll, no JOB_STORE, no threading. The musician copy-pastes one command from the README and gets their output.

Two analysis differences from the URL path:
1. **Input is a local file path.**
2. **Demucs + analysis run on the whole song**, not the 60s SSM window — short demos often have no representative "chorus", and full-song analysis avoids the SSM-window key-fragility documented in `CLAUDE.md` (the global tonic dominates).

## Goals

- A `analyze_file.py` CLI that analyzes a local audio file full-song and prints the same sonic-signature JSON as the pipeline produces.
- Reuse the **exact** pipeline stage functions, so analysis is identical to the server's.
- **Never** modify, move, or delete the user's original file.
- Zero new dependencies (reuse FFmpeg/ffprobe; Demucs/CLAP already present).
- A copy-paste README command that works without hand-editing.

## Non-Goals (out of scope)

- **Any change to `server.py` / `get_sonic_signature` / the MCP tool surface.** MCP stays URL-only for LLMs.
- Base64/byte input, a watched "drop folder", or LLM access to local files.
- Embedded-tag (ID3) parsing — filename + ffprobe only.
- A packaged console-entry-point / PyPI install; it's a repo script run via the venv Python.

## Use Cases

- Musician analyzes a finished 3–4 min track for mood/BPM/key.
- Musician analyzes a sparse demo (percussion + guitar + keys): Demucs routes table-smash → `drums` (BPM), guitar/keys → `other` (key); HPSS fallback covers missing stems.

## Configuration

| Name | Default | Notes |
|---|---|---|
| `FILE_MAX_DURATION_SEC` | `600` (10 min) | Env-overridable (e.g. `900` for 15 min). CLI/file path only; the URL path keeps its 60-min limit. |

Rationale: full-song processing is ~**4× audio length on consumer CPU** (1 min ≈ 4 min; 10-min file ≈ 40 min), far less on GPU. 99% of target songs are 3–4 min (~12–16 min CPU).

## Design

### 1. CLI script — `analyze_file.py` (repo root, beside `smoke_test.py`)

```
python analyze_file.py <path> [--out result.json] [--keep] [--job-id ID]
```
- **Flow** (synchronous, in-process):
  1. `validate_file_path(path)` then `source_info = validate_file_source(path)` (duration via ffprobe + 10-min guard).
  2. Print a pre-run notice: file, duration, and `~duration×4 s` CPU estimate ("faster on GPU").
  3. Create `job_dir = JOBS_ROOT / (job_id or f"file_{uuid8}")`; `save_metadata(job_dir, source_info)`.
  4. `wav = convert_to_wav(Path(path), out_dir=job_dir)` — original read in place.
  5. `stems = separate_stems(wav, full_song=True)`.
  6. `features = analyze_audio(wav, stems, full_song=True)`; `vibe = generate_vibe_vector(wav)`.
  7. `payload = assemble_payload(job_id, features, vibe, inference_time, [], source_info)`.
  8. Print JSON to stdout; if `--out`, also write it there. Print elapsed.
  9. `_cleanup_job_artifacts(job_dir)` unless `--keep` (or `KEEP_JOB_FILES`).
- **PATH bootstrap:** prepend `.venv/Scripts` to `PATH` at startup (so the Demucs/yt-dlp CLIs resolve when run via `.venv\Scripts\python.exe` without an activated shell), mirroring `smoke_test.py`. FFmpeg is assumed on PATH per the existing project requirement.
- **Exit codes:** `0` success; `1` on validation/processing error (message to stderr).
- The cleanup/assembly helpers it reuses (`_cleanup_job_artifacts`, `assemble_payload`) are imported from their modules; if `_cleanup_job_artifacts` lives in `server.py`, move it to `pipeline/` (or inline a tiny local copy) to keep the CLI free of any MCP import. **Decision: inline a minimal cleanup in the CLI** to honor "don't touch MCP."

### 2. Ingestion — `pipeline/ingestion.py`

- **`SUPPORTED_AUDIO_EXTS = {mp3, wav, flac, ogg, m4a, aac}`** (the set already in `_is_direct_audio`).
- **`validate_file_path(path)`** — cheap guard: exists, is a regular file, extension in set. Raises `ValueError`.
- **`validate_file_source(path) -> dict`** — mirrors `validate_source`'s return shape; ffprobe duration + `FILE_MAX_DURATION_SEC` guard. Returns:
  ```python
  {"title": <stem>, "uploader": "local file", "duration_sec": <int>,
   "thumbnail": None, "webpage_url": None, "source_path": <abs path>,
   "extractor": "local", "source_type": "file", "genre_hint": None}
  ```
  Raises `RuntimeError` if ffprobe fails / no audio stream; `ValueError` if over the limit.

### 3. Convert — `pipeline/converter.py`

```python
def convert_to_wav(raw_audio_path: Path, out_dir: Path | None = None) -> Path
```
`out_dir` defaults to `raw_audio_path.parent` (**backward-compatible**; URL path unchanged). The CLI passes `out_dir=job_dir`, so the user's file is **read in place** and `input.wav` is written into `job_dir`. The original is never copied or modified.

### 4. Full-song mode — `pipeline/separator.py`, `pipeline/analyzer.py`

- `separate_stems(wav, full_song=False)` — when `True`, skip `find_ssm_window` and separate the entire WAV.
- `analyze_audio(wav, stems_dir, full_song=False)` — when `True`, skip `find_ssm_window` and analyze offset 0 over full length.
- Defaults are `False`, so the server's URL path is byte-for-byte unchanged.
- **Invariant:** stems and the analyzer's `y_mono` must cover the same span. The CLI sets `full_song=True` on both calls; the server sets neither (window mode).

### 5. Output — `pipeline/assembler.py`

`header.source_metadata` gains `source_type` ("file") and `source_path`; `url` is null for file analyses (additive, does not affect URL output).

### 6. Errors (CLI → stderr + exit 1)

| Condition | Type |
|---|---|
| missing file / not a regular file / bad extension | `ValueError` (`validate_file_path`) |
| ffprobe failure / no audio stream | `RuntimeError` (`validate_file_source`) |
| duration > `FILE_MAX_DURATION_SEC` | `ValueError` (`validate_file_source`) |

### 7. README

Add a **"Analyze a local song (CLI)"** section with the copy-paste command and a one-line example, e.g.:
```powershell
.venv\Scripts\python.exe analyze_file.py "C:\Music\my_demo.mp3"
```
Note the ~4× CPU time estimate and that FFmpeg must be on PATH.

## Testing

- **Unit (no network, synthetic WAV via existing helpers):**
  - `validate_file_path`: accepts allowed exts; rejects missing file, directory, bad ext.
  - `validate_file_source`: correct shape + duration; raises over the limit; raises on a non-audio file.
  - `convert_to_wav(path, out_dir=…)`: writes `input.wav` into `out_dir`, leaves source untouched; default `out_dir` unchanged.
- **Full-song path:** `analyze_audio(short_synthetic_wav, stems=None, full_song=True)` returns a valid result via HPSS fallback (no real Demucs).
- **CLI smoke:** run `analyze_file.py` on a short synthetic WAV (HPSS fallback), assert it prints valid JSON with `source_type:"file"` and exits 0; bad path exits 1.
- Existing URL-path + accuracy tests stay green (regression guard on the new default args).

## Risks / Notes

- **Compute time** at the cap (~40 min CPU @ 10 min); mitigated by the cap + the pre-run estimate.
- **Sparse demos** may yield mostly-silent stems; existing audible-block/HPSS handling covers this, confidence may be lower — acceptable for a "rough idea".
- **Full-song BPM** on tempo-varying tracks reports `bpm_variable`/`bpm_range` (already supported) — desirable here.
- **No `server.py` import in the CLI** keeps the "MCP untouched" guarantee literal; the only shared code is the `pipeline/` package (the analysis core, not MCP logic).
