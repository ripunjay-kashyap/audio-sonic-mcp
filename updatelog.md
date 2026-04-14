# Project Update Log

## [2026-04-13] - analyzer.py OOM Fix (Stage 5 Streaming)

### Bug Fix
- **`analyze_stems` peak-memory OOM**: Refactored `pipeline/analyzer.py` to stream stems one at a time instead of loading all 4 into a dict simultaneously. Each full-res stereo array is freed immediately after per-stem features are extracted. BPM and transient punch now computed at 22050 Hz (downsampled). Key detection receives only the two harmonic stems (bass + other) at 22050 Hz mono (~17 MB each) rather than 4 × ~70 MB full-res stereo. Estimated peak RSS reduction: ~900 MB → ~200 MB for a typical 4-minute song.
- All 100 pytest tests still pass.

### Files Modified
- `pipeline/analyzer.py` — `analyze_stems` rewritten with streaming/free-after-use strategy; sub-extractor signatures unchanged (test-compatible).

---

## [2026-04-12] - Sonic Signature Fields, Test Suite Fixes & Smoke Test Investigation

### New Features
- **`mode_confidence`**: `_detect_key()` in `analyzer.py` now returns a `(key_str, confidence)` tuple instead of just the key string. Confidence is the best Pearson correlation score clipped to `[0, 1]`. Exposed as `mode_confidence` in the `analyze_stems()` return dict and wired through `assembler.py` into `sonic_signature.mode_confidence`.
- **`genre_hint`**: `validate_source()` in `ingestion.py` now extracts `categories[0]` from yt-dlp metadata and returns it as `genre_hint` (e.g. `"Music"` for YouTube Music category, `None` if absent). Wired through `assembler.py` into `header.source_metadata.genre_hint`.

### Bug Fixes
- **`test_assembler.py` latent bug**: `test_header_source_fields` was accessing `payload["header"]["source"]` but the assembler produces `"source_metadata"`. Fixed.
- **`test_analyzer.py` broken tests**: `TestDetectKey` tests were calling `_detect_key(y, SR)` with a raw numpy array instead of the expected `dict`. Fixed to pass `{"bass": y, "other": y}` and unpack the new tuple return value.
- **`asyncio.to_thread` fix in `server.py`**: Stages 5 and 6 (`analyze_stems`, `generate_vibe_vector`) switched from `anyio.to_thread.run_sync()` to `asyncio.to_thread()`. The MCP server's anyio task group was silently cancelling the long-running stages — `anyio.to_thread.run_sync` is subject to anyio's cancellation machinery, while `asyncio.to_thread` uses the standard executor and is not.
- **`except BaseException`**: Changed `except Exception` to `except BaseException` in `server.py`'s `split_audio` handler so `asyncio.CancelledError` (a `BaseException`, not `Exception`) is caught and logged.

### CLAUDE.md Updates
- Corrected Testing section: removed "no automated tests" claim; added `pytest` command and `smoke_test.py` usage
- Added `check_health()` to MCP Tools Exposed section (was undocumented)
- Fixed architecture note: `anyio.to_thread.run_sync()` not `asyncio.to_thread()`
- Added **Gotchas** section: Fast Resume behavior, concurrency lock, `.[clap]` install shorthand

### Test Results
- **100/100 pytest tests passing** (up from 96 — 4 new tests added: `mode_confidence` range check, `_detect_key` confidence, `genre_hint` present/absent in assembler)

### Smoke Test — New Test Target
- URL: `https://www.youtube.com/watch?v=gdx7gN1UyX0`
- Expected: **E Minor, ~95 BPM**
- Validated in isolation: Stage 5 (`analyze_stems`) = 79.5s → **95.7 BPM, E Minor, mode_confidence 0.87** ✓
- Validated in isolation: Stage 6 (`generate_vibe_vector`) = 12.2s → **512-dim, L2-norm=1.0** ✓
- Stage 4 (Demucs): ✓ completed — stems cached at `stems/sig_efcc98a0/stems/htdemucs/`, proxy SDR 18.1 dB

### Smoke Test — Outstanding Crash (NEXT SESSION FIX)
**Symptom**: The MCP server process is killed silently during Stage 5 (~80s in), every run. The thread starts (`analyze_stems: thread started` confirmed in logs), but the process dies before completing. No error payload is written, no exception is logged.

**Root cause**: Almost certainly **OOM (out of memory)**. Loading 4 × 70MB stems = 280MB raw audio, plus HPSS/CQT intermediate arrays (~600MB peak), pushes total process RSS to ~1GB. Windows kills the process before it can log an error.

**Why standalone test works**: Fresh Python process starts at ~50MB overhead. MCP server starts at ~80MB + anyio + FastMCP stack, pushing the peak slightly over the OOM threshold.

**Fix to implement next session**: Reduce `analyze_stems` peak memory footprint:
1. Process stems one at a time instead of loading all 4 into a dict simultaneously
2. Downsample BPM source to 22050 Hz (same as key detection) before `beat_track` — avoids holding a 44100 Hz array
3. Free each stem array after its features are extracted

**Files modified this session**:
- `pipeline/analyzer.py` — `mode_confidence`, thread-start log, memory fix (next session)
- `pipeline/ingestion.py` — `genre_hint`
- `pipeline/assembler.py` — `genre_hint` + `mode_confidence` wired through
- `server.py` — `asyncio.to_thread` for Stage 5+6, `except BaseException`, `CancelScope` debug attempts
- `tests/test_analyzer.py` — fixed `_detect_key` tests, added `mode_confidence` tests
- `tests/test_assembler.py` — fixed `"source"` → `"source_metadata"`, added `genre_hint` + `mode_confidence` tests
- `smoke_test.py` — updated default URL to `gdx7gN1UyX0`
- `example_eval.md` — corrected to match real assembler output structure
- `CLAUDE.md` — testing, tools, gotchas sections updated

## [2026-04-09] - Production Hardening, Full Test Suite & End-to-End Verification

### Critical Bug Fixes
- **stdin isolation**: Added `stdin=subprocess.DEVNULL` to all 5 subprocess calls across the pipeline (`ingestion.py`, `downloader.py`, `converter.py`, `splitter.py`, `server.py`). All child processes were inheriting the MCP stdio pipe as stdin and blocking indefinitely.
- **venv PATH injection**: Added `Path(sys.executable).parent` injection into `os.environ["PATH"]` at server startup so yt-dlp, ffmpeg, and demucs are always resolvable by subprocesses even without explicit venv activation.
- **sys.executable**: Changed `"python" -m demucs` to `sys.executable -m demucs` in `splitter.py` to ensure the venv Python (with demucs installed) is used, not the system Python.
- **torchaudio.save() / torchcodec DLL fix**: torchaudio 2.11 requires torchcodec for `save()`, which in turn needs FFmpeg shared DLLs not available with a static FFmpeg build. Created `pipeline/demucs_runner.py` — a wrapper script that monkey-patches `torchaudio.save()` to use soundfile before importing Demucs. Wired into `splitter.py` so Demucs is invoked via the runner instead of `-m demucs`.
- **analyzer.py crash fixes**: Fixed unsafe `next(iter(stems.values()))` with `or` operator raising numpy array truthiness error; added empty stems guard; fixed harmonic stem fallback logic.
- **check_health deadlock**: Replaced `asyncio.create_subprocess_exec` with `asyncio.to_thread(subprocess.run)` to avoid event loop conflicts under FastMCP/anyio on Windows.

### New Files
- `pipeline/demucs_runner.py` — torchaudio.save() soundfile patch + Demucs CLI runner
- `tests/conftest.py` — shared pytest fixtures (synthetic 44.1kHz WAV, percussive beat signal)
- `tests/test_ingestion.py` — 12 tests (URL validation, duration limit, yt-dlp mocking)
- `tests/test_converter.py` — 5 tests (FFmpeg command, skip-if-exists, error handling)
- `tests/test_splitter.py` — 11 tests (layout detection, stem verification, proxy SDR)
- `tests/test_analyzer.py` — 27 tests (all feature extractors + integration)
- `tests/test_vectorizer.py` — 10 tests (CLAP path, librosa fallback, L2 norm)
- `tests/test_assembler.py` — 17 tests (payload structure, confidence scoring, telemetry)
- `smoke_test.py` — full end-to-end MCP stdio client test
- `pyproject.toml` — installable package config with optional `[clap]` and `[dev]` extras
- `.env.example` — documents STEMS_ROOT, TORCH_HOME, HF_HOME
- `.github/workflows/test.yml` — CI matrix (Python 3.10/3.11/3.12)
- `requirements-dev.txt` — pytest, pytest-asyncio

### Improvements
- **Dockerfile**: Added non-root `appuser` (UID 1000), `HEALTHCHECK` instruction, corrected install order
- **requirements.txt**: All dependencies pinned to exact versions; removed dead `aiofiles` dep
- **server.py**: Added request/response logging to all 3 MCP tools; venv PATH injection at startup
- **CLAUDE.md**: Updated container section to Podman-first with `:Z` volume flags
- **README.md**: Full rewrite — Podman as primary, Docker as alternative
- Deleted `tmp_verify.py` (dead code)

### Test Results
- 96/96 pytest tests passing

### End-to-End Smoke Test Results (partial — interrupted at Stage 5)
- Stage 1 Validate: ✓
- Stage 2 Download: ✓ (3.5 MB webm, ~4s)
- Stage 3 Convert: ✓ (38.2 MB WAV, ~1s)
- Stage 4 Demucs split: ✓ (proxy SDR 13.9 dB — excellent; ~7 min CPU)
- Stage 5 Analyze: running when session ended
- Stage 6 Vectorize: pending

## [2026-04-08] - Dockerization & Cloud Preparedness
- **Dockerfile Creation**: Created a multi-stage Dockerfile using `python:3.10-slim`, configuring a clean build by copying static `ffmpeg` binaries from `mwader/static-ffmpeg`.
- **Resource Optimization**: Implemented CPU-only PyTorch (`torch-cpu`, `torchaudio-cpu`) installation to significantly reduce container image size and allow deployment without GPU reliance.
- **Model Cache Configuration**: Mapped `.cache` locations via `ENV TORCH_HOME=/app/models/torch` and `ENV HF_HOME=/app/models/huggingface` to allow users to mount shared volumes; mitigating the "warmup problem" associated with Demucs and CLAP re-downloading weights.
- **Dockerignore**: Authored `.dockerignore` to keep contexts small (ignored `.venv`, temp outputs, and cached bytes).
- **Documentation**: Formally documented container build steps and correct volume mounting in `CLAUDE.md`.

## [2026-04-08] - Environment Refinement & Health Monitoring
- **Server Configuration**: Changed default `STEMS_ROOT` to a project-local `stems/` directory for easier output management.
- **Improved Monitoring**: Added a `check_health` tool to the MCP server to verify CLI dependencies (`ffmpeg`, `yt-dlp`) and Python packages (`demucs`, `librosa`, etc.) at runtime.
- **Repository Maintenance**: Created `.gitignore` to exclude system files and the new `stems/` directory.
- **Environment Verification**: Implemented and tested an internal health-check mechanism.

## [2026-04-08] - FastMCP Migration
- **Framework Upgrade**: Migrated the server from the low-level `mcp.server.Server` to `FastMCP`.
- **Tool Registration**: Simplified tool definition by replacing manual schemas with `@app.tool()` decorators.
- **Inspector Compatibility**: The server is now fully compatible with the MCP inspector (`mcp dev server.py`).

