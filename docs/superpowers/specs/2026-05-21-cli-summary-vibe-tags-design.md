# CLI Summary View + CLAP Vibe Tags — Design Spec

- **Date:** 2026-05-21
- **Status:** Approved (pending written-spec review)
- **Topic:** Two musician-facing additions to the local CLI (`analyze_file.py`): (1) human-readable `--summary` and `--no-vector` output flags, and (2) a new `vibe_tags` field — CLAP zero-shot mood/genre words derived from the existing audio embedding. The MCP server is **not touched**.

## Problem & Motivation

The local CLI prints the full sonic-signature JSON, which is ~570 lines — dominated by the 512-float `vibe_vector` that means nothing to a human. The target audience (musicians, audio engineers, singers, producers) wants to glance at a track's **key, tempo, vibe, and production profile** in the terminal, not scroll past a numeric array.

Two gaps:
1. **No readable terminal view.** Every field except `vibe_vector` is already human-meaningful, but they're buried in JSON.
2. **The vibe is machine-only.** CLAP produces a 512-dim embedding with no human-readable counterpart. A musician can't read "what does this sound like" from it.

## Goals

- A `--summary` flag that prints a compact, musician-friendly digest **instead of** the JSON.
- A `--no-vector` flag that prints the **full JSON minus** the 512-float `vibe_vector` array.
- The two flags compose; `--out` always writes the **complete** JSON regardless of flags.
- A new `vibe_tags` field: top ~5 CLAP zero-shot mood/genre/texture words, added to the JSON payload, computed **by the CLI only**.
- Graceful, honest fallback when CLAP is unavailable (no junk tags).
- Reuse the exact pipeline functions; keep the MCP/URL path byte-for-byte unchanged.

## Non-Goals (out of scope)

- **Any change to `server.py` / the MCP tool surface.** The MCP/URL path does not compute `vibe_tags` and does not pass it to the assembler; its output is unchanged.
- Changing `generate_vibe_vector`'s signature (the MCP path imports it).
- Training, fine-tuning, or downloading any new model — `vibe_tags` reuses the already-loaded CLAP.
- A configurable/user-supplied label vocabulary (the word list is fixed in code for v1).
- Colorized/ANSI terminal output (plain text for v1).

## Design

### 1. CLI flags — `analyze_file.py`

Two new `argparse` flags:

| Flag | Short | Effect |
|---|---|---|
| `--summary` | `-s` | Print **only** the human-readable digest (no JSON) |
| `--no-vector` | — | Print full JSON with `sonic_signature.vibe_vector` omitted |

- Flags compose. If both are given, `--summary` wins for stdout (digest only).
- `--out FILE` always writes the **complete** JSON (incl. `vibe_vector`), independent of stdout flags.
- Default (no flags): unchanged — full JSON to stdout.

### 2. Summary layout — `_print_summary(payload: dict) -> None` (in `analyze_file.py`)

Pure formatter over a payload dict (no audio, no model). Target layout:

```
🎵 SONIC SIGNATURE — input.mp3  (1:42)

  TEMPO    153.9 BPM  (steady)
  KEY      G Major  ·  shifts to G Phrygian @0:30   (confidence 63%)
  VIBE     aggressive · dark · hip-hop · hard-hitting · gritty

  PRODUCTION
    Vocals     forward
    Punch      0.33  (moderate)
    Stereo     narrow
    Low end    ~49 Hz dominant

  Overall confidence: 78%   ·   analyzed in 5m 23s
```

Formatting rules:
- Title from `header.source_metadata.title`; duration from `duration_sec` as `m:ss`.
- **Tempo:** `(steady)` when `bpm_variable` is false; `(variable NN–MM)` using `bpm_range` when true.
- **Key:** base `key` + `(confidence NN%)` from `mode_confidence`. When `key_variable`, append `· shifts to <next key> @m:ss` from the first differing `key_map` entry.
- **Vibe:** `vibe_tags` joined by ` · `. When `vibe_tags` is `None`/empty: `(unavailable — CLAP not installed)`.
- **Punch:** `transient_punch` value + qualitative label (`low` < 0.33 ≤ `moderate` < 0.66 ≤ `high`).
- **Low end:** rounded **mean** of `production_profile.dominant_freq_peaks_hz.harmonic`.
- **Overall confidence** from `header.confidence_score` as %; time from `telemetry.inference_time_sec` as `m:ss`.

### 3. Vibe tags — `generate_vibe_tags(wav_path: Path, full_song: bool = False) -> list[str] | None` (in `pipeline/vectorizer.py`)

CLAP zero-shot classification reusing the already-loaded CLAP model:
1. If CLAP is unavailable (same check `generate_vibe_vector` uses to fall back to librosa), return `None`.
2. Compute the audio embedding for the WAV (honoring `full_song` window logic, mirroring `_clap_vector`).
3. Embed each candidate prompt with CLAP's **text** encoder.
4. Cosine-similarity audio-vs-each-text; rank descending.
5. Drop tags below a similarity floor (`VIBE_TAG_FLOOR`, tuned constant); take top `VIBE_TAG_TOP_N` (default 5).
6. Return the bare words (prompt template stripped). If nothing clears the floor, return the single highest-ranked word (CLAP ran, so it's still meaningful). **`None` is returned only when CLAP is unavailable** — so in the summary, `None` ⇒ "unavailable", a non-empty list ⇒ real tags.

**Label vocabulary** (fixed constant, ~30–40 words across 4 axes; ranked together):

| Axis | Words |
|---|---|
| Mood | dark, bright, melancholic, uplifting, aggressive, chill, dreamy, tense, warm, romantic, melodic |
| Energy | energetic, mellow, driving, laid-back, hard-hitting, smooth |
| Genre | hip-hop, trap, lo-fi, jazz, rock, electronic, ambient, soul, R&B, pop, funk, classical |
| Texture | gritty, clean, distorted, acoustic, synthetic, lush, sparse |

Prompt template: `"this music sounds {word}"` for mood/energy/texture; `"a {word} track"` for genre (tuned for CLAP text-side signal).

Constants exposed at module top: `VIBE_TAG_LABELS`, `VIBE_TAG_TOP_N = 5`, `VIBE_TAG_FLOOR` (starting value calibrated during implementation against real runs; if nothing clears the floor, return the single top tag rather than `None`, so a confident-enough audio always yields at least one word).

> **Note:** self-contained — one extra CLAP audio forward pass (~1–2s) rather than threading the embedding out of `generate_vibe_vector`. This keeps `generate_vibe_vector`'s signature untouched (MCP-safe).

### 4. Assembler — `pipeline/assembler.py`

`assemble_payload(..., vibe_tags: list[str] | None = None)`:
- When `vibe_tags` is provided (non-None), set `sonic_signature.vibe_tags = vibe_tags`.
- When `None` (the MCP/URL default), the field is **omitted** — payload is otherwise byte-for-byte unchanged.

### 5. CLI flow change — `analyze_file.py`

After `vibe_vector = generate_vibe_vector(wav, full_song=True)`:
- `vibe_tags = generate_vibe_tags(wav, full_song=True)`
- pass `vibe_tags=vibe_tags` into `assemble_payload(...)`.
- stdout: `_print_summary(payload)` if `--summary`, else JSON (with `vibe_vector` popped if `--no-vector`).
- `--out`: always the complete payload.

### 6. README

Document `--summary` / `--no-vector` in the CLI args table; add a short note on `vibe_tags` (CLAP zero-shot, requires the CLAP extra; falls back to unavailable).

## Testing

- **`generate_vibe_tags`:**
  - Returns `None` when CLAP unavailable (mock the availability check).
  - **Selection logic** with a mocked model: controlled similarities → assert top-N ordering and that the floor drops weak tags.
- **`assemble_payload`:**
  - `vibe_tags` present in `sonic_signature` when passed.
  - **Absent, and the rest of the payload unchanged, when not passed** (MCP-safety regression).
- **`_print_summary`** (pure dict, no audio):
  - Renders TEMPO/KEY/VIBE/PRODUCTION/confidence lines for a sample payload.
  - `key_variable` path shows the shift line; `bpm_variable` shows the range.
  - `vibe_tags=None` → `(unavailable — CLAP not installed)`.
- **CLI flags (subprocess on synthetic WAV):**
  - `--summary` → digest on stdout, **no 512-array**.
  - `--no-vector` → JSON without `vibe_vector`.
  - `--out` → file still has the **complete** JSON incl. `vibe_vector`.
- **Regression:** existing CLI + server + pipeline suites stay green (assembler default arg, `generate_vibe_vector` unchanged).

## Risks / Notes

- **CLAP cosine scores are compressed**; the similarity floor + top-N is what prevents junk tags on ambiguous/sparse audio. Floor value is tuned against real runs, not stale fixtures.
- **Tags require the CLAP extra.** Without it the librosa-fallback vector is not in CLAP space, so `vibe_tags` is `None` — the summary says so honestly rather than fabricating.
- **No `server.py` import in the CLI** remains intact; the only shared code is the `pipeline/` package.
- **Extra runtime:** one additional CLAP forward pass on the CLI path only; negligible vs Demucs.
