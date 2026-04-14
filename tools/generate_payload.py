import os
import json
import time
import sys
from pathlib import Path
from pipeline.analyzer import analyze_stems
from pipeline.vectorizer import generate_vibe_vector
from pipeline.assembler import assemble_payload
from pipeline.ingestion import load_metadata

# Prevent torchcodec DLL errors
os.environ["TORCHAUDIO_USE_BACKEND_PREFERENCE"] = "soundfile"


def main():
    job_id = sys.argv[1] if len(sys.argv) > 1 else "sig_970a1a4a"
    stems_base = Path(f"stems/{job_id}/stems")

    # Auto-discover the model subdirectory (htdemucs, htdemucs_ft, etc.)
    stems_dir = None
    for candidate in [stems_base / "htdemucs", stems_base / "htdemucs_ft", stems_base]:
        if all(
            (candidate / f"{s}.wav").exists()
            for s in ["vocals", "drums", "bass", "other"]
        ):
            stems_dir = candidate
            break

    if not stems_dir:
        print(f"ERROR: Could not find stems for job '{job_id}' in {stems_base}")
        sys.exit(1)

    stem_files = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]

    # Load persisted metadata (Single Source of Truth)
    job_dir = Path(f"stems/{job_id}")
    source_info = load_metadata(job_dir) or {"title": "Unknown", "uploader": "Unknown"}
    print(f"Source: '{source_info.get('title')}' by {source_info.get('uploader')}")

    print("[5/6] Extracting BPM, key, transients, frequencies...")
    t0 = time.perf_counter()
    features = analyze_stems(stems_dir, stem_files)
    print(f"Analyze finished in {time.perf_counter() - t0:.2f}s")
    print(f"  BPM={features['bpm']}  Key={features['key']}")

    print("[6/6] Generating vibe vector...")
    t0 = time.perf_counter()
    vibe_vector = generate_vibe_vector(stems_dir, stem_files)
    print(f"Vectorize finished in {time.perf_counter() - t0:.2f}s")

    payload = assemble_payload(
        job_id=job_id,
        stems_dir=stems_dir,
        stem_files=stem_files,
        sdr=float(source_info.get("sdr_ratio", 8.0)),
        features=features,
        vibe_vector=vibe_vector,
        inference_time=0.0,
        cpu_samples=[],
        source_info=source_info,
    )

    out_path = f"gold_master_{job_id}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Successfully wrote {out_path}!")


if __name__ == "__main__":
    main()
