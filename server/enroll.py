from __future__ import annotations
import os, requests
from typing import List, Dict

SPEAKER_DIR = os.getenv("SPEAKER_DIR", "/srv/server/models/speakers")

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _download(url: str, dst_path: str):
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def enroll_speaker(speaker_id: str, clips: List[Dict[str, str]]):
    """
    Zero/few-shot enrollment: store reference WAVs under SPEAKER_DIR/speaker_id.
    Later you can swap this to kick off a VITS fine-tune/LoRA job without changing the API.
    """
    base = os.path.join(SPEAKER_DIR, speaker_id)
    _ensure_dir(base)
    for i, clip in enumerate(clips):
        url = clip["url"]
        out = os.path.join(base, f"ref_{i:02d}.wav")
        _download(url, out)
    return {"status": "ready", "speaker_id": speaker_id, "clips": len(clips)}
