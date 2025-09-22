from __future__ import annotations
import os
import numpy as np
from typing import Optional
from TTS.api import TTS

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "24000"))
MODEL_NAME  = os.getenv("MODEL_NAME", "tts_models/multilingual/multi-dataset/your_tts")
DEVICE      = os.getenv("DEVICE", "cuda")

_tts: Optional[TTS] = None

def get_tts() -> TTS:
    global _tts
    if _tts is None:
        _tts = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=(DEVICE == "cuda"))
    return _tts

def synthesize(
    text: str,
    speaker_id: Optional[str],
    reference_wav: Optional[str],
    language: str,
    speed: float,
    pitch_semitones: float,
    energy: float,
    style_strength: float,
    emotion_id: Optional[str],
) -> np.ndarray:
    tts = get_tts()

    # Base kwargs for YourTTS/Coqui API
    kwargs = {
        "text": text,
        "language": language,
        "speaker": speaker_id,
        "speaker_wav": reference_wav,
        "speed": speed,
    }

    audio = tts.tts(**kwargs)
    audio = np.asarray(audio, dtype=np.float32)

    if energy != 1.0:
        audio *= float(energy)

    # TODO: optional pitch shift with phase vocoder if you want realtime pitch control.
    peak = float(np.max(np.abs(audio)) + 1e-9)
    return (audio / peak).astype(np.float32)
