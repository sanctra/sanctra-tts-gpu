from __future__ import annotations
import json, asyncio
from typing import Optional, List, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from pydantic import BaseModel, Field
import numpy as np

from .inference import synthesize, SAMPLE_RATE
from .audio import wav_bytes_from_float32, float32_to_pcm16_bytes
from .enroll import enroll_speaker

app = FastAPI(title="sanctra-tts-gpu", version="0.1.0")

class TtsPayload(BaseModel):
    text: str
    speaker_id: Optional[str] = None
    reference_wav_url: Optional[str] = None
    language: str = Field(default="en", pattern="^(en|es|fr|zh)$")
    emotion_id: Optional[str] = None
    style_strength: float = 0.6
    speed: float = 1.0
    pitch_semitones: float = 0.0
    energy: float = 1.0
    format: str = Field(default="wav", pattern="^(wav|pcm16)$")

@app.get("/healthz")
def healthz():
    return {"ok": True, "sr": SAMPLE_RATE}

@app.post("/tts")
def tts(payload: TtsPayload):
    audio = synthesize(
        text=payload.text,
        speaker_id=payload.speaker_id,
        reference_wav=payload.reference_wav_url,
        language=payload.language,
        speed=payload.speed,
        pitch_semitones=payload.pitch_semitones,
        energy=payload.energy,
        style_strength=payload.style_strength,
        emotion_id=payload.emotion_id,
    )
    wav = wav_bytes_from_float32(audio, SAMPLE_RATE)
    from fastapi.responses import Response
    return Response(content=wav, media_type="audio/wav")

@app.websocket("/tts/stream")
async def tts_stream(ws: WebSocket):
    await ws.accept()
    try:
        init_msg = await ws.receive_text()
        cfg = TtsPayload(**json.loads(init_msg))
        # For now synthesize whole text, then stream out as PCM16 frames (160 ms)
        audio = synthesize(
            text=cfg.text,
            speaker_id=cfg.speaker_id,
            reference_wav=cfg.reference_wav_url,
            language=cfg.language,
            speed=cfg.speed,
            pitch_semitones=cfg.pitch_semitones,
            energy=cfg.energy,
            style_strength=cfg.style_strength,
            emotion_id=cfg.emotion_id,
        )
        pcm = float32_to_pcm16_bytes(audio)
        bytes_per_frame = int(0.160 * SAMPLE_RATE) * 2  # 160ms * 2 bytes (16-bit mono)
        for i in range(0, len(pcm), bytes_per_frame):
            await ws.send_bytes(pcm[i:i+bytes_per_frame])
            await asyncio.sleep(0)  # yield to event loop
        await ws.close()
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.close(code=1011, reason=str(e))

class EnrollReq(BaseModel):
    speaker_id: str
    clips: List[Dict[str, str]]
    language: str = "en"

@app.post("/speakers/enroll")
def enroll(req: EnrollReq):
    return enroll_speaker(req.speaker_id, req.clips)
