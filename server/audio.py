import io
import numpy as np
import soundfile as sf

def wav_bytes_from_float32(float_audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, float_audio, samplerate=sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()

def float32_to_pcm16_bytes(float_audio: np.ndarray) -> bytes:
    clipped = np.clip(float_audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()
