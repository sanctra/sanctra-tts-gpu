# sanctra-tts-gpu

Self-hosted GPU TTS (YourTTS/VITS lineage) for Sanctra.

## Endpoints
- `GET /healthz` → `{ok: true}`
- `POST /tts` → `audio/wav` (sync)
- `WS /tts/stream` → PCM16 frames (160 ms)
- `POST /speakers/enroll` → store reference WAVs for zero/few-shot

## Env
- `MODEL_NAME` (default `tts_models/multilingual/multi-dataset/your_tts`)
- `SAMPLE_RATE` (default `24000`)
- `DEVICE` (`cuda` or `cpu`, default `cuda`)
- `SPEAKER_DIR` (default `/srv/server/models/speakers`)
