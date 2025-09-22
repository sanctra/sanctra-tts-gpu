#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME="${MODEL_NAME:-tts_models/multilingual/multi-dataset/your_tts}"
export SAMPLE_RATE="${SAMPLE_RATE:-24000}"
export DEVICE="${DEVICE:-cuda}"
export SPEAKER_DIR="${SPEAKER_DIR:-/srv/server/models/speakers}"

mkdir -p "$SPEAKER_DIR"

echo "Starting TTS with MODEL_NAME=$MODEL_NAME, SAMPLE_RATE=$SAMPLE_RATE, DEVICE=$DEVICE"
exec python3 -m uvicorn server.main:app --host 0.0.0.0 --port 9200
