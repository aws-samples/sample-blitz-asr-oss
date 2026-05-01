# Quick Start Guide

## Build & Run (One Command)

```bash
# Build
DOCKER_BUILDKIT=1 docker build -t blitz:latest .

# Run
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8003:8000 --name blitz blitz:latest
```

## Test API

```bash
# Single file
curl -X POST "http://localhost:8003/transcribe" \
  -H "accept: application/json" \
  -F "file=@audio.wav"

# Expected output:
# {
#   "transcription": "...",
#   "language": "en",
#   "model": "tier2",
#   "metrics": { ... }
# }
```

## Generate Subtitles

```bash
# SRT format
curl -X POST "http://localhost:8003/transcribe" \
  -F "file=@audio.wav" \
  -F "format=srt" > subtitles.srt

# VTT format
curl -X POST "http://localhost:8003/transcribe" \
  -F "file=@audio.wav" \
  -F "format=vtt" > subtitles.vtt
```

## Process Long Audio

```bash
# Split into 10-minute chunks
ffmpeg -i long_audio.wav -f segment -segment_time 600 -c copy chunk_%03d.wav

# Process all chunks
for f in chunk_*.wav; do
  curl -sS -X POST "http://localhost:8003/transcribe" \
    -F "file=@${f}" | jq '.'
done
```

## Configure Tier 1 (English Fast Mode)

```bash
# Default: English routes to Tier 1 (CTC-110M, 233x real-time)
docker run --gpus all -p 8003:8000 blitz:latest

# Disable Tier 1 (route English to Tier 2 instead)
docker run --gpus all -p 8003:8000 \
  -e "TIER1_LANGUAGES=" \
  blitz:latest
```

## Configure Precision

```bash
# BFloat16 (Default) - Best balance of speed and accuracy
docker run --gpus all -p 8003:8000 blitz:latest

# FP32 - Maximum accuracy
docker run --gpus all -p 8003:8000 \
  -e "TIER2_PRECISION=fp32" \
  blitz:latest

# FP16 ONNX - Maximum speed (build with ONNX enabled)
docker build --build-arg ENABLE_ONNX=true -t blitz-onnx .
docker run --gpus all -p 8003:8000 \
  -e "TIER2_BACKEND=onnx" \
  blitz-onnx
```

## Configure Languages

```bash
# Route English and German to Tier 1 (fastest)
docker run --gpus all -p 8003:8000 \
  -e "TIER1_LANGUAGES=en,de" \
  blitz:latest

# All languages to Tier 3 (Whisper) - disable Tier 1 and 2
docker run --gpus all -p 8003:8000 \
  -e "TIER1_LANGUAGES=" \
  -e "TIER2_LANGUAGES=" \
  blitz:latest
```

## Configure Efficiency Thresholds

```bash
# For A100 80GB (more aggressive, longer audio)
docker run --gpus all -p 8003:8000 \
  -e "SHORT_AUDIO_THRESHOLD_SEC=900" \
  -e "LOCAL_ATTENTION_THRESHOLD_SEC=3600" \
  blitz:latest

# For T4 16GB (conservative, prevent OOM)
docker run --gpus all -p 8003:8000 \
  -e "SHORT_AUDIO_THRESHOLD_SEC=300" \
  -e "LOCAL_ATTENTION_THRESHOLD_SEC=720" \
  blitz:latest

# Defaults (L4 24GB): SHORT_AUDIO_THRESHOLD_SEC=300, LOCAL_ATTENTION_THRESHOLD_SEC=1440
```

## Useful Commands

```bash
# View logs
docker logs -f blitz

# Check GPU usage
docker exec -it blitz nvidia-smi

# Stop container
docker stop blitz

# Restart
docker restart blitz

# Remove
docker rm -f blitz
```

## Model Routing Logic

1. **Detect language** with Whisper (GPU) + VoxLingua107 fallback
2. **Route to Tier 1** if language in `TIER1_LANGUAGES` (default: English)
3. **Route to Tier 2** if language in `TIER2_LANGUAGES` (default: 25 EU languages)
4. **Route to Tier 3** for all other languages (Whisper catch-all)

```
Audio --> Language Detection --> in TIER1_LANGUAGES? --> Tier 1: CTC-110M (233x)
                             --> in TIER2_LANGUAGES? --> Tier 2: Parakeet 0.6B (154x)
                             --> else                --> Tier 3: Whisper (~50x)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port already in use | Use `-p 8004:8000` instead |
| Out of memory | Process shorter chunks or reduce `MAX_BATCH_SIZE` |
| Wrong language detected | Check audio quality, ensure 3+ seconds |
| Models not downloading | `docker build --network=host ...` |

## Performance

**English Fast Mode (Default):**
- **Speed**: 233x real-time (RTF 0.0043)
- **Accuracy**: 92.5% (7.49% WER)
- **Model**: CTC-110M (110M params)

**Parakeet 0.6B (European languages):**
- **Speed**: 154x real-time (RTF 0.0065)
- **Accuracy**: 94.46% (5.54% WER)
- **Languages**: 25 European languages

**Model Comparison:**

| Model | Speed | Accuracy | Languages | Use Case |
|-------|-------|----------|-----------|----------|
| CTC-110M | 233x | 92.5% | English only | Fast English |
| Parakeet 0.6B | 154x | 94.5% | 25 European | Production |
| Whisper | ~50x | 90%+ | 100+ | Wide coverage |

See [benchmark-results.md](benchmark-results.md) for detailed analysis.
