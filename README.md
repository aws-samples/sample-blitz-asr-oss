# Blitz ASR

GPU-accelerated speech-to-text with intelligent model routing. Supports 100+ languages via Parakeet (25 EU languages) and Whisper (100+ languages).

## Features

- **3-Tier Model Routing**: Tier 1 (CTC-110M, English), Tier 2 (Parakeet 0.6B, 25 EU languages), Tier 3 (Whisper, 100+ languages)
- **Automatic language detection** with Whisper + VoxLingua107 fallback
- **Intelligent routing** to optimal model based on detected language
- **Silent audio detection**: VAD on CPU skips silent files, saving GPU cost
- **High throughput**: 1.6 min/s capacity, 0.55 req/s
- **Multiple outputs**: text, JSON, SRT, VTT with timestamps
- **Long-form audio**: chunking with overlap for hours of content

## Deployment Options

| Mode | Use Case | Trigger |
|------|----------|---------|
| **HTTP API** | Real-time transcription | REST endpoint |
| **AWS Batch** | Bulk file processing | S3 upload |

---

## HTTP API Mode

Synchronous FastAPI service with micro-batching for optimal GPU utilization.

### Quick Start (Docker Compose)

The easiest way to get started:

```bash
# Copy environment config
cp .env.example .env

# Build and start (first run: ~15-20 min for model downloads)
docker compose up -d --build

# Watch logs until "Application startup complete"
docker compose logs -f

# Test
curl -X POST http://localhost:8000/transcribe \
  -F "file=@tests/fixtures/english.mp3" -F "format=json"
```

### Quick Start (Manual)

```bash
# Build
docker build -t blitz:latest .

# Run
docker run --gpus all -p 8000:8000 blitz:latest

# Test
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" -F "format=json"
```

### Interactive Notebooks

For step-by-step guides, see the `notebooks/` directory:
- `01_quickstart_local.ipynb` - Get running on a GPU instance
- `02_aws_batch_deployment.ipynb` - Deploy AWS Batch for production scale

### API

**POST /transcribe**

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | File | Audio file (WAV, MP3, M4A, FLAC) |
| `format` | String | `text`, `json`, `srt`, `vtt` (default: text) |
| `language_hint` | String | ISO 639-1 code to skip detection |

**Response (JSON):**
```json
{
  "transcription": "Hello world.",
  "language": "en",
  "model": "tier2",
  "metrics": {
    "audio_duration_sec": 24.0,
    "real_time_factor": 0.069
  }
}
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TIER1_LANGUAGES` | en | Languages routed to Tier 1 (CTC-110M) |
| `TIER2_LANGUAGES` | bg,hr,cs,... | Languages routed to Tier 2 (Parakeet 0.6B) |
| `TIER2_PRECISION` | bfloat16 | `fp32`, `bfloat16`, `fp16` |
| `BATCH_TIMEOUT_MS` | 15 | Max wait for micro-batch (0-15ms latency) |
| `MAX_BATCH_SIZE` | 4 | Requests per batch |

---

## AWS Batch Mode

Event-driven processing triggered by S3 uploads.

```
S3 Input --> EventBridge --> AWS Batch --> S3 Output
```

### Deploy

```bash
./scripts/deploy_batch.sh \
  --environment dev \
  --region us-east-1 \
  --vpc-id vpc-xxx \
  --subnet-ids subnet-aaa,subnet-bbb
```

### Use

```bash
# Upload triggers transcription
aws s3 cp audio.mp3 s3://blitz-input-dev-123456789012/

# Get result
aws s3 cp s3://blitz-output-dev-123456789012/transcripts/audio.json .
```

### Output

```json
{
  "transcription": {
    "text": "Hello world.",
    "language": {"code": "en", "confidence": 0.98}
  },
  "audio": {"duration_sec": 1845.2},
  "metrics": {"rtf": 0.008}
}
```

See [docs/BATCH_PROCESSING.md](docs/BATCH_PROCESSING.md) for full configuration.

---

## Performance

Benchmarked on g6.xlarge (NVIDIA L4 24GB):

| Metric | Value |
|--------|-------|
| Throughput | 0.55 req/s |
| Capacity | 1.6 min audio/s |
| RTF | 0.006-0.01 |

### Cost per 1000 Minutes (Spot)

| Engine | Languages | Cost |
|--------|-----------|------|
| tier1 | English | **$0.03** |
| tier2 | 25 EU languages | $0.06-0.08 |
| tier3 | 100+ languages | $0.08-0.15 |

## Model Routing

| Audio Language | Model | Speed | Accuracy | Notes |
|----------------|-------|-------|----------|-------|
| English | CTC-110M | 233x | 92.5% | Fast mode (default) |
| 24 EU languages | Parakeet 0.6B | 154x | 94.5% | High accuracy |
| 100+ languages | Whisper | ~50x | 90%+ | Wide coverage |

## Precision Modes

| Mode | Speed | Accuracy | Use Case |
|------|-------|----------|----------|
| bfloat16 (default) | 154x | 94.5% | Production |
| fp32 | 100x | 94.3% | Max accuracy |
| onnx fp16 | 167x | 89.7% | Max speed |

## Requirements

- Docker with NVIDIA GPU support
- NVIDIA GPU with 24GB+ VRAM (L4, A10G, A100)
- CUDA 12.0+

## Project Structure

```
.env.example        # Configuration template
docker-compose.yml  # One-command setup
Dockerfile          # HTTP API image
blitz/
  models/           # Parakeet, Whisper, VoxLingua wrappers
  services/         # Transcription, language detection, VAD
  batch/            # AWS Batch processing
  server.py         # FastAPI HTTP server
  config.py         # Environment configuration
docker/
  Dockerfile.batch  # AWS Batch image
  Dockerfile.onnx   # ONNX backend image
notebooks/          # Interactive getting-started guides
scripts/            # Deployment and utility scripts
tests/fixtures/     # Test audio files (25+ languages)
```

## Documentation

- [API Documentation](docs/API_DOCUMENTATION.md)
- [Batch Processing](docs/BATCH_PROCESSING.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- [Benchmark Results](docs/benchmark-results.md)

## License

MIT
