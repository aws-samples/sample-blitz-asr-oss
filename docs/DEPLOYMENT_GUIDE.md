# Deployment Guide - Modular Architecture

## Overview

This guide covers deploying the refactored Blitz transcription service with the new modular architecture. The service has been refactored from a monolithic 1,706-line `server.py` to a clean modular structure with a 76% reduction in the main server file.

---

## What Changed

### Architecture Improvements

**Before (Monolithic):**
- Single 1,706-line `server.py` file
- Global variables for configuration and state
- ~300 lines of duplicated batch worker code
- No type safety or validation
- Difficult to test and maintain

**After (Modular):**
- Modular package structure (`blitz/`)
- Dependency injection (FastAPI `Depends()`)
- Generic batch worker with callbacks
- Full Pydantic type safety
- 76% smaller main server file (415 lines)

### Key Benefits

- ✅ **Code Reduction:** 76% smaller main file (1,706 → 415 lines)
- ✅ **Type Safety:** Pydantic schemas throughout
- ✅ **Testability:** Dependency injection, no global state
- ✅ **Maintainability:** Clear separation of concerns
- ✅ **Performance:** Same micro-batching and GPU optimizations

---

## Environment Variables

### Updated Variable Names

⚠️ **BREAKING CHANGES:** Two environment variables have been renamed for consistency:

| Old Name | New Name | Description |
|----------|----------|-------------|
| `SHORT_AUDIO_SEC` | `SHORT_AUDIO_THRESHOLD_SEC` | Threshold for direct array processing |
| `LOCAL_ATTENTION_AFTER_SEC` | `LOCAL_ATTENTION_THRESHOLD_SEC` | Threshold for local attention mode |

### Complete Configuration Reference

#### 3-Tier Model Routing
```bash
TIER1_LANGUAGES=en             # Languages routed to Tier 1 (CTC-110M, default: en)
TIER2_LANGUAGES=bg,hr,cs,...   # Languages routed to Tier 2 (Parakeet 0.6B, default: 25 EU)
# Tier 3 (Whisper) catches all other languages
```

Tier 1 (CTC-110M) provides 51% faster processing (233x real-time) compared to Tier 2 (154x real-time). Set `TIER1_LANGUAGES=""` to disable Tier 1 routing.

#### Micro-Batching Configuration
```bash
BATCH_TIMEOUT_MS=15.0          # Wait time to gather batch (0-1000ms)
MAX_BATCH_SIZE=4               # Max requests per batch (1-16)
```

#### Audio Processing Thresholds
```bash
SHORT_AUDIO_THRESHOLD_SEC=600      # Direct array processing threshold
LOCAL_ATTENTION_THRESHOLD_SEC=1440 # Local attention mode threshold
CHUNK_LENGTH_SEC=480               # Chunk size for long audio
CHUNK_OVERLAP_SEC=1                # Overlap between chunks
```

#### Language Configuration
```bash
TIER1_LANGUAGES=en              # Languages routed to Tier 1 (CTC-110M)
TIER2_LANGUAGES=bg,hr,cs,da,nl,et,fi,fr,de,el,hu,it,lv,lt,mt,pl,pt,ro,sk,sl,es,sv,ru,uk
# Tier 3 (Whisper) catches all other languages - no configuration needed
```

#### Language Detection
```bash
LANGUAGE_DETECTION_MODEL=large-v3-turbo  # or "tiny" to save 8GB
PRIMARY_LANGUAGE_DETECTOR=whisper        # "whisper" (GPU) or "voxlingua" (CPU)
WHISPER_CONFIDENCE_THRESHOLD=0.70        # Falls back to VoxLingua if below
VOXLINGUA_CONFIDENCE_THRESHOLD=0.90      # VoxLingua threshold (higher due to lower scores)
```

#### Model Configuration
```bash
TIER1_MODEL=nvidia/parakeet-tdt_ctc-110m          # Tier 1: English fast mode (110M params)
TIER1_PRECISION=fp16                               # Tier 1 precision
TIER2_MODEL=nvidia/parakeet-tdt-0.6b-v3           # Tier 2: Multilingual (25 EU languages)
TIER2_PRECISION=bfloat16                           # Tier 2 precision
TIER2_BACKEND=nemo                                 # Tier 2 backend: nemo or onnx
TIER3_MODEL=openai/whisper-large-v3-turbo         # Tier 3: Catch-all
VOXLINGUA_MODEL_NAME=speechbrain/lang-id-voxlingua107-ecapa
```

#### Device Configuration
```bash
DEVICE=cuda  # or "cpu" for CPU-only mode
```

#### Voice Activity Detection (VAD)
```bash
# VAD is always enabled (CPU-based Silero VAD)
# No configuration needed - automatic optimization for silent audio
# Silently skips transcription when <10% speech ratio detected
```

**VAD Benefits:**
- 🚀 **~80% faster** for silent files (150ms vs 700ms)
- 💰 **Zero GPU cost** for silent audio
- 🎯 **No hallucinations** from background noise

**VAD Performance:**
- Analysis overhead: ~80ms for first 60 seconds
- Savings on silent files: ~500ms+ (skips language detection + transcription)
- Net benefit: ~420ms faster for silent audio

**When VAD triggers:**
- Audio with <10% speech ratio (e.g., silence, white noise, ambient sound)
- Returns empty transcript with `engine="none"` and zero GPU cost
- Only first 60 seconds analyzed for efficiency on long files

---

## Docker Deployment

### Building the Image

The Dockerfile has been updated to support the modular architecture:

```bash
# Build the image
docker build -t blitz:latest .

# Build with custom configuration
docker build \
  --build-arg BATCH_TIMEOUT_MS=20.0 \
  --build-arg MAX_BATCH_SIZE=2 \
  -t blitz:custom .
```

### Running the Container

**Basic usage:**
```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  --name blitz \
  blitz:latest
```

**With custom configuration:**
```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  --name blitz \
  -e SHORT_AUDIO_THRESHOLD_SEC=300 \
  -e LOCAL_ATTENTION_THRESHOLD_SEC=1800 \
  -e MAX_BATCH_SIZE=2 \
  -e BATCH_TIMEOUT_MS=20.0 \
  blitz:latest
```

**For A100 80GB (higher thresholds):**
```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  --name blitz \
  -e SHORT_AUDIO_THRESHOLD_SEC=1200 \
  -e LOCAL_ATTENTION_THRESHOLD_SEC=3600 \
  -e MAX_BATCH_SIZE=4 \
  blitz:latest
```

### Dockerfile Changes

The updated Dockerfile includes:

1. **Added `pydantic-settings`** dependency
   ```dockerfile
   RUN pip install --no-cache-dir \
       pydantic \
       pydantic-settings \
       ...
   ```

2. **Updated environment variable names**
   ```dockerfile
   ENV SHORT_AUDIO_THRESHOLD_SEC=600
   ENV LOCAL_ATTENTION_THRESHOLD_SEC=1440
   ```

3. **Copy modular architecture**
   ```dockerfile
   COPY blitz/ /app/blitz/
   ```

4. **Use integrated server**
   ```dockerfile
   CMD ["uvicorn", "blitz.server:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

---

## Local Development

### Prerequisites

```bash
# Python 3.10+
python3 --version

# CUDA Toolkit (for GPU support)
nvcc --version

# FFmpeg (for audio processing)
ffmpeg -version
```

### Installation

1. **Clone and navigate:**
   ```bash
   git clone <repository>
   cd blitz
   git checkout refactor/modular-architecture
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

   # Or manually:
   pip install \
       nemo_toolkit[asr]==2.4.0 \
       soundfile \
       fastapi uvicorn[standard] \
       python-multipart \
       pydantic pydantic-settings \
       transformers torch torchaudio \
       sentencepiece torchmetrics \
       git+https://github.com/speechbrain/speechbrain.git@develop
   ```

3. **Set environment variables (optional):**
   ```bash
   export SHORT_AUDIO_THRESHOLD_SEC=600
   export LOCAL_ATTENTION_THRESHOLD_SEC=1440
   export MAX_BATCH_SIZE=4
   export BATCH_TIMEOUT_MS=15.0
   export DEVICE=cuda
   ```

### Running the Server

**Using integrated server (recommended):**
```bash
python3 -m blitz.server
```

**Using uvicorn directly:**
```bash
uvicorn blitz.server:app --host 0.0.0.0 --port 8000
```

**With live reload (development):**
```bash
uvicorn blitz.server:app --reload --host 0.0.0.0 --port 8000
```

---

## Testing the Deployment

### Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "models": {
    "tier1": "nvidia/parakeet-tdt_ctc-110m",
    "tier2": "nvidia/parakeet-tdt-0.6b-v3",
    "tier3": "openai/whisper-large-v3-turbo"
  },
  "config": {
    "short_audio_threshold_sec": 600,
    "local_attention_threshold_sec": 1440,
    "tier1_languages": ["en"],
    "tier2_languages": 25
  }
}
```

### Models Information

```bash
curl http://localhost:8000/models
```

### Test Transcription

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "format=json"
```

### Load Testing

```bash
# Run integration tests
python3 tests/test_integration.py

# Run load tests (5 concurrent requests)
python3 tests/test_load.py -n 5

# Run load tests (10 concurrent requests)
python3 tests/test_load.py -n 10
```

---

## Migration Guide

### Migrating from Original server.py

If you're currently using the original `server.py`, follow these steps:

1. **Update Docker environment variables:**
   ```bash
   # Old:
   -e SHORT_AUDIO_SEC=600
   -e LOCAL_ATTENTION_AFTER_SEC=1440

   # New:
   -e SHORT_AUDIO_THRESHOLD_SEC=600
   -e LOCAL_ATTENTION_THRESHOLD_SEC=1440
   ```

2. **Rebuild Docker image:**
   ```bash
   docker build -t blitz:latest .
   docker stop blitz
   docker rm blitz
   docker run -d --gpus all -p 8000:8000 --name blitz blitz:latest
   ```

3. **For local deployments:**
   ```bash
   # Update environment variables in your .env or systemd service
   # Then restart the service
   systemctl restart blitz
   ```

### API Compatibility

✅ **No API changes** - All endpoints remain the same:
- `POST /transcribe` - Same request/response format
- `GET /health` - Same response structure (enhanced with config info)
- `GET /models` - Same response structure

### Performance Characteristics

✅ **No performance regression** - All optimizations preserved:
- Micro-batching (30-50% throughput improvement)
- Local attention switching for long audio
- Chunking strategies for memory efficiency
- GPU memory cleanup (aggressive cleanup after batches)

### Voice Activity Detection (VAD) Optimization

The service includes automatic VAD-based optimization to skip transcription for silent audio.

**How VAD Works:**

1. **Early Detection (Pre-Transcription):**
   - Runs **before** language detection and transcription
   - Uses Silero VAD (CPU-based, no GPU required)
   - Analyzes first 60 seconds of audio for efficiency
   - Threshold: <10% speech ratio = silent

2. **Processing Pipeline:**
   ```
   Audio Upload → VAD Analysis (80ms) → Decision Point
                                              ├─ Silent (<10% speech)
                                              │  └─ Return empty transcript
                                              │     (150ms total, $0 GPU cost)
                                              │
                                              └─ Speech Detected (≥10% speech)
                                                 └─ Continue to language detection
                                                    and transcription
                                                    (700ms+ total, normal GPU cost)
   ```

3. **VAD Response Format:**
   - Empty text: `"text": ""`
   - Engine: `"engine": "none"`
   - Zero GPU cost: `"estimated_gpu_cost_usd": 0.0`
   - VAD metadata: speech ratio, duration, segments
   - Detection method: `"detection_method": "skipped_silent_audio"`

**Performance Impact:**

| Metric | Silent Audio (VAD) | Normal Audio |
|--------|-------------------|--------------|
| Processing Time | ~150ms | ~700ms+ |
| GPU Usage | None | Normal |
| Cost per Request | $0.000 | $0.00015-0.0008 |
| Speed Improvement | **~80% faster** | Baseline |

**Use Cases:**

- ✅ **Silence detection:** Pure silence files
- ✅ **Noise filtering:** White noise, ambient sound only
- ✅ **Recording gaps:** Dead air in recordings
- ✅ **Empty audio:** Blank or corrupted files
- ❌ **Low volume speech:** Still transcribed (VAD detects speech)
- ❌ **Whispered audio:** Still transcribed (VAD detects speech)

**Production Benefits:**

1. **Cost Savings:** Zero GPU cost for silent files reduces overall bill
2. **Faster Response:** 80% faster processing for silent audio
3. **No Hallucinations:** Avoids transcribing "phantom words" from noise
4. **Scalability:** Reduces GPU load during silent audio processing
5. **Concurrent Processing:** VAD runs on CPU while GPU handles transcription

**CPU/GPU Parallelization:**

```
┌─────────────────────────────────────────────────────────────┐
│                     Async Request Flow                       │
├─────────────────────────────────────────────────────────────┤
│  Request A: [Audio Load] ──► [Lang Detect GPU] ──► [Transcribe GPU]
│  Request B:    [Audio Load] ──► [VAD CPU] ──► silent → return
│  Request C:       [Audio Load] ──► [VAD CPU] ──► [Queue] ──► [GPU]
└─────────────────────────────────────────────────────────────┘
```

When multiple requests are processed concurrently:
- VAD analysis (CPU) runs in parallel with transcription (GPU)
- Silent files detected on CPU cost effectively **$0.00** while GPU is busy
- Async FastAPI allows overlapping of CPU and GPU workloads

**Monitoring VAD:**

Check the `vad` field in API responses:
```json
{
  "vad": {
    "is_silent": true,           // VAD triggered
    "speech_ratio": 0.05,        // Only 5% speech detected
    "speech_duration_sec": 0.5,  // 0.5 seconds of speech
    "vad_time_ms": 82.3,         // VAD analysis time
    "method": "silero_vad",      // VAD engine used
    "num_speech_segments": 2     // Number of speech chunks
  }
}
```

**Configuration:**

VAD is **always enabled** and requires no configuration. The service automatically:
- Uses CPU-based Silero VAD (no GPU overhead)
- Analyzes first 60 seconds only (for long files)
- Applies 10% speech ratio threshold
- Returns early for silent audio

To disable VAD (not recommended), modify `blitz/server.py` to skip the VAD check.

---

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'pydantic_settings'`

**Solution:**
```bash
pip install pydantic-settings
```

The code includes a fallback, but for full functionality, install `pydantic-settings`.

### Configuration Not Loading

**Problem:** Environment variables not being read

**Solution:**
1. Ensure variables are exported: `export VAR_NAME=value`
2. Check variable names match new naming (see "Updated Variable Names" above)
3. Verify `pydantic-settings` is installed for environment variable parsing

### GPU Out of Memory

**Problem:** CUDA OOM errors on long audio

**Solution:**
1. Lower `LOCAL_ATTENTION_THRESHOLD_SEC` (try 1200 for A10G, 2400 for A100)
2. Reduce `MAX_BATCH_SIZE` (try 2 for long audio)
3. Lower `CHUNK_LENGTH_SEC` (try 360 instead of 480)

### Module Import Issues

**Problem:** `ImportError: cannot import name 'ApplicationState'`

**Solution:**
1. Ensure you're on the `refactor/modular-architecture` branch
2. Verify the `blitz/` directory exists
3. Check that `blitz/__init__.py` exists

---

## Production Recommendations

### Instance Sizing

**Recommended GPU instances:**
- **g6.xlarge** (NVIDIA L4, 24GB): Good for most workloads
  - `LOCAL_ATTENTION_THRESHOLD_SEC=1440` (24 minutes)
  - `MAX_BATCH_SIZE=2-4`

- **g5.xlarge** (NVIDIA A10G, 24GB): Better performance
  - `LOCAL_ATTENTION_THRESHOLD_SEC=1440` (24 minutes)
  - `MAX_BATCH_SIZE=4`

- **p4d.24xlarge** (NVIDIA A100, 80GB): Maximum performance
  - `LOCAL_ATTENTION_THRESHOLD_SEC=3600` (60 minutes)
  - `MAX_BATCH_SIZE=8`

### Scaling Considerations

**Horizontal scaling (recommended architecture):**

```
┌──────────┐    ┌────────────┐    ┌───────────┐    ┌─────────────────────────┐
│ Producer │───►│  S3 Bucket │───►│ SQS Queue │───►│  Auto-Scaling Consumers │
└──────────┘    │  (audio)   │    └───────────┘    │  ┌───────┐ ┌───────┐    │
                └────────────┘          │          │  │ Blitz │ │ Blitz │ .. │
                      │                 │          │  │  GPU  │ │  GPU  │    │
                      │                 ▼          │  └───────┘ └───────┘    │
                      │          ┌────────────┐    └─────────────────────────┘
                      └─────────►│  S3 Bucket │◄───── (transcripts)
                                 │  (output)  │
                                 └────────────┘
```

**Flow:**
1. Producer uploads audio to S3 input bucket
2. S3 event notification triggers SQS message
3. SQS consumer reads message, downloads audio from S3
4. Consumer calls Blitz HTTP API to transcribe
5. Consumer writes transcript to S3 output bucket
6. Consumer deletes SQS message on success

**Auto-scaling metrics:**
- `ApproximateNumberOfMessages` - queue depth
- Composite metric: `queue_count * avg_audio_duration` for cost-aware scaling
- Scale to 0 when queue is empty (cost optimization)

> **Note:** This SQS-based architecture is not currently implemented but can be built
> using the existing Blitz container with an SQS consumer wrapper. See AWS Batch mode
> in [BATCH_PROCESSING.md](BATCH_PROCESSING.md) for a similar S3-triggered pattern.

**Current implementation (HTTP API):**
- Deploy multiple containers behind a load balancer
- Each instance handles its own micro-batching queue
- No shared state between instances
- Suitable for real-time synchronous transcription

**Vertical scaling:**
- Increase `MAX_BATCH_SIZE` for larger GPUs
- Increase thresholds for GPUs with more memory
- Monitor GPU memory usage and adjust accordingly

### Monitoring

**Key metrics to monitor:**
- GPU utilization (target: >70%)
- GPU memory usage (keep below 80%)
- Request latency (P50, P95, P99)
- Batch sizes (avg batch size should be close to max)
- Real-time factor (RTF < 0.1 for good performance)

**Logging:**
```bash
# View container logs
docker logs -f blitz

# Export logs to file
docker logs blitz > logs.txt
```

---

## File Structure

The new modular architecture organizes code as follows:

```
blitz/
├── blitz/                   # Main package
│   ├── __init__.py                # Package exports
│   ├── app_state.py               # Application state & DI
│   ├── batch_processing.py        # Engine-specific batch processing
│   ├── config.py                  # Centralized configuration
│   ├── models/                    # Model wrappers
│   │   ├── __init__.py
│   │   ├── parakeet.py           # Parakeet model wrapper
│   │   ├── whisper.py            # Whisper model wrapper
│   │   └── voxlingua.py          # VoxLingua model wrapper
│   ├── services/                  # Business logic
│   │   ├── __init__.py
│   │   ├── batch.py              # Generic batch worker
│   │   ├── language_detection.py # Language detection service
│   │   └── transcription.py      # Transcription orchestration
│   ├── schemas/                   # API schemas
│   │   ├── __init__.py
│   │   └── api.py                # Pydantic response models
│   └── utils/                     # Utilities
│       ├── __init__.py
│       ├── audio.py              # Audio processing utils
│       └── formatting.py         # Subtitle formatting utils
│   └── server.py                 # Main server (~400 lines)
├── docker/
│   ├── Dockerfile.batch          # AWS Batch image
│   └── Dockerfile.onnx           # ONNX backend image
├── Dockerfile                    # Production Docker image
├── tests/
│   ├── test_integration.py       # Integration tests
│   └── test_load.py              # Load tests
└── tests/fixtures/               # Test audio files
```

---

## Support

For issues, questions, or contributions:
- **Issues:** Check existing issues or create a new one
- **Documentation:** See `docs/` directory for detailed guides
- **Architecture:** See `docs/REFACTORING_PROGRESS.md` for architecture details

---

## Changelog

### v2.2.0 - 3-Tier Model Routing (January 2026)

**Added:**
- Generalized 3-tier model routing with configurable language lists
- `TIER1_LANGUAGES`, `TIER2_LANGUAGES` environment variables
- `TIER1_MODEL`, `TIER2_MODEL`, `TIER3_MODEL` configuration
- `TIER2_BACKEND`, `TIER2_PRECISION` configuration
- Cost analysis per tier in benchmark-results.md

**Breaking Changes:**
- `ENGLISH_FAST_MODE` replaced with `TIER1_LANGUAGES=en`
- `PARAKEET_PRECISION` replaced with `TIER2_PRECISION`
- `PARAKEET_BACKEND` replaced with `TIER2_BACKEND`
- Engine names changed: `parakeet_ctc` -> `tier1`, `parakeet` -> `tier2`, `whisper` -> `tier3`

**Performance:**
- Tier 1: 233x real-time (RTF 0.0043) with CTC-110M
- Tier 2: 154x real-time (RTF 0.0065) with Parakeet 0.6B
- Tier 3: ~50x real-time with Whisper

**Cost (g6.xlarge Spot):**
- Tier 1 (CTC-110M): $0.03/1000 min
- Tier 2 (Parakeet 0.6B): $0.06-0.08/1000 min
- Tier 3 (Whisper): $0.08-0.15/1000 min
- Silent audio (VAD skip): ~$0.00/1000 min (CPU only)

### v2.0.0 - Modular Architecture (December 2025)

**Added:**
- Modular package structure (`blitz/`)
- Dependency injection pattern
- Pydantic configuration and schemas
- Generic batch worker with callbacks
- Comprehensive type safety

**Changed:**
- Environment variables renamed for consistency
- Server entry point: `blitz.server`
- 76% code reduction in main server file

**Improved:**
- Testability through dependency injection
- Maintainability through separation of concerns
- Type safety through Pydantic validation

**Backward Compatibility:**
- Original `server.py` still available
- All API endpoints unchanged
- No performance regression
