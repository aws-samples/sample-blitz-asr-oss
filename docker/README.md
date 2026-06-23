# Docker Build Variants

Blitz ASR provides multiple Dockerfile variants optimized for different deployment scenarios.

## Dockerfile Overview

| File | Base Image | Purpose | GPU Required |
|------|-----------|---------|--------------|
| `Dockerfile` (root) | `nvcr.io/nvidia/pytorch:24.02-py3` | **Primary** — Full API server with all 3 tiers | ✅ Yes |
| `docker/Dockerfile.onnx` | `nvcr.io/nvidia/pytorch:24.02-py3` | ONNX-optimized — 50% smaller Tier 2 model, faster inference | ✅ Yes |
| `docker/Dockerfile.batch` | `nvcr.io/nvidia/pytorch:24.02-py3` | Batch processing — Optimized for AWS Batch (S3 input/output) | ✅ Yes |

---

## Primary Dockerfile (Root)

The main Dockerfile used for the HTTP API deployment.

### Build Options

```bash
# Standard build (downloads models during build — 30-40 min)
docker build -t blitz:latest .

# Skip model downloads (use with volume-mounted pre-cached models)
docker build --build-arg SKIP_MODEL_DOWNLOAD=true -t blitz:latest .

# Enable ONNX backend for Tier 2
docker build --build-arg ENABLE_ONNX=true -t blitz-onnx:latest .

# Both options
docker build --build-arg SKIP_MODEL_DOWNLOAD=true --build-arg ENABLE_ONNX=true -t blitz:latest .
```

### Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `SKIP_MODEL_DOWNLOAD` | `false` | Skip HuggingFace model downloads during build. Use when mounting pre-cached models at runtime via `-v ~/.cache/huggingface:/root/.cache/huggingface` |
| `ENABLE_ONNX` | `false` | Install ONNX Runtime and download FP16 Parakeet models for faster Tier 2 inference |

### Runtime

```bash
# With models baked into image (Path 1: Fresh Deploy)
docker run -d --gpus all --ipc=host --shm-size=8g \
  -p 8000:8000 --name blitz --env-file .env \
  blitz:latest

# With models mounted from host (Path 2: Pre-cached Deploy)
docker run -d --gpus all --ipc=host --shm-size=8g \
  -p 8000:8000 --name blitz --env-file .env \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  blitz:latest
```

---

## Dockerfile.onnx

Optimized for **faster inference** with ONNX Runtime. The Tier 2 model (Parakeet 0.6B) is converted to FP16 ONNX format:
- 50% smaller model footprint
- Faster inference on supported hardware
- Slightly lower accuracy (~90% vs 94% with NeMo bfloat16)

### When to Use

- High-throughput scenarios where speed > accuracy
- Memory-constrained environments
- When you need maximum requests/second

### Build & Run

```bash
docker build -f docker/Dockerfile.onnx -t blitz-onnx:latest .

docker run -d --gpus all --ipc=host --shm-size=8g \
  -p 8000:8000 --name blitz --env-file .env \
  -e TIER2_BACKEND=onnx \
  blitz-onnx:latest
```

---

## Dockerfile.batch

Optimized for **AWS Batch** workloads — processes audio files from S3 in bulk.

### Differences from Primary

- No HTTP server (processes files and exits)
- Reads audio from S3, writes transcripts back to S3
- Optimized for throughput over latency
- Designed for AWS Batch job definitions

### When to Use

- Batch transcription of large audio archives
- Asynchronous processing pipelines
- Event-driven transcription (S3 upload → Lambda → Batch)

### Build & Run

```bash
docker build -f docker/Dockerfile.batch -t blitz-batch:latest .

# Local testing
docker run --gpus all --ipc=host --shm-size=8g \
  -e INPUT_S3_URI=s3://my-bucket/audio/meeting.mp3 \
  -e OUTPUT_S3_URI=s3://my-bucket/transcripts/meeting.json \
  blitz-batch:latest

# Deploy with AWS Batch
./scripts/deploy_batch.sh --region us-east-1
```

See [infrastructure/batch/](../infrastructure/batch/) for the CloudFormation template.

---

## Important Runtime Flags

These apply to **all** Dockerfile variants:

| Flag | Required | Why |
|------|----------|-----|
| `--gpus all` | ✅ | GPU access for model inference |
| `--ipc=host` | ✅ | PyTorch shared memory for DataLoader workers |
| `--shm-size=8g` | ✅ | **Must be 8g+** — Diarization uses Python multiprocessing (NeMo MSDD). Using 2g causes `SemLock._rebuild FileNotFoundError` |
| `--ulimit memlock=-1` | Recommended | Prevents CUDA memory allocation limits |
| `--ulimit stack=67108864` | Recommended | Larger stack for deep model graphs |

---

## Image Sizes (Approximate)

| Image | Compressed | Uncompressed | Notes |
|-------|------------|--------------|-------|
| `blitz:latest` (with models) | ~15 GB | ~30 GB | All models baked in |
| `blitz:latest` (no models) | ~8 GB | ~18 GB | Built with `SKIP_MODEL_DOWNLOAD=true` |
| `blitz-onnx:latest` | ~12 GB | ~25 GB | Smaller Tier 2 model |
| `blitz-batch:latest` | ~15 GB | ~30 GB | Same as primary + batch tools |

---

## Multi-Stage Build (Advanced)

For production CI/CD pipelines, you can split the build into stages:

```bash
# Stage 1: Build dependencies (cached across builds)
docker build --target deps -t blitz-deps:latest .

# Stage 2: Add models (only rebuilds when models change)
docker build --target models -t blitz-models:latest .

# Stage 3: Add application code (fast, only code changes)
docker build -t blitz:latest .
```

This is not implemented yet but would be the recommended approach for CI/CD pipelines where you want fast iteration on code changes without rebuilding the entire image.
