# SageMaker G6 Instance Deployment Guide

## Prerequisites
- SageMaker G6 instance with GPU (NVIDIA L4, 24GB)
- Git installed
- Docker installed
- AWS CLI configured

## Deployment Steps

### 1. Clone Repository on SageMaker Instance

```bash
# SSH into your SageMaker G6 instance
cd ~
git clone https://github.com/<your-username>/blitz.git
cd blitz
```

### 2. Verify Directory Structure

```bash
# Should show blitz folder
ls -la
# Expected output includes:
# drwxr-xr-x  blitz/
# -rw-r--r--  Dockerfile
# drwxr-xr-x  docker/
```

### 3. Build Docker Image

```bash
# Clean build (recommended for first deployment)
docker build --no-cache -t blitz:latest .

# This will take 10-15 minutes to:
# - Download NVIDIA PyTorch base image
# - Install dependencies
# - Pre-download models (Parakeet, Whisper, VoxLingua)
```

### 4. Run Container

```bash
# Run with GPU support
docker run --gpus all -p 8000:8000 --name blitz blitz:latest

# Or run in detached mode
docker run -d --gpus all -p 8000:8000 --name blitz blitz:latest

# View logs
docker logs -f blitz
```

### 5. Test the Service

```bash
# Health check
curl http://localhost:8000/health

# Test transcription (replace with your audio file)
curl -X POST http://localhost:8000/transcribe \
  -F "file=@test_audio.wav" \
  -F "output_format=json"
```

## Configuration for G6 Instance (24GB GPU)

The Dockerfile is pre-configured for G6.xlarge (24GB):

```dockerfile
# Optimized for 24GB GPU memory
ENV LOCAL_ATTENTION_THRESHOLD_SEC=1440  # 24 minutes max
ENV SHORT_AUDIO_THRESHOLD_SEC=600       # 10 minutes threshold
ENV MAX_BATCH_SIZE=2                    # Conservative for long audio
ENV BATCH_TIMEOUT_MS=15.0               # Low latency
```

## Troubleshooting

### Issue: ModuleNotFoundError for blitz
**Cause**: Directory named incorrectly (blitz vs blitz)
**Fix**: Ensure directory is `blitz` with underscore

### Issue: Docker build fails at COPY
**Cause**: Not in correct directory or directory missing
**Fix**:
```bash
cd ~/blitz  # Git repo directory
ls blitz/   # Verify package exists
docker build .     # Build from repo root
```

### Issue: GPU not detected
**Check GPU**:
```bash
nvidia-smi  # Should show NVIDIA L4 GPU
docker run --gpus all nvidia/cuda:12.0-base nvidia-smi
```

## Performance Expectations on G6.xlarge

- **Real-time Factor**: 0.02-0.05 (20-50x faster than real-time)
- **Throughput**: 1200-3000 minutes/hour (with micro-batching)
- **Memory Usage**: ~19GB for both models loaded
- **Cost**: ~$0.29-0.51 per 1000 minutes (depending on pricing tier)

## Accessing from External IP

If accessing from outside the instance:

1. **Configure Security Group**: Allow inbound TCP on port 8000
2. **Get Public IP**: `curl http://checkip.amazonaws.com`
3. **Access API**: `http://<public-ip>:8000/docs`

## Stopping the Container

```bash
# Stop container
docker stop blitz

# Remove container
docker rm blitz

# Remove image
docker rmi blitz:latest
```

## Updates and Redeployment

```bash
# Pull latest changes
cd ~/blitz
git pull

# Rebuild and restart
docker stop blitz
docker rm blitz
docker build --no-cache -t blitz:latest .
docker run -d --gpus all -p 8000:8000 --name blitz blitz:latest
```
