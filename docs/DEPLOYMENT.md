# Blitz ASR — Deployment Guide

This guide covers deploying Blitz ASR to your own AWS environment. Two deployment paths are supported depending on your network constraints.

---

## Deployment Paths

| Path | When to Use | Build Time | Model Download |
|------|-------------|------------|----------------|
| **Path 1: Fresh Deploy** | HuggingFace accessible, first-time setup | 30-40 min | During Docker build |
| **Path 2: Pre-cached (S3)** | Air-gapped, rate-limited, or repeated deploys | 15-20 min | From S3 (pre-downloaded) |

> **Recommendation**: Use Path 1 for first-time setup. If you encounter HuggingFace download failures (rate limits, timeouts, gated model access), switch to Path 2.

---

## Prerequisites

### AWS Resources

- **EC2 Instance**: `g6.xlarge` (NVIDIA L4, 24GB VRAM) — minimum recommended
- **AMI**: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.11 (Amazon Linux 2023)
- **EBS Volume**: 150 GB minimum (models are ~12GB + Docker layers ~20GB)
- **Security Group**: Port 8000 (API), Port 22 (SSH) open to your IP
- **Subnet**: Public subnet with Internet Gateway (for Path 1) or VPC endpoint for S3 (for Path 2)
- **IAM Role**: S3 read access (for Path 2 only)

### Software (Pre-installed on DLAMI)

- Docker (with NVIDIA Container Toolkit)
- `nvidia-smi` must show the GPU
- AWS CLI configured

### Verify GPU Access

```bash
# Should show NVIDIA L4 GPU
nvidia-smi

# Should show Docker with GPU support
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## Path 1: Fresh Deploy (HuggingFace Access)

Use this when your EC2 instance can reach `huggingface.co` without rate limits.

### Option A: CloudFormation (Fully Automated)

```bash
git clone <your-repo-url> blitz-oss
cd blitz-oss

# Deploy via script
./scripts/deploy_ec2_api.sh \
  --vpc-id vpc-XXXXXXXX \
  --subnet-id subnet-XXXXXXXX \
  --key-pair my-keypair

# Wait 30-40 minutes, then test
PUBLIC_IP=$(aws cloudformation describe-stacks \
  --stack-name blitz-ec2-api-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`PublicIP`].OutputValue' \
  --output text)

curl http://$PUBLIC_IP:8000/health
```

See [infrastructure/ec2-api/README.md](../infrastructure/ec2-api/README.md) for full CloudFormation options.

### Option B: Docker Compose (Manual EC2)

```bash
# SSH to your GPU instance
ssh -i my-key.pem ec2-user@<PUBLIC_IP>

# Clone repo
git clone <your-repo-url> blitz-oss
cd blitz-oss

# Create .env from example
cp .env.example .env
# Edit .env if needed (defaults work for most setups)

# Build and run (downloads models during build — 30-40 min first time)
docker compose up -d --build

# Watch logs until "Application startup complete"
docker compose logs -f
```

### Option C: Docker Run (Direct)

```bash
cd blitz-oss

# Build image
docker build -t blitz:latest .

# Run with GPU
docker run -d --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --shm-size=8g \
  -p 8000:8000 \
  --name blitz \
  --env-file .env \
  blitz:latest
```

---

## Path 2: Pre-cached Deploy (S3 Backup)

Use this when:
- HuggingFace downloads fail (rate limits, gated models, network restrictions)
- You want faster, repeatable deploys (skip 10+ min of model downloads)
- Air-gapped environments (no internet access from EC2)

### Step 1: One-time Model Cache Setup

Run this **once** from a machine with HuggingFace access (local workstation or an EC2 with internet):

```bash
# Use the provided script to download all models and upload to S3
./scripts/setup_model_cache.sh --bucket my-blitz-models --region us-east-1
```

This downloads ~12GB of models and uploads them to your S3 bucket. See [scripts/setup_model_cache.sh](../scripts/setup_model_cache.sh) for details.

**What gets cached:**
| Model | Size | Purpose |
|-------|------|---------|
| `nvidia/parakeet-tdt-0.6b-v3` | ~2.5 GB | Tier 2: Multilingual (24 EU languages) |
| `nvidia/parakeet-tdt_ctc-110m` | ~450 MB | Tier 1: English fast mode |
| `openai/whisper-large-v3-turbo` | ~3.1 GB | Tier 3: Catch-all (100+ languages) |
| `openai/whisper-tiny` | ~150 MB | Language detection (lite mode) |
| `speechbrain/lang-id-voxlingua107-ecapa` | ~100 MB | Fallback language detection |

### Step 2: Deploy with Pre-cached Models

```bash
# SSH to your GPU instance
ssh -i my-key.pem ec2-user@<PUBLIC_IP>

# Download and run the restore script
aws s3 cp s3://my-blitz-models/restore.sh .
chmod +x restore.sh
./restore.sh
```

**Or manually:**

```bash
# 1. Clone source code
git clone <your-repo-url> blitz-oss
cd blitz-oss
cp .env.example .env

# 2. Download model cache from S3
mkdir -p ~/.cache
aws s3 cp s3://my-blitz-models/huggingface-cache.tar.gz .
tar -xzf huggingface-cache.tar.gz -C ~/.cache/
rm huggingface-cache.tar.gz

# 3. Build Docker image (skips model downloads since we'll mount the cache)
docker build --build-arg SKIP_MODEL_DOWNLOAD=true -t blitz:latest .

# 4. Run with mounted model cache
docker run -d --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --shm-size=8g \
  -p 8000:8000 \
  --name blitz \
  --env-file .env \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  blitz:latest
```

### Step 3: Enable Speaker Diarization (Optional)

Diarization requires additional dependencies installed inside the running container:

```bash
# Wait for container startup (health check passes)
until curl -s http://localhost:8000/health | grep -q "healthy"; do
  echo "Waiting for blitz to start..."
  sleep 10
done

# Install diarization pipeline
docker exec blitz bash -c "
pip install prometheus-client &&
cd /tmp && git clone https://github.com/MahmoudAshraf97/whisper-diarization &&
pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git deepmultilingualpunctuation &&
sed -i 's/model = AutoModelForCTC.from_pretrained(model_path, dtype=dtype)/model = AutoModelForCTC.from_pretrained(model_path)/' /usr/local/lib/python3.12/dist-packages/ctc_forced_aligner/alignment_utils.py &&
sed -i 's/punct_model.predict(words_list, chunk_size=230)/punct_model.predict(words_list)/' /tmp/whisper-diarization/diarize_parallel.py &&
echo '✅ Diarization ready'
"
```

> **Tip**: After confirming diarization works, save the container state so you don't need to reinstall on restart:
> ```bash
> docker commit blitz blitz:full
> ```
> Then use `blitz:full` in future `docker run` commands.

---

## Verify Deployment

```bash
# Health check (all models should show loaded=true)
curl http://localhost:8000/health | jq .

# Test transcription
curl -X POST http://localhost:8000/transcribe \
  -F "file=@tests/fixtures/english.mp3" \
  -F "format=json" | jq .

# Test diarization (if installed)
curl -X POST http://localhost:8000/diarize \
  -F "file=@tests/fixtures/english.mp3" \
  -F "format=json" | jq .

# Prometheus metrics
curl http://localhost:8000/metrics
```

---

## Configuration

### Environment Variables (.env)

Copy `.env.example` and customize:

```bash
cp .env.example .env
```

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `TIER1_MODEL` | `nvidia/parakeet-tdt_ctc-110m` | English-only fast model |
| `TIER2_MODEL` | `nvidia/parakeet-tdt-0.6b-v3` | Multilingual model |
| `TIER3_MODEL` | `openai/whisper-large-v3-turbo` | Catch-all model |
| `TIER2_PRECISION` | `bfloat16` | Model precision (bfloat16 recommended) |
| `MAX_BATCH_SIZE` | `4` | Concurrent request batching |
| `LOCAL_ATTENTION_THRESHOLD_SEC` | `1440` | Switch to local attention for audio >24 min |
| `PRIMARY_LANGUAGE_DETECTOR` | `whisper` | Primary language detection model |

See [Dockerfile](../Dockerfile) for the full list of environment variables with documentation.

### Docker Run Options

| Flag | Purpose |
|------|---------|
| `--gpus all` | Enable GPU access |
| `--ipc=host` | Share host IPC namespace (required for PyTorch) |
| `--shm-size=8g` | ⚠️ **Must be 8g+** (diarization uses multiprocessing; 2g causes SemLock errors) |
| `-v ~/.cache/huggingface:/root/.cache/huggingface` | Mount pre-downloaded models (Path 2) |
| `-v ./blitz:/app/blitz` | Mount source for live code updates (dev only) |

---

## GPU Instance Sizing

| Instance | GPU | VRAM | Max Audio (single request) | Throughput | Cost (Spot) |
|----------|-----|------|---------------------------|------------|-------------|
| `g6.xlarge` | L4 | 24GB | ~24 min | 1.87 req/s | ~$0.42/hr |
| `g6.2xlarge` | L4 | 24GB | ~24 min | 1.87 req/s | ~$0.67/hr |
| `g5.xlarge` | A10G | 24GB | ~24 min | ~2 req/s | ~$0.42/hr |
| `p3.2xlarge` | V100 | 16GB | ~15 min | ~1.5 req/s | ~$0.92/hr |
| `g5.2xlarge` | A10G | 24GB | ~24 min | ~2 req/s | ~$0.60/hr |

**Recommended**: `g6.xlarge` (best price/performance for most workloads).

---

## Troubleshooting

### HuggingFace Download Failures

**Symptom**: Docker build hangs or fails at model download steps.

**Fix**: Switch to Path 2 (pre-cached deploy):
```bash
# Build without model downloads
docker build --build-arg SKIP_MODEL_DOWNLOAD=true -t blitz:latest .

# Mount models from host
docker run ... -v ~/.cache/huggingface:/root/.cache/huggingface blitz:latest
```

### GPU Not Detected

**Symptom**: `nvidia-smi` works on host but `docker exec blitz nvidia-smi` fails.

**Fix**: Ensure NVIDIA Container Toolkit is installed:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo yum install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Diarization 500 Error (SemLock / FileNotFoundError)

**Symptom**: `/diarize` returns 500 with `SemLock._rebuild FileNotFoundError`.

**Fix**: Container needs more shared memory. Recreate with `--shm-size=8g`:
```bash
docker stop blitz && docker rm blitz
docker run -d --gpus all --ipc=host --shm-size=8g ...
```

### Out of Memory (OOM)

**Symptom**: Container crashes or returns 500 on long audio files.

**Fix**: Reduce `LOCAL_ATTENTION_THRESHOLD_SEC` in `.env`:
```bash
# For 24GB GPU, reduce from 1440 to 900 if experiencing OOM
LOCAL_ATTENTION_THRESHOLD_SEC=900
```

### Port 8000 Not Accessible from Outside

**Checklist**:
1. Security Group has port 8000 open to your IP
2. Instance is in a **public subnet** with Internet Gateway
3. Instance has a **public IP** assigned (or Elastic IP)
4. Network ACL allows inbound TCP 8000

**Workaround** (SSH tunnel):
```bash
ssh -i my-key.pem -L 8000:localhost:8000 ec2-user@<PUBLIC_IP>
# Then use http://localhost:8000 from your local machine
```

---

## Backup & Restore

### Create Backup (Before Tearing Down)

```bash
BUCKET="my-blitz-backup"

# Source code
tar -czf blitz-source.tar.gz blitz-oss/
aws s3 cp blitz-source.tar.gz s3://$BUCKET/

# Model cache
tar -czf huggingface-cache.tar.gz -C ~/.cache huggingface/
aws s3 cp huggingface-cache.tar.gz s3://$BUCKET/

# Environment config
aws s3 cp /root/blitz-oss/.env s3://$BUCKET/dot-env-latest

# Verify
aws s3 ls s3://$BUCKET/ --human-readable
```

### Restore on New Instance

```bash
BUCKET="my-blitz-backup"

# Download and extract
aws s3 cp s3://$BUCKET/blitz-source.tar.gz . && tar -xzf blitz-source.tar.gz && rm blitz-source.tar.gz
aws s3 cp s3://$BUCKET/dot-env-latest /root/blitz-oss/.env
mkdir -p ~/.cache && aws s3 cp s3://$BUCKET/huggingface-cache.tar.gz . && tar -xzf huggingface-cache.tar.gz -C ~/.cache/ && rm huggingface-cache.tar.gz

# Build and run
cd /root/blitz-oss
docker build --build-arg SKIP_MODEL_DOWNLOAD=true -t blitz:latest .
docker run -d --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --shm-size=8g -p 8000:8000 --name blitz --env-file .env \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /root/blitz-oss/blitz:/app/blitz \
  blitz:latest
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  EC2 Instance (g6.xlarge)                                       │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Docker Container (blitz:latest)                         │    │
│  │                                                          │    │
│  │  FastAPI Server (:8000)                                  │    │
│  │    ├── POST /transcribe  → Tier Routing → GPU Inference  │    │
│  │    ├── POST /diarize     → whisper-diarization pipeline  │    │
│  │    ├── GET  /health      → Model status check            │    │
│  │    └── GET  /metrics     → Prometheus metrics            │    │
│  │                                                          │    │
│  │  Models (mounted or built-in):                           │    │
│  │    ├── Tier 1: Parakeet CTC-110M (English, fastest)      │    │
│  │    ├── Tier 2: Parakeet 0.6B (Multilingual)              │    │
│  │    ├── Tier 3: Whisper large-v3-turbo (100+ languages)   │    │
│  │    └── VoxLingua107 (language detection fallback)         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Volumes:                                                        │
│    ~/.cache/huggingface → /root/.cache/huggingface (models)     │
│    ./blitz             → /app/blitz (source, dev only)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Cost Estimates (us-east-1)

| Usage Pattern | Instance | Pricing | Monthly Cost |
|---------------|----------|---------|--------------|
| Dev (8h/day, weekdays) | g6.xlarge Spot | $0.42/hr | ~$72 |
| Dev (24/7) | g6.xlarge Spot | $0.42/hr | ~$302 |
| Prod (24/7) | g6.xlarge On-Demand | $1.05/hr | ~$756 |
| Prod (Reserved 1yr) | g6.xlarge | ~$0.68/hr | ~$490 |

**Cost per 1,000 minutes transcribed** (based on 1.87 req/s throughput): ~$0.19 (Spot)
