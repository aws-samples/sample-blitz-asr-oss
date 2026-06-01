# Blitz ASR - AWS Batch Processing

Event-driven audio transcription using AWS Batch, triggered by S3 uploads.

## Overview

Blitz Batch provides high-throughput, serverless transcription by:
- Automatically processing audio files uploaded to S3
- Leveraging GPU instances (g5.xlarge/g6.xlarge) for fast inference
- Supporting 100+ languages with intelligent Parakeet/Whisper routing
- Producing JSON output with timestamps, language detection, and cost metrics

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   S3 Input  │────▶│ EventBridge │────▶│  AWS Batch  │
│   Bucket    │     │    Rule     │     │     Job     │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │ Blitz ASR   │
                                        │ Container   │
                                        │ (GPU: L4)   │
                                        └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  S3 Output  │
                                        │   Bucket    │
                                        └─────────────┘
```

## Quick Start

### Prerequisites

- AWS CLI configured with appropriate credentials
- Docker installed
- VPC with private subnets (for Batch compute)

### Deploy

```bash
# Find your VPC and subnets
aws ec2 describe-vpcs --query 'Vpcs[*].[VpcId,Tags[?Key==`Name`].Value|[0]]' --output table
aws ec2 describe-subnets --filters Name=vpc-id,Values=vpc-xxx --query 'Subnets[*].[SubnetId,AvailabilityZone]' --output table

# Deploy (first time takes ~15 min for Docker build)
./scripts/deploy_batch.sh \
    --environment dev \
    --region us-east-1 \
    --vpc-id vpc-xxxxxxxxx \
    --subnet-ids subnet-aaa,subnet-bbb
```

### Use

```bash
# Upload audio to trigger transcription
aws s3 cp interview.mp3 s3://blitz-input-dev-123456789012/

# Monitor job
aws logs tail /aws/batch/blitz-dev --follow

# Download transcript
aws s3 cp s3://blitz-output-dev-123456789012/transcripts/interview.json .
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OUTPUT_BUCKET_NAME` | (required) | S3 bucket for results |
| `OUTPUT_PREFIX` | `transcripts/` | S3 key prefix for outputs |
| `TIER1_LANGUAGES` | `en` | Languages routed to Tier 1 (CTC-110M) |
| `TIER2_BACKEND` | `nemo` | `nemo` (PyTorch) or `onnx` (FP16) |
| `TIER2_PRECISION` | `bfloat16` | `fp32`, `bfloat16`, or `fp16` |
| `SKIP_SILENCE` | `true` | Skip VAD-detected silent files |
| `INCLUDE_TIMESTAMPS` | `true` | Include segment timestamps |
| `INCLUDE_WORD_TIMESTAMPS` | `false` | Include word-level timestamps |
| `INCLUDE_METRICS` | `true` | Include processing metrics |

### Model Selection

**NeMo Backend (default):**
- Best accuracy (94%+ on English)
- Supports BFloat16 for optimal speed/accuracy
- Recommended for production

**ONNX Backend:**
- 50% smaller model
- 30% faster inference
- Lower accuracy (~90%)
- Good for high-volume, cost-sensitive workloads

Deploy with ONNX:
```bash
./scripts/deploy_batch.sh --enable-onnx ...
```

## Output Format

Transcripts are saved as JSON:

```json
{
  "input": {
    "bucket": "blitz-input-dev-123456789012",
    "key": "interview.mp3",
    "filename": "interview.mp3"
  },
  "transcription": {
    "text": "Hello, welcome to our interview...",
    "language": {
      "code": "en",
      "confidence": 0.98,
      "engine": "parakeet"
    },
    "timestamps": {
      "segment": [
        {"start": 0.0, "end": 2.5, "text": "Hello,"},
        {"start": 2.5, "end": 5.0, "text": "welcome to our interview..."}
      ]
    }
  },
  "audio": {
    "duration_sec": 1845.2,
    "sample_rate": 16000
  },
  "vad": {
    "is_silent": false,
    "speech_ratio": 0.92
  },
  "metrics": {
    "total_time_sec": 15.3,
    "rtf": 0.0083,
    "lang_detect_ms": 150,
    "transcribe_sec": 14.5,
    "cost_per_1000_min": {
      "on_demand": 0.22,
      "spot": 0.14
    }
  },
  "status": "success",
  "timestamp": "2025-01-03T12:00:00Z"
}
```

## Language Routing

Blitz automatically routes audio to the optimal model:

**Parakeet (25 languages):**
Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish, Russian, Ukrainian

**Whisper (75+ languages):**
Chinese, Japanese, Korean, Arabic, Hindi, Thai, Vietnamese, and many more.

Language detection uses a cascade:
1. Primary: Whisper (GPU, high accuracy)
2. Fallback: VoxLingua107 (CPU, 107 languages)

## Supported Audio Formats

- MP3, WAV, M4A, FLAC, OGG, WebM
- Automatically converted to 16kHz mono via FFmpeg
- Maximum recommended: 4 hours per file

## Scaling

### Instance Types

| Type | GPU | VRAM | Cost (Spot) | RTF |
|------|-----|------|-------------|-----|
| g5.xlarge | A10G | 24GB | ~$0.50/hr | 0.008 |
| g6.xlarge | L4 | 24GB | ~$0.35/hr | 0.01 |

### Throughput

Single g6.xlarge:
- ~100x real-time transcription
- ~6,000 minutes of audio per hour
- ~$0.006 per minute of audio (Spot)

### Cost Optimization

1. **Use Spot instances** (60% savings)
2. **Enable silence detection** (skip empty audio)
3. **Use ONNX backend** for high volume (faster, lower accuracy)

## Monitoring

### CloudWatch Logs

```bash
# Tail logs
aws logs tail /aws/batch/blitz-dev --follow

# Search for errors
aws logs filter-log-events \
    --log-group-name /aws/batch/blitz-dev \
    --filter-pattern "ERROR"
```

### Batch Console

View job status at:
https://console.aws.amazon.com/batch/

### Metrics

Jobs output metrics to CloudWatch:
- `BlitzBatch/Duration` - Processing time
- `BlitzBatch/AudioDuration` - Input audio length
- `BlitzBatch/RTF` - Real-time factor

## Troubleshooting

### Job Failed: OOM

Increase memory in CloudFormation:
```yaml
ResourceRequirements:
  - Type: MEMORY
    Value: '24576'  # 24GB
```

### Job Stuck in RUNNABLE

Check:
1. Instance type availability in region
2. Subnet has internet access (NAT gateway)
3. Compute environment max vCPUs

### No Jobs Triggering

Verify:
1. EventBridge rule is enabled
2. S3 bucket has EventBridge notifications enabled
3. File extension is supported (.mp3, .wav, etc.)

## Manual Job Submission

Test without S3 trigger:

```bash
aws batch submit-job \
    --job-name test-transcribe \
    --job-queue blitz-transcription-dev \
    --job-definition blitz-transcribe-dev \
    --parameters input-bucket=blitz-input-dev-123456789012,input-key=test.mp3
```

## Local Testing

Test the batch job locally:

```bash
# Build image
docker build -f docker/Dockerfile.batch -t blitz-batch .

# Run with S3 access
docker run --gpus all \
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    -e OUTPUT_BUCKET_NAME=my-output-bucket \
    blitz-batch \
    --input-bucket my-input-bucket \
    --input-key test.mp3
```

## Cleanup

Remove all resources:

```bash
# Empty S3 buckets first
aws s3 rm s3://blitz-input-dev-123456789012 --recursive
aws s3 rm s3://blitz-output-dev-123456789012 --recursive

# Delete stack
aws cloudformation delete-stack --stack-name blitz-batch-dev

# Delete ECR repository
aws ecr delete-repository --repository-name blitz-batch --force
```
