FROM nvcr.io/nvidia/pytorch:24.12-py3

# Build argument to skip model downloads (use when mounting pre-cached models)
# Build with: docker build --build-arg SKIP_MODEL_DOWNLOAD=true -t blitz:latest .
# Then run with: -v ~/.cache/huggingface:/root/.cache/huggingface
# See docs/DEPLOYMENT.md "Path 2: Pre-cached Deploy" for details
ARG SKIP_MODEL_DOWNLOAD=false

# Build argument to enable ONNX backend (optional)
# Build with: docker build --build-arg ENABLE_ONNX=true -t blitz-onnx .
ARG ENABLE_ONNX=false

# System deps (FFmpeg included)
RUN apt-get update && apt-get install -y ffmpeg sox libsndfile1 && rm -rf /var/lib/apt/lists/*

# Python deps
# Pin ruamel.yaml to avoid 'max_depth' AttributeError with speechbrain/hyperpyyaml
RUN pip install --no-cache-dir "ruamel.yaml>=0.17.0,<0.18.0"

RUN pip install --no-cache-dir \
    nemo_toolkit[asr]==2.4.0 \
    soundfile \
    fastapi uvicorn[standard] \
    python-multipart \
    pydantic \
    pydantic-settings \
    faster-whisper \
    "transformers[torch]" \
    sentencepiece \
    torchmetrics \
    git+https://github.com/speechbrain/speechbrain.git@develop

# Disable Flash Attention for better compatibility across GPU types
RUN pip uninstall -y flash-attn flash_attn || true
ENV HF_USE_FLASH_ATTENTION=0

# Remove image/vision-related extras that break in this PyTorch build
RUN pip uninstall -y torchvision || true

# Upgrade cuda-python so NeMo can use CUDA graphs with conditional nodes
# (avoids "Found cuda-python 12.3.0rc4... at least version 12.3.0 is needed" warning)
RUN /usr/bin/python3 -m pip install --no-cache-dir "cuda-python>=12.3.0"

# Fix torchaudio: install version compatible with container's CUDA (12.6)
# Must be done BEFORE model downloads (NeMo imports torchaudio during model loading)
RUN pip uninstall -y torchaudio && pip install --no-cache-dir torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# ============================================================================
# 3-Tier Model Routing Configuration
# ============================================================================

# Tier 1: Fastest (CTC-110M) - default: English only
ENV TIER1_MODEL=nvidia/parakeet-tdt_ctc-110m
ENV TIER1_PRECISION=fp16
ENV TIER1_LANGUAGES=en

# Tier 2: Fast Multilingual (Parakeet 0.6B) - default: 24 EU languages
ENV TIER2_MODEL=nvidia/parakeet-tdt-0.6b-v3
ENV TIER2_LANGUAGES=bg,hr,cs,da,nl,et,fi,fr,de,el,hu,it,lv,lt,mt,pl,pt,ro,sk,sl,es,sv,ru,uk

# Tier 3: Catch-all (Whisper) - handles all other languages
ENV TIER3_MODEL=openai/whisper-large-v3-turbo

# ============================================================================
# Efficiency Thresholds for Long-Form Audio
# ============================================================================

# Audio shorter than this uses direct array processing (fastest, lowest memory)
# Audio longer than this uses temp file processing (memory-efficient)
# Recommended: 300s (5 minutes) - covers 80%+ of production use cases
ENV SHORT_AUDIO_THRESHOLD_SEC=300

# Audio longer than this switches to local attention mode (prevents OOM)
# Recommended values:
#   - L4 24GB: 1440 (24 minutes) ✅ Default
#   - A10G 24GB: 1440 (24 minutes)
#   - A100 80GB: 3600 (60 minutes)
ENV LOCAL_ATTENTION_THRESHOLD_SEC=1440

# ============================================================================
# Micro-Batching Configuration (8-9% Throughput Improvement, 4x Concurrency Speedup)
# ============================================================================

# Wait up to this many milliseconds to gather a batch of requests
# Lower = lower latency, higher = better GPU utilization
# Recommended: 15ms (adds 0-15ms latency, 2x+ throughput)
ENV BATCH_TIMEOUT_MS=15.0

# Maximum number of requests to batch together
# Higher = better GPU utilization but more memory
# Recommended: 4 for optimal throughput (based on L4 benchmarks)
ENV MAX_BATCH_SIZE=4

# Cleanup after every N batches (lower = more stable, higher = faster)
# Default: 1 (cleanup after every batch for maximum stability)
ENV BATCH_CLEANUP_FREQUENCY=1

# ============================================================================
# Language Detection Model Configuration
# ============================================================================

# Model to use for Whisper language detection
# Options: "tiny" (saves ~8GB memory) or "large-v3-turbo" (better accuracy, already loaded)
# Default: "large-v3-turbo" - model already loaded, 6.5% faster detection, better accuracy
ENV LANGUAGE_DETECTION_MODEL=large-v3-turbo

# Primary language detector: "whisper" (GPU, more accurate) or "voxlingua" (CPU, frees GPU)
# Default: "whisper" - GPU-based, faster (~150ms), better accuracy on English
ENV PRIMARY_LANGUAGE_DETECTOR=whisper

# Secondary language detector for fallback when primary confidence is low
# Default: "voxlingua" - CPU-based fallback for edge cases
ENV SECONDARY_LANGUAGE_DETECTOR=voxlingua

# Confidence thresholds per model
# VoxLingua threshold: Higher because VoxLingua tends to produce lower peak confidence
# Whisper threshold: Standard threshold for Whisper confidence scores
ENV VOXLINGUA_CONFIDENCE_THRESHOLD=0.90
ENV WHISPER_CONFIDENCE_THRESHOLD=0.70

# Legacy threshold (deprecated, use per-model thresholds above)
ENV MIN_LANGUAGE_CONFIDENCE=0.70

# ============================================================================
# PyTorch CUDA Memory Configuration
# ============================================================================

# Reduce memory fragmentation by allowing expandable segments
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# Tier 2 Backend Configuration (Parakeet 0.6B)
# ============================================================================
# Backend: "nemo" (default, PyTorch/NeMo) or "onnx" (ONNX Runtime with FP16)
# - NeMo: Full-featured, supports local attention for long audio
# - ONNX: 50% smaller model, faster inference but lower accuracy (~90%)
ENV TIER2_BACKEND=nemo

# Precision: "fp32", "bfloat16" (RECOMMENDED DEFAULT), "fp16" (for ONNX)
#
# BENCHMARK RESULTS (English, 59m audio):
#   - bfloat16: 94.46% accuracy, 125x real-time (RECOMMENDED - best balance)
#   - fp32:     94.34% accuracy, 100x real-time (maximum precision)
#   - fp16:     89.74% accuracy, 167x real-time (maximum speed, ONNX only)
#
# See docs/benchmark-results.md for detailed analysis
ENV TIER2_PRECISION=bfloat16

# Path to ONNX models (only used when TIER2_BACKEND=onnx)
ENV TIER2_ONNX_MODEL_PATH=/app/models/parakeet

# Auto-download ONNX models from HuggingFace if missing
ENV TIER2_AUTO_DOWNLOAD=true

# ============================================================================
# Precision Configuration
# ============================================================================
# Both Parakeet and Whisper use FP32 for stability
# Memory usage: ~19GB total (both models loaded)
# Timestamps: DISABLED for stability (NeMo 2.0.0 compatibility issues)

# ============================================================================
# GPU Memory Management
# ============================================================================
# Aggressive cleanup strategy implemented in blitz/batch_processing.py:
# - Explicit tensor deletion after each batch
# - CUDA synchronization + cache clearing
# - Python garbage collection
# - Applied at 6 strategic locations
#
# Expected behavior:
# - Baseline: ~19GB VRAM (both models loaded)
# - Peak: ~22GB during batch processing
# - Return to baseline after batch completion
#
# This prevents 3GB+ memory accumulation on g6.xlarge (24GB VRAM)
# See commit d9271da for implementation details

# ============================================================================
# Recommended Deployment Configuration
# ============================================================================
# For NVIDIA L4 GPU (g6.xlarge, 24GB VRAM):
# - Optimal concurrency: 10-20 concurrent requests
# - Expected throughput: 1.87 req/s
# - Audio capacity: 0.75 min/s
# - Cost (Spot): $0.19 per 1,000 minutes
#
# Load balancer/proxy settings:
# - Connection timeout: 60s (for long audio)
# - Max concurrent: 20 (optimal throughput)
# - Health check: GET /health every 30s

# ============================================================================
# Model Pre-download (conditional)
# ============================================================================
# When SKIP_MODEL_DOWNLOAD=false (default): models are downloaded during build
# When SKIP_MODEL_DOWNLOAD=true: skip downloads, mount pre-cached models at runtime
#   via -v ~/.cache/huggingface:/root/.cache/huggingface

# Cache Parakeet 0.6B model during build (multilingual, 25 European languages)
RUN if [ "$SKIP_MODEL_DOWNLOAD" != "true" ]; then \
    python3 -c "from nemo.collections.asr.models import ASRModel; ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3')"; \
    fi

# Cache Parakeet CTC-110M model during build (English fast mode, 51% faster)
RUN if [ "$SKIP_MODEL_DOWNLOAD" != "true" ]; then \
    python3 -c "from nemo.collections.asr.models import ASRModel; ASRModel.from_pretrained('nvidia/parakeet-tdt_ctc-110m')"; \
    fi

# Cache Whisper tiny (processor + model) during build
# Used for language detection if LANGUAGE_DETECTION_MODEL=tiny
RUN if [ "$SKIP_MODEL_DOWNLOAD" != "true" ]; then \
    python3 -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; \
    WhisperProcessor.from_pretrained('openai/whisper-tiny'); \
    WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny')"; \
    fi

# Cache Whisper large-v3-turbo (processor + model) during build
# Used for transcription (always) and language detection (if LANGUAGE_DETECTION_MODEL=large-v3-turbo)
RUN if [ "$SKIP_MODEL_DOWNLOAD" != "true" ]; then \
    python3 -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; \
    WhisperProcessor.from_pretrained('openai/whisper-large-v3-turbo'); \
    WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3-turbo')"; \
    fi

# Cache VoxLingua107 model during build (used for fallback language detection)
# 107 languages, trained on VoxPopuli + CommonVoice
RUN if [ "$SKIP_MODEL_DOWNLOAD" != "true" ]; then \
    python3 -c "from speechbrain.inference.classifiers import EncoderClassifier; \
    EncoderClassifier.from_hparams(source='speechbrain/lang-id-voxlingua107-ecapa', savedir='tmp/voxlingua')"; \
    fi

# ============================================================================
# Optional: ONNX Runtime for FP16 Parakeet Models
# ============================================================================
# Build with: docker build --build-arg ENABLE_ONNX=true -t blitz-onnx .
# Provides 50% smaller model size and potentially faster inference

# Install ONNX dependencies if enabled
RUN if [ "$ENABLE_ONNX" = "true" ]; then \
    pip install --no-cache-dir onnx-asr onnxruntime-gpu huggingface-hub && \
    echo "✅ ONNX Runtime installed"; \
    fi

# Download FP16 ONNX models if enabled
RUN if [ "$ENABLE_ONNX" = "true" ]; then \
    mkdir -p /app/models/parakeet && \
    python3 -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('grikdotnet/parakeet-tdt-0.6b-fp16', 'encoder-model.fp16.onnx', local_dir='/app/models/parakeet'); \
hf_hub_download('grikdotnet/parakeet-tdt-0.6b-fp16', 'decoder_joint-model.fp16.onnx', local_dir='/app/models/parakeet'); \
hf_hub_download('istupakov/parakeet-tdt-0.6b-v3-onnx', 'config.json', local_dir='/app/models/parakeet'); \
hf_hub_download('istupakov/parakeet-tdt-0.6b-v3-onnx', 'vocab.txt', local_dir='/app/models/parakeet'); \
hf_hub_download('istupakov/parakeet-tdt-0.6b-v3-onnx', 'nemo128.onnx', local_dir='/app/models/parakeet')" && \
    echo "✅ FP16 ONNX models downloaded"; \
    fi

WORKDIR /app
# Copy modular architecture (includes server.py and banner.py)
COPY blitz/ /app/blitz/

# Copy utility scripts
COPY scripts/ /app/scripts/

# Security: Run as non-root user
RUN groupadd -r blitz && useradd -r -g blitz -d /app -s /sbin/nologin blitz \
    && chown -R blitz:blitz /app
USER blitz

# Liveness check (CKV_DOCKER_2): probe the API health endpoint. Uses Python's
# stdlib to avoid depending on curl being present in the image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python3", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"]

# Use integrated server with modular architecture
CMD ["uvicorn", "blitz.server:app", "--host", "0.0.0.0", "--port", "8000"]
