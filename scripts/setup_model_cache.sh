#!/bin/bash
set -euo pipefail

# ============================================================================
# Blitz ASR — Model Cache Setup Script
# ============================================================================
# Downloads all required ML models from HuggingFace and uploads them to S3.
# Run this ONCE from a machine with HuggingFace access (local workstation,
# CI/CD, or an EC2 with internet connectivity).
#
# After running this, deploy with Path 2 (pre-cached) — see docs/DEPLOYMENT.md
#
# Usage:
#   ./scripts/setup_model_cache.sh --bucket my-blitz-models --region us-east-1
#
# Requirements:
#   - Python 3.8+ with pip
#   - AWS CLI configured with S3 write access
#   - ~15GB free disk space
#   - Internet access to huggingface.co
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
BUCKET=""
REGION="us-east-1"
CACHE_DIR="${HOME}/.cache/huggingface"
UPLOAD_TO_S3=true
KEEP_LOCAL=true
HF_TOKEN=""

# ============================================================================
# Functions
# ============================================================================

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Downloads Blitz ASR models from HuggingFace and optionally uploads to S3.

Required:
  --bucket BUCKET_NAME       S3 bucket for storing model cache

Optional:
  --region REGION            AWS region (default: us-east-1)
  --cache-dir DIR            Local cache directory (default: ~/.cache/huggingface)
  --hf-token TOKEN           HuggingFace token (for gated models, if needed)
  --no-upload                Download models locally only (skip S3 upload)
  --no-keep-local            Delete local cache after S3 upload
  --help                     Show this help message

Examples:
  # Download models and upload to S3
  $0 --bucket my-blitz-models

  # Download only (no S3 upload)
  $0 --bucket my-blitz-models --no-upload

  # Use a specific HuggingFace token
  $0 --bucket my-blitz-models --hf-token hf_xxxxx

EOF
    exit 0
}

log() {
    echo "[$(date +'%H:%M:%S')] $*"
}

error() {
    echo "[ERROR] $*" >&2
    exit 1
}

check_dependencies() {
    command -v python3 &>/dev/null || error "python3 is required"
    command -v pip3 &>/dev/null || command -v pip &>/dev/null || error "pip is required"
    if [[ "$UPLOAD_TO_S3" == "true" ]]; then
        command -v aws &>/dev/null || error "aws CLI is required for S3 upload"
    fi
}

# ============================================================================
# Parse Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --bucket) BUCKET="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --cache-dir) CACHE_DIR="$2"; shift 2 ;;
        --hf-token) HF_TOKEN="$2"; shift 2 ;;
        --no-upload) UPLOAD_TO_S3=false; shift ;;
        --no-keep-local) KEEP_LOCAL=false; shift ;;
        --help) usage ;;
        *) error "Unknown option: $1" ;;
    esac
done

[[ -z "$BUCKET" ]] && error "Missing required parameter: --bucket"

check_dependencies

# ============================================================================
# Step 1: Install Python Dependencies (in a temp venv)
# ============================================================================

log "============================================"
log "Blitz ASR — Model Cache Setup"
log "============================================"
log "Bucket:     s3://$BUCKET"
log "Region:     $REGION"
log "Cache Dir:  $CACHE_DIR"
log "Upload S3:  $UPLOAD_TO_S3"
log "============================================"
echo ""

log "📦 Installing download dependencies..."

# Create a temporary venv for downloading
VENV_DIR=$(mktemp -d)/blitz-download-venv
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --quiet --upgrade pip
pip install --quiet transformers torch nemo_toolkit[asr] speechbrain huggingface_hub

if [[ -n "$HF_TOKEN" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    log "🔑 Using provided HuggingFace token"
fi

# ============================================================================
# Step 2: Download Models
# ============================================================================

log ""
log "🧠 Downloading models to $CACHE_DIR..."
log "   This will download ~12GB of model files."
log ""

mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR"

# --- Tier 1: Parakeet CTC-110M (English fast mode) ---
log "📥 [1/5] Downloading nvidia/parakeet-tdt_ctc-110m (~450MB)..."
python3 -c "
from nemo.collections.asr.models import ASRModel
model = ASRModel.from_pretrained('nvidia/parakeet-tdt_ctc-110m')
print('  ✅ parakeet-tdt_ctc-110m cached')
" || log "  ⚠️ Failed to download parakeet-tdt_ctc-110m (may need GPU for full init)"

# --- Tier 2: Parakeet 0.6B (Multilingual) ---
log "📥 [2/5] Downloading nvidia/parakeet-tdt-0.6b-v3 (~2.5GB)..."
python3 -c "
from nemo.collections.asr.models import ASRModel
model = ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3')
print('  ✅ parakeet-tdt-0.6b-v3 cached')
" || log "  ⚠️ Failed to download parakeet-tdt-0.6b-v3 (may need GPU for full init)"

# --- Tier 3: Whisper large-v3-turbo ---
log "📥 [3/5] Downloading openai/whisper-large-v3-turbo (~3.1GB)..."
python3 -c "
from transformers import WhisperProcessor, WhisperForConditionalGeneration
WhisperProcessor.from_pretrained('openai/whisper-large-v3-turbo')
WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3-turbo')
print('  ✅ whisper-large-v3-turbo cached')
"

# --- Whisper tiny (language detection) ---
log "📥 [4/5] Downloading openai/whisper-tiny (~150MB)..."
python3 -c "
from transformers import WhisperProcessor, WhisperForConditionalGeneration
WhisperProcessor.from_pretrained('openai/whisper-tiny')
WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny')
print('  ✅ whisper-tiny cached')
"

# --- VoxLingua107 (language detection fallback) ---
log "📥 [5/5] Downloading speechbrain/lang-id-voxlingua107-ecapa (~100MB)..."
python3 -c "
from speechbrain.inference.classifiers import EncoderClassifier
EncoderClassifier.from_hparams(source='speechbrain/lang-id-voxlingua107-ecapa', savedir='$CACHE_DIR/voxlingua')
print('  ✅ voxlingua107 cached')
"

# Cleanup venv
deactivate
rm -rf "$(dirname $VENV_DIR)"

# ============================================================================
# Step 3: Package and Upload to S3
# ============================================================================

log ""
log "📊 Cache size:"
du -sh "$CACHE_DIR"

if [[ "$UPLOAD_TO_S3" == "true" ]]; then
    log ""
    log "📤 Packaging and uploading to s3://$BUCKET/..."
    
    # Create S3 bucket if it doesn't exist
    aws s3 mb "s3://$BUCKET" --region "$REGION" 2>/dev/null || true

    # Compress the cache
    ARCHIVE="/tmp/huggingface-cache.tar.gz"
    log "   Compressing model cache (this may take a few minutes)..."
    tar -czf "$ARCHIVE" -C "$(dirname $CACHE_DIR)" "$(basename $CACHE_DIR)"
    
    ARCHIVE_SIZE=$(du -sh "$ARCHIVE" | cut -f1)
    log "   Archive size: $ARCHIVE_SIZE"
    
    # Upload to S3
    log "   Uploading to S3 (this may take several minutes for ~10GB)..."
    aws s3 cp "$ARCHIVE" "s3://$BUCKET/huggingface-cache.tar.gz" --region "$REGION"
    
    # Cleanup archive
    rm -f "$ARCHIVE"
    
    log "   ✅ Model cache uploaded to s3://$BUCKET/huggingface-cache.tar.gz"
    
    # Also upload a restore script
    cat > /tmp/restore.sh << 'RESTORE'
#!/bin/bash
set -euo pipefail
BUCKET="BUCKET_PLACEHOLDER"
REGION="REGION_PLACEHOLDER"

echo "🔄 Restoring Blitz ASR model cache from S3..."
echo "   Bucket: s3://$BUCKET"
echo ""

# Download and extract model cache
echo "🧠 Downloading model cache (~12GB)..."
mkdir -p ~/.cache
aws s3 cp "s3://$BUCKET/huggingface-cache.tar.gz" /tmp/ --region "$REGION"
echo "   Extracting..."
tar -xzf /tmp/huggingface-cache.tar.gz -C ~/.cache/
rm -f /tmp/huggingface-cache.tar.gz

echo ""
echo "✅ Model cache restored to ~/.cache/huggingface/"
echo ""
echo "Next steps:"
echo "  1. Clone the blitz-oss repo"
echo "  2. Build: docker build --build-arg SKIP_MODEL_DOWNLOAD=true -t blitz:latest ."
echo "  3. Run:   docker run -d --gpus all --ipc=host --shm-size=8g -p 8000:8000 \\"
echo "              --name blitz --env-file .env \\"
echo "              -v ~/.cache/huggingface:/root/.cache/huggingface \\"
echo "              blitz:latest"
RESTORE

    sed -i.bak "s/BUCKET_PLACEHOLDER/$BUCKET/g" /tmp/restore.sh
    sed -i.bak "s/REGION_PLACEHOLDER/$REGION/g" /tmp/restore.sh
    rm -f /tmp/restore.sh.bak
    
    aws s3 cp /tmp/restore.sh "s3://$BUCKET/restore.sh" --region "$REGION"
    rm -f /tmp/restore.sh
    
    log "   ✅ Restore script uploaded to s3://$BUCKET/restore.sh"
fi

if [[ "$KEEP_LOCAL" == "false" ]]; then
    log "🧹 Cleaning up local cache..."
    rm -rf "$CACHE_DIR"
fi

# ============================================================================
# Summary
# ============================================================================

log ""
log "============================================"
log "✅ Model cache setup complete!"
log "============================================"
log ""
log "S3 contents:"
aws s3 ls "s3://$BUCKET/" --human-readable --region "$REGION" 2>/dev/null || true
log ""
log "To deploy on a new EC2 instance:"
log "  aws s3 cp s3://$BUCKET/restore.sh . && chmod +x restore.sh && ./restore.sh"
log ""
log "See docs/DEPLOYMENT.md for full deployment instructions."
