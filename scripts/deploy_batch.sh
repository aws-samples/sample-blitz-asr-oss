#!/bin/bash
# ============================================================================
# Blitz ASR - AWS Batch Deployment Script
# ============================================================================
# Deploys Blitz ASR batch processing infrastructure to AWS.
#
# Prerequisites:
#   - AWS CLI configured with appropriate credentials
#   - Docker installed and running
#   - ECR repository created (or will be created)
#
# Usage:
#   ./scripts/deploy_batch.sh [OPTIONS]
#
# Options:
#   --environment ENV    Environment name (dev, staging, prod). Default: dev
#   --region REGION      AWS region. Default: us-east-1
#   --vpc-id VPC_ID      VPC ID (required)
#   --subnet-ids IDS     Comma-separated subnet IDs (required)
#   --enable-onnx        Enable ONNX backend for FP16 inference
#   --skip-build         Skip Docker build (use existing image)
#   --dry-run            Show what would be deployed without executing
#   --profile PROFILE    AWS CLI profile to use
#   --help               Show this help message

set -euo pipefail

# ============================================================================
# Configuration Defaults
# ============================================================================
ENVIRONMENT="dev"
REGION="us-east-1"
VPC_ID=""
SUBNET_IDS=""
ENABLE_ONNX="false"
SKIP_BUILD="false"
DRY_RUN="false"
STACK_NAME=""
ECR_REPO_NAME="blitz-batch"
IMAGE_TAG="latest"
AWS_PROFILE_ARG=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_help() {
    head -30 "$0" | tail -25
    exit 0
}

# ============================================================================
# Parse Arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --vpc-id)
            VPC_ID="$2"
            shift 2
            ;;
        --subnet-ids)
            SUBNET_IDS="$2"
            shift 2
            ;;
        --enable-onnx)
            ENABLE_ONNX="true"
            shift
            ;;
        --skip-build)
            SKIP_BUILD="true"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --profile)
            AWS_PROFILE_ARG="--profile $2"
            export AWS_PROFILE="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Validation
# ============================================================================
STACK_NAME="blitz-batch-${ENVIRONMENT}"

log_info "Validating configuration..."

if [[ -z "$VPC_ID" ]]; then
    log_error "--vpc-id is required"
    log_info "Find your VPC ID: aws ec2 describe-vpcs --query 'Vpcs[*].[VpcId,Tags[?Key==\`Name\`].Value|[0]]' --output table"
    exit 1
fi

if [[ -z "$SUBNET_IDS" ]]; then
    log_error "--subnet-ids is required (comma-separated)"
    log_info "Find private subnets: aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC_ID --query 'Subnets[*].[SubnetId,AvailabilityZone,Tags[?Key==\`Name\`].Value|[0]]' --output table"
    exit 1
fi

# Verify AWS credentials
if ! aws sts get-caller-identity &>/dev/null; then
    log_error "AWS credentials not configured. Run 'aws configure' first."
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
ECR_IMAGE_URI="${ECR_REGISTRY}/${ECR_REPO_NAME}:${IMAGE_TAG}"

log_success "Configuration validated"
log_info "  Environment: $ENVIRONMENT"
log_info "  Region: $REGION"
log_info "  AWS Profile: ${AWS_PROFILE:-default}"
log_info "  VPC ID: $VPC_ID"
log_info "  Subnet IDs: $SUBNET_IDS"
log_info "  ECR Image: $ECR_IMAGE_URI"
log_info "  ONNX Backend: $ENABLE_ONNX"

if [[ "$DRY_RUN" == "true" ]]; then
    log_warn "DRY RUN - No changes will be made"
fi

# ============================================================================
# Step 1: Create ECR Repository (if needed)
# ============================================================================
log_info "Step 1: Checking ECR repository..."

if ! aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$REGION" &>/dev/null; then
    log_info "Creating ECR repository: $ECR_REPO_NAME"
    if [[ "$DRY_RUN" != "true" ]]; then
        aws ecr create-repository \
            --repository-name "$ECR_REPO_NAME" \
            --region "$REGION" \
            --image-scanning-configuration scanOnPush=true
        log_success "ECR repository created"
    fi
else
    log_success "ECR repository already exists"
fi

# ============================================================================
# Step 2: Build and Push Docker Image (using buildx for Apple Silicon support)
# ============================================================================
if [[ "$SKIP_BUILD" != "true" ]]; then
    log_info "Step 2: Building Docker image..."

    # Login to ECR
    if [[ "$DRY_RUN" != "true" ]]; then
        aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"
    fi

    # Ensure buildx builder exists for cross-platform builds (Apple Silicon → linux/amd64)
    if [[ "$DRY_RUN" != "true" ]]; then
        if ! docker buildx inspect blitz-builder &>/dev/null; then
            log_info "Creating buildx builder for cross-platform builds..."
            docker buildx create --name blitz-builder --use --bootstrap
        else
            docker buildx use blitz-builder
        fi
    fi

    # Build image with buildx for linux/amd64 (required for AWS Batch GPU instances)
    # Note: SKIP_MODEL_CACHE=true is required for Apple Silicon cross-compilation
    # Models will be downloaded at runtime on first job (~5-10 min extra startup)
    BUILD_ARGS="--build-arg SKIP_MODEL_CACHE=true"
    if [[ "$ENABLE_ONNX" == "true" ]]; then
        BUILD_ARGS="$BUILD_ARGS --build-arg ENABLE_ONNX=true"
    fi

    log_info "Building docker/Dockerfile.batch for linux/amd64 (this may take 5-10 minutes)..."
    log_info "Using docker buildx for Apple Silicon compatibility"
    log_warn "Model caching skipped (will download at first job runtime)"
    if [[ "$DRY_RUN" != "true" ]]; then
        docker buildx build \
            --platform linux/amd64 \
            -f docker/Dockerfile.batch \
            $BUILD_ARGS \
            -t "$ECR_IMAGE_URI" \
            --push \
            .
        log_success "Docker image built and pushed to ECR"
    fi
else
    log_warn "Skipping Docker build (--skip-build)"
fi

# ============================================================================
# Step 3: Deploy CloudFormation Stack
# ============================================================================
log_info "Step 3: Deploying CloudFormation stack..."

CFN_TEMPLATE="infrastructure/batch/cloudformation.yaml"

# Check if template exists
if [[ ! -f "$CFN_TEMPLATE" ]]; then
    log_error "CloudFormation template not found: $CFN_TEMPLATE"
    exit 1
fi

# Validate template
log_info "Validating CloudFormation template..."
if [[ "$DRY_RUN" != "true" ]]; then
    aws cloudformation validate-template \
        --template-body "file://$CFN_TEMPLATE" \
        --region "$REGION" > /dev/null
    log_success "Template is valid"
fi

# Determine Tier 2 settings based on ONNX flag
if [[ "$ENABLE_ONNX" == "true" ]]; then
    TIER2_BACKEND="onnx"
    TIER2_PRECISION="fp16"
else
    TIER2_BACKEND="nemo"
    TIER2_PRECISION="bfloat16"
fi

# Deploy or update stack
log_info "Deploying stack: $STACK_NAME"
if [[ "$DRY_RUN" != "true" ]]; then
    aws cloudformation deploy \
        --template-file "$CFN_TEMPLATE" \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --capabilities CAPABILITY_NAMED_IAM \
        --parameter-overrides \
            Environment="$ENVIRONMENT" \
            VpcId="$VPC_ID" \
            SubnetIds="$SUBNET_IDS" \
            ECRImageUri="$ECR_IMAGE_URI" \
            Tier2Backend="$TIER2_BACKEND" \
            Tier2Precision="$TIER2_PRECISION" \
        --tags \
            Project=blitz-asr \
            Environment="$ENVIRONMENT"

    log_success "CloudFormation stack deployed"
fi

# ============================================================================
# Step 4: Get Stack Outputs
# ============================================================================
log_info "Step 4: Retrieving stack outputs..."

if [[ "$DRY_RUN" != "true" ]]; then
    INPUT_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='InputBucketName'].OutputValue" \
        --output text)

    OUTPUT_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='OutputBucketName'].OutputValue" \
        --output text)

    LOG_GROUP=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='LogGroupName'].OutputValue" \
        --output text)
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================================"
echo -e "${GREEN}Blitz ASR Batch Deployment Complete!${NC}"
echo "============================================================================"
echo ""
echo "USAGE:"
echo "  Upload audio files to trigger transcription:"
echo "    aws s3 cp audio.mp3 s3://${INPUT_BUCKET:-blitz-input-$ENVIRONMENT-$ACCOUNT_ID}/"
echo ""
echo "  Download transcripts:"
echo "    aws s3 ls s3://${OUTPUT_BUCKET:-blitz-output-$ENVIRONMENT-$ACCOUNT_ID}/transcripts/"
echo ""
echo "MONITORING:"
echo "  View jobs:"
echo "    https://${REGION}.console.aws.amazon.com/batch/"
echo ""
echo "  View logs:"
echo "    aws logs tail ${LOG_GROUP:-/aws/batch/blitz-$ENVIRONMENT} --follow"
echo ""
echo "MANUAL JOB SUBMISSION:"
echo "  aws batch submit-job \\"
echo "    --job-name test-transcribe \\"
echo "    --job-queue blitz-transcription-${ENVIRONMENT} \\"
echo "    --job-definition blitz-transcribe-${ENVIRONMENT} \\"
echo "    --parameters input-bucket=${INPUT_BUCKET:-<INPUT_BUCKET>},input-key=<KEY>"
echo ""
echo "============================================================================"
