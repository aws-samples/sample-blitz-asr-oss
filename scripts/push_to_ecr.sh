#!/bin/bash
# =============================================================================
# Push Blitz ASR Image to Amazon ECR
# =============================================================================
#
# This script builds the Blitz Docker image and pushes it to Amazon ECR.
# Use this to store your image for reuse across g6.xlarge instances or
# for AWS Batch deployments.
#
# Usage:
#   ./scripts/push_to_ecr.sh <aws-account-id> <region> [image-tag]
#
# Examples:
#   # Push with default 'latest' tag
#   ./scripts/push_to_ecr.sh 123456789012 us-east-1
#
#   # Push with custom tag
#   ./scripts/push_to_ecr.sh 123456789012 us-east-1 v1.0.0
#
#   # Push ONNX variant
#   ENABLE_ONNX=true ./scripts/push_to_ecr.sh 123456789012 us-east-1 onnx-latest
#
# Prerequisites:
#   - AWS CLI configured with appropriate permissions
#   - Docker installed and running
#   - NVIDIA Container Toolkit (for local GPU builds)
#
# Required IAM Permissions:
#   - ecr:GetAuthorizationToken
#   - ecr:CreateRepository
#   - ecr:DescribeRepositories
#   - ecr:BatchCheckLayerAvailability
#   - ecr:CompleteLayerUpload
#   - ecr:InitiateLayerUpload
#   - ecr:PutImage
#   - ecr:UploadLayerPart
#
# =============================================================================

set -e

# Parse arguments
ACCOUNT_ID=${1:?Usage: $0 <aws-account-id> <region> [image-tag]}
REGION=${2:?Usage: $0 <aws-account-id> <region> [image-tag]}
IMAGE_TAG=${3:-latest}

# Configuration
REPO_NAME="blitz-asr"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}"
ENABLE_ONNX=${ENABLE_ONNX:-false}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==> Blitz ASR ECR Push Script${NC}"
echo "    Account: ${ACCOUNT_ID}"
echo "    Region:  ${REGION}"
echo "    Tag:     ${IMAGE_TAG}"
echo "    ONNX:    ${ENABLE_ONNX}"
echo ""

# Step 1: Authenticate with ECR
echo -e "${YELLOW}==> Step 1/5: Authenticating with ECR...${NC}"
aws ecr get-login-password --region "${REGION}" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
echo -e "${GREEN}    Authentication successful${NC}"

# Step 2: Create repository if it doesn't exist
echo -e "${YELLOW}==> Step 2/5: Ensuring ECR repository exists...${NC}"
if aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${REGION}" 2>/dev/null; then
  echo -e "${GREEN}    Repository '${REPO_NAME}' already exists${NC}"
else
  echo "    Creating repository '${REPO_NAME}'..."
  aws ecr create-repository \
    --repository-name "${REPO_NAME}" \
    --region "${REGION}" \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256
  echo -e "${GREEN}    Repository created${NC}"
fi

# Step 3: Build the image
echo -e "${YELLOW}==> Step 3/5: Building Docker image...${NC}"
echo "    This may take 15-20 minutes on first build (model downloads)"
if [ "${ENABLE_ONNX}" = "true" ]; then
  echo "    Building with ONNX support enabled"
  docker build --build-arg ENABLE_ONNX=true -t "${REPO_NAME}:${IMAGE_TAG}" .
else
  docker build -t "${REPO_NAME}:${IMAGE_TAG}" .
fi
echo -e "${GREEN}    Build complete${NC}"

# Step 4: Tag for ECR
echo -e "${YELLOW}==> Step 4/5: Tagging image for ECR...${NC}"
docker tag "${REPO_NAME}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
echo -e "${GREEN}    Tagged: ${ECR_URI}:${IMAGE_TAG}${NC}"

# Step 5: Push to ECR
echo -e "${YELLOW}==> Step 5/5: Pushing to ECR...${NC}"
echo "    This may take a few minutes depending on your connection"
docker push "${ECR_URI}:${IMAGE_TAG}"
echo -e "${GREEN}    Push complete${NC}"

# Summary
echo ""
echo -e "${GREEN}==============================================================================${NC}"
echo -e "${GREEN}SUCCESS: Image pushed to ECR${NC}"
echo -e "${GREEN}==============================================================================${NC}"
echo ""
echo "Image URI:"
echo "  ${ECR_URI}:${IMAGE_TAG}"
echo ""
echo "To pull this image:"
echo "  aws ecr get-login-password --region ${REGION} | \\"
echo "    docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
echo "  docker pull ${ECR_URI}:${IMAGE_TAG}"
echo ""
echo "To run on g6.xlarge:"
echo "  docker run --gpus all -p 8000:8000 ${ECR_URI}:${IMAGE_TAG}"
echo ""
echo "For AWS Batch deployment, use this image URI in your job definition."
