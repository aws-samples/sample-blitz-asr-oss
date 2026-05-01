#!/bin/bash
# Deploy Blitz to SageMaker via ECR

set -e

# Configuration
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="blitz"
IMAGE_TAG="latest"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

echo "🔧 AWS Account: ${AWS_ACCOUNT_ID}"
echo "🔧 ECR Repository: ${ECR_URI}"
echo ""

# Step 1: Create ECR repository if it doesn't exist
echo "📦 Creating ECR repository (if needed)..."
aws ecr describe-repositories --repository-names ${ECR_REPO} --region ${AWS_REGION} 2>/dev/null || \
    aws ecr create-repository --repository-name ${ECR_REPO} --region ${AWS_REGION}

# Step 2: Login to ECR
echo ""
echo "🔐 Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_URI}

# Step 3: Build image locally
echo ""
echo "🔨 Building Docker image locally..."
docker build -t blitz:${IMAGE_TAG} .

# Step 4: Tag for ECR
echo ""
echo "🏷️  Tagging image for ECR..."
docker tag blitz:${IMAGE_TAG} ${ECR_URI}:${IMAGE_TAG}

# Step 5: Push to ECR
echo ""
echo "⬆️  Pushing image to ECR..."
docker push ${ECR_URI}:${IMAGE_TAG}

echo ""
echo "✅ Image pushed to ECR!"
echo ""
echo "📋 To run on SageMaker G4 instance:"
echo ""
echo "   # Login to ECR on SageMaker instance"
echo "   aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_URI}"
echo ""
echo "   # Pull and run"
echo "   docker pull ${ECR_URI}:${IMAGE_TAG}"
echo "   docker run --gpus all -p 8000:8000 ${ECR_URI}:${IMAGE_TAG}"
echo ""
