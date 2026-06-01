#!/bin/bash
# Deploy Blitz directly on SageMaker G6 instance
# Run this script ON the SageMaker instance after cloning the repo

set -e

echo "🚀 Deploying Blitz on SageMaker G6 Instance"
echo ""

# Step 1: Ensure we're in the repo directory
if [ ! -f "Dockerfile" ]; then
    echo "❌ Error: Dockerfile not found. Please run this from the blitz directory."
    echo ""
    echo "Clone the repo first:"
    echo "  git clone https://github.com/bmaguir/blitz.git"
    echo "  cd blitz"
    exit 1
fi

# Step 2: Check for GPU
echo "🔍 Checking for GPU..."
nvidia-smi || { echo "❌ Error: No GPU found. Ensure you're on a G6 instance."; exit 1; }

# Step 3: Build Docker image
echo ""
echo "🔨 Building Docker image (this will take 10-15 minutes)..."
docker build --no-cache -t blitz:latest .

# Step 4: Run container
echo ""
echo "✅ Build complete! Starting container..."
echo ""
docker run --gpus all -p 8000:8000 --name blitz blitz:latest

echo ""
echo "🎉 Server should be running on http://<instance-ip>:8000"
echo "📖 API docs: http://<instance-ip>:8000/docs"
