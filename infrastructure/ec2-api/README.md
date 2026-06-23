# Blitz ASR - EC2 API Infrastructure

CloudFormation template for deploying Blitz ASR as a GPU-accelerated HTTP API on EC2.

## Architecture

```
Internet → Elastic IP → EC2 (g6.xlarge) → Docker (Blitz API) → NVIDIA GPU
                           ↓
                     CloudWatch Logs
```

- **Instance**: GPU-enabled EC2 (g6.xlarge with NVIDIA L4)
- **AMI**: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.11 (Amazon Linux 2023)
- **Docker**: FastAPI server with GPU-accelerated models
- **Monitoring**: CloudWatch Logs + Prometheus metrics
- **Security**: IMDSv2, encrypted EBS, security group, optional API keys

## Quick Start

### Prerequisites

1. **AWS Account** with permissions for EC2, VPC, IAM, CloudFormation
2. **AWS CLI** configured: `aws configure`
3. **VPC** with public subnet (internet gateway attached)
4. **EC2 Key Pair** for SSH access
5. **Python 3** (for API key generation)

### Deploy

```bash
# Basic deployment (dev environment)
./scripts/deploy_ec2_api.sh \
  --vpc-id vpc-0123456789abcdef0 \
  --subnet-id subnet-0123456789abcdef0 \
  --key-pair my-keypair

# Production deployment with authentication
./scripts/deploy_ec2_api.sh \
  --environment prod \
  --vpc-id vpc-0123456789abcdef0 \
  --subnet-id subnet-0123456789abcdef0 \
  --key-pair my-keypair \
  --enable-auth \
  --api-keys '{"client1":["blitz_abc123"],"client2":["blitz_def456"]}' \
  --allowed-ssh 10.0.0.0/8 \
  --allowed-api 10.0.0.0/8

# Dev with auto-shutdown (saves costs)
./scripts/deploy_ec2_api.sh \
  --vpc-id vpc-0123456789abcdef0 \
  --subnet-id subnet-0123456789abcdef0 \
  --key-pair my-keypair \
  --auto-shutdown
```

### Manual Deployment (AWS Console)

1. Open [CloudFormation Console](https://console.aws.amazon.com/cloudformation)
2. Click **Create Stack** → **With new resources**
3. Upload `infrastructure/ec2-api/cloudformation.yaml`
4. Fill in parameters:
   - **VpcId**: Select your VPC
   - **SubnetId**: Select public subnet
   - **KeyPairName**: Select your key pair
   - Configure other parameters as needed
5. Click **Next** → **Next** → **Create Stack**
6. Wait 30-40 minutes for instance to initialize

## Parameters

### Required

| Parameter | Description | Example |
|-----------|-------------|---------|
| `VpcId` | VPC for the instance | `vpc-abc123` |
| `SubnetId` | Public subnet with IGW | `subnet-def456` |
| `KeyPairName` | EC2 key pair for SSH | `my-keypair` |

### Optional

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Environment` | `dev` | Environment name (`dev`, `staging`, `prod`) |
| `InstanceType` | `g6.xlarge` | GPU instance type (L4, A10G) |
| `EBSVolumeSize` | `150` | Root volume size in GB |
| `AllowedSSHCIDR` | `0.0.0.0/0` | CIDR for SSH access |
| `AllowedAPICIDR` | `0.0.0.0/0` | CIDR for API access |
| `EnableAuth` | `false` | Enable API key authentication |
| `ApiKeys` | `` | JSON map of API keys |
| `RateLimitRPM` | `60` | Requests per minute limit |
| `Tier2Backend` | `nemo` | Parakeet backend (`nemo`, `onnx`) |
| `Tier2Precision` | `bfloat16` | Model precision (`fp32`, `bfloat16`, `fp16`) |
| `EnableAutoShutdown` | `No` | Auto-shutdown at 11 PM (dev/staging) |
| `BlitzDockerImage` | `` | Pre-built image URI (skips build) |

## Post-Deployment

### Check Instance Status

```bash
# Get stack outputs
aws cloudformation describe-stacks \
  --stack-name blitz-ec2-api-dev \
  --query 'Stacks[0].Outputs' \
  --output table

# SSH to instance
ssh -i my-keypair.pem ec2-user@<PUBLIC_IP>

# Check Docker container
docker ps
docker logs -f blitz

# View setup logs
tail -f /var/log/blitz-setup.log
```

### Test API

```bash
# Health check
curl http://<PUBLIC_IP>:8000/health

# Test transcription
curl -X POST http://<PUBLIC_IP>:8000/transcribe \
  -F "file=@audio.mp3" \
  -F "format=json"

# With authentication (if enabled)
curl -X POST http://<PUBLIC_IP>:8000/transcribe \
  -H "X-API-Key: blitz_abc123" \
  -F "file=@audio.mp3" \
  -F "format=json"

# Prometheus metrics
curl http://<PUBLIC_IP>:8000/metrics
```

### Monitor Logs

```bash
# CloudWatch Logs
aws logs tail /aws/ec2/blitz-api-dev --follow

# Setup logs (initial deployment)
aws logs tail /aws/ec2/blitz-api-dev --log-stream-names 'i-xxx/setup' --follow

# API logs (runtime)
aws logs tail /aws/ec2/blitz-api-dev --log-stream-names 'i-xxx/api' --follow
```

## Security Best Practices

### Production Checklist

- [ ] Restrict SSH access: `--allowed-ssh 10.0.0.0/8` (your corporate CIDR)
- [ ] Restrict API access: `--allowed-api 10.0.0.0/8` (or use ALB + WAF)
- [ ] Enable authentication: `--enable-auth`
- [ ] Generate strong API keys: `python3 -c "import secrets; print(f'blitz_{secrets.token_urlsafe(32)}')"`
- [ ] Store API keys in Secrets Manager (not in CloudFormation parameters)
- [ ] Use private subnet + ALB/API Gateway for public access
- [ ] Enable CloudTrail for API auditing
- [ ] Set up CloudWatch alarms for instance health and API errors
- [ ] Enable EBS encryption at rest (enabled by default in template)
- [ ] Use IMDSv2 (enabled by default in template)
- [ ] Regularly update AMI to latest version
- [ ] Enable termination protection for production instances

### API Key Management

```bash
# Generate API key
python3 -c "import secrets; print(f'blitz_{secrets.token_urlsafe(32)}')"

# Store in Secrets Manager
aws secretsmanager create-secret \
  --name blitz-api-keys-prod \
  --secret-string '{"client1":["blitz_abc123"]}' \
  --region us-east-1

# Retrieve from Secrets Manager (for rotation)
aws secretsmanager get-secret-value \
  --secret-id blitz-api-keys-prod \
  --query SecretString \
  --output text
```

## Cost Optimization

### Instance Costs (us-east-1)

| Instance | GPU | VRAM | Spot ($/hr) | On-Demand ($/hr) | Monthly (24/7 Spot) |
|----------|-----|------|-------------|------------------|---------------------|
| g6.xlarge | L4 | 24GB | $0.42 | $1.05 | ~$302 |
| g6.2xlarge | L4 | 24GB | $0.67 | $1.68 | ~$482 |
| g5.xlarge | A10G | 24GB | $0.42 | $1.01 | ~$302 |
| g5.2xlarge | A10G | 24GB | $0.60 | $1.51 | ~$432 |

### Cost Reduction Strategies

1. **Auto-Shutdown** (dev/staging only):
   ```bash
   --auto-shutdown  # Shuts down at 11 PM daily
   ```
   Savings: ~8 hours/day = ~$100/month for g6.xlarge

2. **Spot Instances** (coming soon):
   - Not supported in CloudFormation for EC2 with EBS (use Auto Scaling Groups)
   - Savings: ~60% vs On-Demand

3. **Right-Sizing**:
   - Use `g6.xlarge` for dev/staging (sufficient for most workloads)
   - Scale to `g6.2xlarge` or multi-instance for production high-throughput

4. **Reserved Instances** (production):
   - 1-year savings plan: ~35% discount
   - 3-year savings plan: ~55% discount

5. **Scheduled Scaling** (if using Auto Scaling):
   - Scale down outside business hours
   - Scale to zero on weekends

## Troubleshooting

### Instance Not Responding

```bash
# Check instance status
aws ec2 describe-instance-status --instance-ids i-xxx

# View system logs
aws ec2 get-console-output --instance-ids i-xxx

# Connect via Session Manager (no SSH key required)
aws ssm start-session --target i-xxx
```

### Docker Container Failed

```bash
# SSH to instance
ssh -i my-keypair.pem ec2-user@<PUBLIC_IP>

# Check container status
docker ps -a

# View container logs
docker logs blitz

# Check setup logs
tail -100 /var/log/blitz-setup.log

# Restart container
docker restart blitz

# Rebuild and restart (if needed)
cd /opt/blitz
docker stop blitz && docker rm blitz
docker build -t blitz:latest .
docker run -d --name blitz --restart unless-stopped \
  --gpus all --ipc=host --shm-size=2g -p 8000:8000 \
  --env-file .env -v /opt/blitz/models:/root/.cache/huggingface \
  blitz:latest
```

### Models Not Loading

```bash
# Check disk space
df -h

# Check GPU availability
nvidia-smi

# Check CUDA environment
docker exec blitz nvidia-smi
docker exec blitz python -c "import torch; print(torch.cuda.is_available())"

# Check HuggingFace cache
ls -lh /opt/blitz/models/hub/

# Manually download models
docker exec blitz python -c "
from transformers import WhisperForConditionalGeneration
WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3-turbo')
"
```

### API Returning Errors

```bash
# Check API logs
docker logs -f blitz

# Test health endpoint
curl -v http://localhost:8000/health

# Check container resource usage
docker stats blitz

# Test with small file
curl -X POST http://localhost:8000/transcribe \
  -F "file=@small-audio.mp3" \
  -F "format=json"
```

## Stack Updates

### Update Parameters Only

```bash
# Example: Change rate limit
./scripts/deploy_ec2_api.sh \
  --vpc-id vpc-xxx \
  --subnet-id subnet-xxx \
  --key-pair my-keypair \
  --rate-limit 120
```

### Update Stack (Zero Downtime)

For configuration changes that don't require instance replacement:

```bash
# Update stack with new parameters
aws cloudformation update-stack \
  --stack-name blitz-ec2-api-dev \
  --template-body file://infrastructure/ec2-api/cloudformation.yaml \
  --parameters ParameterKey=RateLimitRPM,ParameterValue=120 \
  --capabilities CAPABILITY_NAMED_IAM

# SSH to instance and update .env
ssh -i my-keypair.pem ec2-user@<PUBLIC_IP>
sudo vi /opt/blitz/.env
sudo docker restart blitz
```

### Rollback

```bash
# CloudFormation rollback
aws cloudformation rollback-stack --stack-name blitz-ec2-api-dev

# Manual rollback via console
# CloudFormation → Stacks → blitz-ec2-api-dev → Update → Rollback
```

## Cleanup

```bash
# Delete stack (also deletes EIP, instance, security group)
aws cloudformation delete-stack --stack-name blitz-ec2-api-dev

# Wait for deletion
aws cloudformation wait stack-delete-complete --stack-name blitz-ec2-api-dev

# Verify cleanup
aws cloudformation describe-stacks --stack-name blitz-ec2-api-dev
# Should return error: Stack does not exist
```

## Advanced Configuration

### Use Pre-Built Docker Image

To skip 20-30 minute build time, pre-build and push to ECR:

```bash
# Build image locally or in CI/CD
cd /path/to/blitz-oss
docker build -t blitz:latest .

# Push to ECR
aws ecr create-repository --repository-name blitz --region us-east-1
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
docker tag blitz:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/blitz:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/blitz:latest

# Deploy with pre-built image
./scripts/deploy_ec2_api.sh \
  --vpc-id vpc-xxx \
  --subnet-id subnet-xxx \
  --key-pair my-keypair \
  --docker-image 123456789012.dkr.ecr.us-east-1.amazonaws.com/blitz:latest
```

### Use Specific AMI Version

```bash
# Find AMI ID
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text

# Update CloudFormation parameter
# Change AmiId parameter from SSM parameter to explicit AMI ID:
# AmiId: ami-0123456789abcdef0
```

### Private Subnet Deployment

For production, deploy in private subnet with NAT Gateway + ALB:

1. Deploy instance in private subnet (no public IP)
2. Create Application Load Balancer in public subnet
3. Configure ALB target group pointing to instance:8000
4. Update security group to allow traffic from ALB only
5. Optional: Add AWS WAF to ALB for DDoS protection

See `infrastructure/ec2-api/cloudformation-alb.yaml` (coming soon).

## Support

- **GitHub Issues**: https://github.com/anandmandilwar/blitz-oss/issues
- **Documentation**: https://github.com/anandmandilwar/blitz-oss/tree/main/docs
- **Email**: (your contact email)

## License

MIT
