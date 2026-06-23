# Blitz ASR EC2 Deployment - Quick Start Guide

This guide helps you deploy Blitz ASR to a new AWS account using CloudFormation.

## Prerequisites Checklist

- [ ] AWS Account with administrative access
- [ ] AWS CLI installed and configured (`aws configure`)
- [ ] VPC with public subnet (has internet gateway)
- [ ] EC2 Key Pair created in target region
- [ ] Python 3 installed (for API key generation)

## Step-by-Step Deployment

### 1. Gather Required Information

You need the following from your AWS account:

```bash
# Get your VPC ID
aws ec2 describe-vpcs --query 'Vpcs[].VpcId' --output table

# Get public subnet IDs (must have internet gateway)
aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=vpc-YOUR_VPC_ID" \
  --query 'Subnets[?MapPublicIpOnLaunch==`true`].[SubnetId,AvailabilityZone,CidrBlock]' \
  --output table

# List available key pairs
aws ec2 describe-key-pairs --query 'KeyPairs[].KeyName' --output table

# Get current AWS account ID and region
aws sts get-caller-identity
aws configure get region
```

### 2. Deploy Using Script (Recommended)

```bash
# Navigate to project root
cd /path/to/blitz-oss

# Run deployment script
./scripts/deploy_ec2_api.sh \
  --vpc-id vpc-0123456789abcdef0 \
  --subnet-id subnet-0123456789abcdef0 \
  --key-pair my-keypair

# For production with authentication:
./scripts/deploy_ec2_api.sh \
  --environment prod \
  --vpc-id vpc-0123456789abcdef0 \
  --subnet-id subnet-0123456789abcdef0 \
  --key-pair my-keypair \
  --enable-auth \
  --allowed-ssh 10.0.0.0/8 \
  --allowed-api 10.0.0.0/8
```

### 3. Alternative: Deploy Via AWS Console

1. **Open CloudFormation Console**:
   - Navigate to: https://console.aws.amazon.com/cloudformation
   - Click "Create Stack" → "With new resources (standard)"

2. **Upload Template**:
   - Select "Upload a template file"
   - Choose file: `infrastructure/ec2-api/cloudformation.yaml`
   - Click "Next"

3. **Configure Stack**:
   - **Stack name**: `blitz-ec2-api-dev`
   - **Parameters**:
     - Environment: `dev`
     - VpcId: Select your VPC
     - SubnetId: Select public subnet
     - KeyPairName: Select your key pair
     - InstanceType: `g6.xlarge` (recommended)
     - Leave other parameters as default for initial deployment

4. **Configure Options** (Optional):
   - Add tags if needed
   - Click "Next"

5. **Review and Create**:
   - Check "I acknowledge that AWS CloudFormation might create IAM resources"
   - Click "Create Stack"

6. **Wait for Completion**:
   - Stack creation takes 30-40 minutes (first time, downloads and builds)
   - Status will change to `CREATE_COMPLETE`

### 4. Verify Deployment

```bash
# Get stack outputs (includes Public IP and API endpoint)
aws cloudformation describe-stacks \
  --stack-name blitz-ec2-api-dev \
  --query 'Stacks[0].Outputs' \
  --output table

# Save Public IP for testing
PUBLIC_IP=$(aws cloudformation describe-stacks \
  --stack-name blitz-ec2-api-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`PublicIP`].OutputValue' \
  --output text)

echo "API Endpoint: http://$PUBLIC_IP:8000"

# Test health endpoint
curl http://$PUBLIC_IP:8000/health

# Expected output:
# {
#   "status": "healthy",
#   "models": {
#     "parakeet": {"loaded": true, "device": "cuda"},
#     "whisper": {"loaded": true, "device": "cuda"},
#     "voxlingua": {"loaded": true, "device": "cuda"}
#   }
# }
```

### 5. Test Transcription

```bash
# Download a test audio file
curl -O https://www2.cs.uic.edu/~i101/SoundFiles/StarWars60.wav

# Test transcription
curl -X POST http://$PUBLIC_IP:8000/transcribe \
  -F "file=@StarWars60.wav" \
  -F "format=json" | jq .

# Expected output includes:
# {
#   "text": "...",
#   "language": {
#     "code": "en",
#     "confidence": 0.98
#   },
#   "engine": "tier1",
#   "metrics": {
#     "real_time_factor": 0.01,
#     ...
#   }
# }
```

### 6. Monitor Instance

```bash
# View CloudWatch logs (setup logs)
aws logs tail /aws/ec2/blitz-api-dev --follow

# SSH to instance (for troubleshooting)
ssh -i ~/.ssh/my-keypair.pem ec2-user@$PUBLIC_IP

# Once connected:
docker ps                          # Check container status
docker logs -f blitz               # View application logs
tail -f /var/log/blitz-setup.log   # View setup logs
nvidia-smi                         # Check GPU status
```

## Common Deployment Scenarios

### Scenario 1: Development Environment

Quick setup for testing:

```bash
./scripts/deploy_ec2_api.sh \
  --vpc-id vpc-xxx \
  --subnet-id subnet-xxx \
  --key-pair my-key \
  --auto-shutdown
```

**Features**:
- No authentication (open access)
- Auto-shutdown at 11 PM to save costs
- Open security group (0.0.0.0/0)

**Cost**: ~$100/month with auto-shutdown (8h/day savings)

### Scenario 2: Production Environment

Secure deployment with authentication:

```bash
# Generate API key
API_KEY=$(python3 -c "import secrets; print(f'blitz_{secrets.token_urlsafe(32)}')")
echo "Save this key: $API_KEY"

# Deploy
./scripts/deploy_ec2_api.sh \
  --environment prod \
  --vpc-id vpc-xxx \
  --subnet-id subnet-xxx \
  --key-pair my-key \
  --enable-auth \
  --api-keys "{\"production\":[\"$API_KEY\"]}" \
  --allowed-ssh 10.0.0.0/8 \
  --allowed-api 10.0.0.0/8 \
  --instance-type g6.xlarge

# Test with API key
curl -X POST http://$PUBLIC_IP:8000/transcribe \
  -H "X-API-Key: $API_KEY" \
  -F "file=@audio.mp3" \
  -F "format=json"
```

**Features**:
- API key authentication required
- Restricted access (corporate network only)
- No auto-shutdown
- Rate limiting enabled (60 req/min)

**Cost**: ~$302/month (24/7 Spot pricing)

### Scenario 3: Fast Deployment (Pre-Built Image)

Skip 20-30 minute build time by using pre-built image:

```bash
# First, build and push to ECR (one-time setup)
# See "Building Docker Image" section below

# Deploy with pre-built image
./scripts/deploy_ec2_api.sh \
  --vpc-id vpc-xxx \
  --subnet-id subnet-xxx \
  --key-pair my-key \
  --docker-image 123456789012.dkr.ecr.us-east-1.amazonaws.com/blitz:latest
```

**Time**: ~5-10 minutes (vs 30-40 minutes building from source)

## Building Docker Image (Optional)

To use pre-built image and skip build time during deployment:

```bash
# 1. Create ECR repository
aws ecr create-repository \
  --repository-name blitz \
  --region us-east-1

# 2. Build image locally (or in CI/CD)
cd /path/to/blitz-oss
docker build -t blitz:latest -f Dockerfile .

# 3. Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789012.dkr.ecr.us-east-1.amazonaws.com

# 4. Tag and push
docker tag blitz:latest \
  123456789012.dkr.ecr.us-east-1.amazonaws.com/blitz:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/blitz:latest

# 5. Use in deployment
./scripts/deploy_ec2_api.sh \
  --vpc-id vpc-xxx \
  --subnet-id subnet-xxx \
  --key-pair my-key \
  --docker-image 123456789012.dkr.ecr.us-east-1.amazonaws.com/blitz:latest
```

## Multi-Account Deployment

To deploy to multiple AWS accounts:

### Option 1: Use Profiles

```bash
# Configure AWS profiles for each account
aws configure --profile account1
aws configure --profile account2

# Deploy to account1
AWS_PROFILE=account1 ./scripts/deploy_ec2_api.sh \
  --vpc-id vpc-account1 \
  --subnet-id subnet-account1 \
  --key-pair key-account1

# Deploy to account2
AWS_PROFILE=account2 ./scripts/deploy_ec2_api.sh \
  --vpc-id vpc-account2 \
  --subnet-id subnet-account2 \
  --key-pair key-account2
```

### Option 2: Use StackSets (AWS Organizations)

For deploying to many accounts automatically:

```bash
# Create StackSet (from organization management account)
aws cloudformation create-stack-set \
  --stack-set-name blitz-ec2-api \
  --template-body file://infrastructure/ec2-api/cloudformation.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameters file://infrastructure/ec2-api/parameters.example.json

# Deploy to multiple accounts
aws cloudformation create-stack-instances \
  --stack-set-name blitz-ec2-api \
  --accounts 123456789012 234567890123 \
  --regions us-east-1 us-west-2
```

## Troubleshooting

### Issue: Stack creation fails with "VPC not found"

**Cause**: VPC ID doesn't exist in target region

**Solution**:
```bash
# Verify VPC exists
aws ec2 describe-vpcs --vpc-ids vpc-xxx --region us-east-1

# If not, create VPC or use correct region
```

### Issue: Instance not responding after 40 minutes

**Cause**: Build might have failed or models still downloading

**Solution**:
```bash
# Check CloudWatch logs
aws logs tail /aws/ec2/blitz-api-dev --follow

# SSH to instance
ssh -i my-key.pem ec2-user@$PUBLIC_IP

# Check setup log
tail -100 /var/log/blitz-setup.log

# Check Docker logs
docker logs blitz

# Manually restart if needed
docker restart blitz
```

### Issue: "No space left on device"

**Cause**: EBS volume too small for models + Docker images

**Solution**:
```bash
# Delete stack and redeploy with larger volume
aws cloudformation delete-stack --stack-name blitz-ec2-api-dev

# Redeploy with larger EBS
./scripts/deploy_ec2_api.sh \
  --vpc-id vpc-xxx \
  --subnet-id subnet-xxx \
  --key-pair my-key \
  --ebs-size 200
```

### Issue: API key authentication not working

**Cause**: API key format incorrect or missing from request

**Solution**:
```bash
# Verify API key format (must start with "blitz_")
echo '{"client1":["blitz_abc123"]}' | jq .

# Include in request header
curl -X POST http://$PUBLIC_IP:8000/transcribe \
  -H "X-API-Key: blitz_abc123" \
  -F "file=@audio.mp3" \
  -F "format=json"

# Check container logs for auth errors
docker logs blitz | grep -i auth
```

## Cost Estimation

### Monthly Costs (us-east-1, 24/7)

| Configuration | Instance | GPU | Spot $/month | On-Demand $/month |
|---------------|----------|-----|--------------|-------------------|
| Dev (8h/day) | g6.xlarge | L4 | ~$100 | ~$252 |
| Staging | g6.xlarge | L4 | ~$302 | ~$756 |
| Production | g6.xlarge | L4 | ~$302 | ~$756 |
| Production (HA) | 2x g6.xlarge | L4 | ~$604 | ~$1,512 |

**Additional costs**:
- EBS storage: ~$15/month for 150GB gp3
- Data transfer: $0.09/GB out (first 100GB free)
- CloudWatch Logs: $0.50/GB ingested

### Cost Optimization Tips

1. **Use auto-shutdown** for dev/staging (`--auto-shutdown`)
2. **Use Spot instances** (60% savings, requires Auto Scaling Group)
3. **Use Reserved Instances** for production (35-55% savings)
4. **Right-size instance** based on load
5. **Use smaller EBS** if not storing many models (100GB minimum)

## Next Steps

After successful deployment:

1. **Update Streamlit app** with new API endpoint:
   ```python
   # In blitz_app.py, update line 13:
   BLITZ_API = "http://<YOUR_PUBLIC_IP>:8000"
   ```

2. **Set up monitoring**:
   - Create CloudWatch dashboard
   - Configure alarms for API errors, high latency, GPU memory
   - Enable AWS X-Ray for request tracing

3. **Production hardening**:
   - Deploy behind Application Load Balancer
   - Add AWS WAF for DDoS protection
   - Use private subnet + NAT Gateway
   - Store API keys in Secrets Manager
   - Enable AWS Config for compliance

4. **Scale out** (if needed):
   - Create Auto Scaling Group with multiple instances
   - Add Application Load Balancer for traffic distribution
   - Use ElastiCache for rate limiting across instances

## Support

- **Documentation**: See [README.md](README.md) for detailed reference
- **GitHub Issues**: https://github.com/anandmandilwar/blitz-oss/issues
- **AWS Support**: For account-level issues

## Additional Resources

- [CloudFormation Template](cloudformation.yaml)
- [Deployment Script](../../scripts/deploy_ec2_api.sh)
- [Parameter Examples](parameters.example.json)
- [Main Documentation](../../README.md)
- [Architecture Guide](../../docs/ARCHITECTURE.md)
