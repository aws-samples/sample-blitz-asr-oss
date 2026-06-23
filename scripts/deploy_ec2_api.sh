#!/bin/bash
set -euo pipefail

# ============================================================================
# Blitz ASR - EC2 API Deployment Script
# ============================================================================
# Deploys GPU EC2 instance with Blitz ASR HTTP API using CloudFormation
#
# Usage:
#   ./deploy_ec2_api.sh --environment dev --region us-east-1 \
#     --vpc-id vpc-xxx --subnet-id subnet-xxx --key-pair my-key
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CFN_TEMPLATE="$PROJECT_ROOT/infrastructure/ec2-api/cloudformation.yaml"

# Default values
ENVIRONMENT="dev"
REGION="us-east-1"
INSTANCE_TYPE="g6.xlarge"
EBS_VOLUME_SIZE=150
TIER2_BACKEND="nemo"
TIER2_PRECISION="bfloat16"
ENABLE_AUTH="false"
RATE_LIMIT_RPM=60
ALLOWED_SSH_CIDR="0.0.0.0/0"
ALLOWED_API_CIDR="0.0.0.0/0"
ENABLE_AUTO_SHUTDOWN="No"
DOCKER_IMAGE=""

# ============================================================================
# Functions
# ============================================================================

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Required:
  --vpc-id VPC_ID              VPC ID for the instance
  --subnet-id SUBNET_ID        Public subnet ID (must have internet gateway)
  --key-pair KEY_NAME          EC2 key pair name for SSH access

Optional:
  --environment ENV            Environment name (dev|staging|prod, default: dev)
  --region REGION              AWS region (default: us-east-1)
  --instance-type TYPE         GPU instance type (default: g6.xlarge)
  --ebs-size GB                Root volume size in GB (default: 150)
  --tier2-backend BACKEND      Parakeet backend: nemo|onnx (default: nemo)
  --tier2-precision PREC       Model precision: fp32|bfloat16|fp16 (default: bfloat16)
  --enable-auth                Enable API key authentication (default: disabled)
  --api-keys JSON              API keys as JSON map (required if --enable-auth)
  --rate-limit RPM             Requests per minute limit (default: 60)
  --allowed-ssh CIDR           CIDR for SSH access (default: 0.0.0.0/0)
  --allowed-api CIDR           CIDR for API access (default: 0.0.0.0/0)
  --auto-shutdown              Enable auto-shutdown at 11 PM (dev/staging only)
  --docker-image URI           Pre-built Docker image URI (skips build)
  --help                       Show this help message

Examples:
  # Basic deployment
  $0 --vpc-id vpc-abc123 --subnet-id subnet-def456 --key-pair my-key

  # Production with auth and restricted access
  $0 --environment prod --vpc-id vpc-abc123 --subnet-id subnet-def456 \\
     --key-pair my-key --enable-auth \\
     --api-keys '{"client1":["blitz_abc123"]}' \\
     --allowed-ssh 10.0.0.0/8 --allowed-api 10.0.0.0/8

  # Dev with auto-shutdown and pre-built image
  $0 --vpc-id vpc-abc123 --subnet-id subnet-def456 --key-pair my-key \\
     --auto-shutdown --docker-image 123456789012.dkr.ecr.us-east-1.amazonaws.com/blitz:latest

EOF
    exit 0
}

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[ERROR] $*" >&2
    exit 1
}

check_dependencies() {
    local deps=("aws" "jq")
    for cmd in "${deps[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error "$cmd is required but not installed"
        fi
    done
}

validate_cidr() {
    local cidr=$1
    if ! echo "$cidr" | grep -qE '^([0-9]{1,3}\.){3}[0-9]{1,3}/[0-9]{1,2}$'; then
        error "Invalid CIDR format: $cidr"
    fi
}

generate_api_key() {
    python3 -c "import secrets; print(f'blitz_{secrets.token_urlsafe(32)}')"
}

# ============================================================================
# Parse Arguments
# ============================================================================

VPC_ID=""
SUBNET_ID=""
KEY_PAIR=""
API_KEYS=""

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
        --subnet-id)
            SUBNET_ID="$2"
            shift 2
            ;;
        --key-pair)
            KEY_PAIR="$2"
            shift 2
            ;;
        --instance-type)
            INSTANCE_TYPE="$2"
            shift 2
            ;;
        --ebs-size)
            EBS_VOLUME_SIZE="$2"
            shift 2
            ;;
        --tier2-backend)
            TIER2_BACKEND="$2"
            shift 2
            ;;
        --tier2-precision)
            TIER2_PRECISION="$2"
            shift 2
            ;;
        --enable-auth)
            ENABLE_AUTH="true"
            shift
            ;;
        --api-keys)
            API_KEYS="$2"
            shift 2
            ;;
        --rate-limit)
            RATE_LIMIT_RPM="$2"
            shift 2
            ;;
        --allowed-ssh)
            ALLOWED_SSH_CIDR="$2"
            shift 2
            ;;
        --allowed-api)
            ALLOWED_API_CIDR="$2"
            shift 2
            ;;
        --auto-shutdown)
            ENABLE_AUTO_SHUTDOWN="Yes"
            shift
            ;;
        --docker-image)
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# ============================================================================
# Validation
# ============================================================================

log "Validating configuration..."

[[ -z "$VPC_ID" ]] && error "Missing required parameter: --vpc-id"
[[ -z "$SUBNET_ID" ]] && error "Missing required parameter: --subnet-id"
[[ -z "$KEY_PAIR" ]] && error "Missing required parameter: --key-pair"

if [[ "$ENABLE_AUTH" == "true" && -z "$API_KEYS" ]]; then
    log "⚠️  Auth enabled but no API keys provided. Generating a key..."
    GENERATED_KEY=$(generate_api_key)
    API_KEYS="{\"default\":[\"$GENERATED_KEY\"]}"
    log "Generated API key: $GENERATED_KEY"
    log "⚠️  Save this key securely - it won't be shown again!"
fi

validate_cidr "$ALLOWED_SSH_CIDR"
validate_cidr "$ALLOWED_API_CIDR"

check_dependencies

# Verify CloudFormation template exists
[[ ! -f "$CFN_TEMPLATE" ]] && error "CloudFormation template not found: $CFN_TEMPLATE"

# Verify AWS credentials and region
aws sts get-caller-identity --region "$REGION" > /dev/null 2>&1 || \
    error "AWS credentials not configured or invalid for region $REGION"

# Verify VPC and subnet exist
log "Verifying VPC and subnet..."
aws ec2 describe-vpcs --vpc-ids "$VPC_ID" --region "$REGION" > /dev/null 2>&1 || \
    error "VPC $VPC_ID not found in region $REGION"
aws ec2 describe-subnets --subnet-ids "$SUBNET_ID" --region "$REGION" > /dev/null 2>&1 || \
    error "Subnet $SUBNET_ID not found in region $REGION"

# Verify key pair exists
log "Verifying EC2 key pair..."
aws ec2 describe-key-pairs --key-names "$KEY_PAIR" --region "$REGION" > /dev/null 2>&1 || \
    error "Key pair $KEY_PAIR not found in region $REGION"

# ============================================================================
# Deploy CloudFormation Stack
# ============================================================================

STACK_NAME="blitz-ec2-api-${ENVIRONMENT}"

log "=========================================="
log "Blitz ASR - EC2 API Deployment"
log "=========================================="
log "Environment:     $ENVIRONMENT"
log "Region:          $REGION"
log "Stack Name:      $STACK_NAME"
log "VPC:             $VPC_ID"
log "Subnet:          $SUBNET_ID"
log "Instance Type:   $INSTANCE_TYPE"
log "Key Pair:        $KEY_PAIR"
log "Auth Enabled:    $ENABLE_AUTH"
log "Auto Shutdown:   $ENABLE_AUTO_SHUTDOWN"
log "=========================================="

read -p "Proceed with deployment? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log "Deployment cancelled"
    exit 0
fi

log "Validating CloudFormation template..."
aws cloudformation validate-template \
    --template-body "file://$CFN_TEMPLATE" \
    --region "$REGION" \
    > /dev/null || error "Template validation failed"

log "Deploying CloudFormation stack..."

PARAMETERS=(
    "ParameterKey=Environment,ParameterValue=$ENVIRONMENT"
    "ParameterKey=VpcId,ParameterValue=$VPC_ID"
    "ParameterKey=SubnetId,ParameterValue=$SUBNET_ID"
    "ParameterKey=KeyPairName,ParameterValue=$KEY_PAIR"
    "ParameterKey=InstanceType,ParameterValue=$INSTANCE_TYPE"
    "ParameterKey=EBSVolumeSize,ParameterValue=$EBS_VOLUME_SIZE"
    "ParameterKey=AllowedSSHCIDR,ParameterValue=$ALLOWED_SSH_CIDR"
    "ParameterKey=AllowedAPICIDR,ParameterValue=$ALLOWED_API_CIDR"
    "ParameterKey=Tier2Backend,ParameterValue=$TIER2_BACKEND"
    "ParameterKey=Tier2Precision,ParameterValue=$TIER2_PRECISION"
    "ParameterKey=EnableAuth,ParameterValue=$ENABLE_AUTH"
    "ParameterKey=ApiKeys,ParameterValue=$API_KEYS"
    "ParameterKey=RateLimitRPM,ParameterValue=$RATE_LIMIT_RPM"
    "ParameterKey=EnableAutoShutdown,ParameterValue=$ENABLE_AUTO_SHUTDOWN"
    "ParameterKey=BlitzDockerImage,ParameterValue=$DOCKER_IMAGE"
)

aws cloudformation deploy \
    --template-file "$CFN_TEMPLATE" \
    --stack-name "$STACK_NAME" \
    --parameter-overrides "${PARAMETERS[@]}" \
    --capabilities CAPABILITY_NAMED_IAM \
    --region "$REGION" \
    --tags \
        Environment="$ENVIRONMENT" \
        Project=blitz-asr \
        ManagedBy=CloudFormation

if [[ $? -ne 0 ]]; then
    error "CloudFormation deployment failed"
fi

log "✓ Stack deployed successfully"

# ============================================================================
# Retrieve Outputs
# ============================================================================

log "Retrieving stack outputs..."

STACK_INFO=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0]' \
    --output json)

get_output() {
    echo "$STACK_INFO" | jq -r ".Outputs[] | select(.OutputKey==\"$1\") | .OutputValue"
}

INSTANCE_ID=$(get_output "InstanceId")
PUBLIC_IP=$(get_output "PublicIP")
API_ENDPOINT=$(get_output "ApiEndpoint")
HEALTH_CHECK=$(get_output "HealthCheckUrl")
SSH_COMMAND=$(get_output "SSHCommand")

log "=========================================="
log "Deployment Complete!"
log "=========================================="
log "Instance ID:     $INSTANCE_ID"
log "Public IP:       $PUBLIC_IP"
log "API Endpoint:    $API_ENDPOINT"
log "Health Check:    $HEALTH_CHECK"
log "SSH Command:     $SSH_COMMAND"
log "=========================================="

if [[ -n "$DOCKER_IMAGE" ]]; then
    log "⏱️  Instance will be ready in ~5-10 minutes (using pre-built image)"
else
    log "⏱️  Instance will be ready in ~30-40 minutes (building from source)"
fi

log ""
log "Next Steps:"
log "1. Wait for instance initialization to complete"
log "2. Check health: curl $HEALTH_CHECK"
log "3. Test transcription:"
log "   curl -X POST $API_ENDPOINT/transcribe \\"
log "     -F 'file=@audio.mp3' -F 'format=json'"
log "4. View logs:"
log "   aws logs tail /aws/ec2/blitz-api-${ENVIRONMENT} --follow --region $REGION"
log "5. SSH access:"
log "   $SSH_COMMAND"

if [[ "$ENABLE_AUTH" == "true" && -n "$GENERATED_KEY" ]]; then
    log ""
    log "⚠️  IMPORTANT - Your generated API key:"
    log "   $GENERATED_KEY"
    log "   Include this in requests: -H 'X-API-Key: $GENERATED_KEY'"
fi

log ""
log "✓ Deployment script completed"
