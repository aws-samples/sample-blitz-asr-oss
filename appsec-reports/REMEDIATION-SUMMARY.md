# AppSec Remediation Summary — Blitz ASR

**Date:** 2026-06-30  
**Scope:** Infrastructure (CloudFormation), Container (Dockerfile)  
**Status:** All HIGH and MEDIUM findings remediated

---

## Findings Addressed

### HIGH — Infrastructure

| # | Finding | File | Fix |
|---|---------|------|-----|
| 1 | **Security Groups open to 0.0.0.0/0** (SSH + API) | `infrastructure/ec2-api/cloudformation.yaml` | Default CIDR changed from `0.0.0.0/0` to `10.0.0.0/8` (private networks). Operators must explicitly override for public access. |
| 2 | **API keys passed as plaintext in UserData** | `infrastructure/ec2-api/cloudformation.yaml` | Created `AWS::SecretsManager::Secret` resource; UserData now receives the Secret ARN (`API_KEYS_SECRET_ARN`). Application retrieves the value at runtime via Secrets Manager SDK. |
| 3 | **Overly permissive IAM (ECR `Resource: *`)** | `infrastructure/ec2-api/cloudformation.yaml` | Split into two statements: `ecr:GetAuthorizationToken` → `Resource: *` (AWS requirement); image-pull actions → scoped to `arn:aws:ecr:${Region}:${Account}:repository/blitz-*`. |

### MEDIUM — Infrastructure

| # | Finding | File | Fix |
|---|---------|------|-----|
| 4 | **Wildcard CORS (`BLITZ_CORS_ORIGINS=*`)** | `infrastructure/ec2-api/cloudformation.yaml` | Replaced with parameterized `AllowedCORSOrigins` (default: empty string = deny all). |
| 5 | **Secrets Manager IAM `Resource: *`** | `infrastructure/ec2-api/cloudformation.yaml` | Scoped to `!Ref ApiKeySecret` (the specific secret created in the same stack). |

### MEDIUM — Container

| # | Finding | File | Fix |
|---|---------|------|-----|
| 6 | **Container runs as root** | `Dockerfile` | Added `groupadd`/`useradd` for a `blitz` system user + `USER blitz` directive before `CMD`. |

### MEDIUM — Application (Bandit B104, B108) — Accepted Risk

| # | Finding | File | Disposition |
|---|---------|------|-------------|
| 7 | Bind to `0.0.0.0` | `blitz/server.py`, `blitz/banner.py` | **Accepted** — required for Docker networking. Already annotated with `# nosec B104`. |
| 8 | Hardcoded `/tmp` path | `blitz/diarize_endpoint.py` | **Accepted** — container-internal path to vendored diarization script. Already annotated with `# nosec B108`. |

---

## Files Modified

1. `infrastructure/ec2-api/cloudformation.yaml` — 5 changes (CIDR defaults, Secrets Manager resource, IAM scoping, API key removal from UserData, CORS param)
2. `Dockerfile` — 1 change (non-root user)

## Remaining Recommendations (not blocking)

- **Pin HuggingFace model revisions** (Bandit B615) — The `from_pretrained()` calls in vendor code should specify `revision=` to prevent supply-chain attacks via model poisoning. Lower risk since models are downloaded at build time and baked into the image.
- **Restrict Prometheus port 9090** — Consider using a separate, more restrictive CIDR for the metrics endpoint, or remove it entirely if not needed and use a pull-based metrics agent inside the VPC.
- **IMDSv2 enforcement** — The template already sets `HttpTokens: required` ✅.
