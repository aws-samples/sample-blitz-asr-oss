# Security Exceptions & Deferred Findings

**Last updated:** 2026-08-02
**Scanner:** AWS Automated Security Helper (ASH) v3.2.4 — `ash scan --mode local`

This document records security-scan findings that are intentionally **deferred**
(accepted as residual risk) or **suppressed as false positives**, with the
rationale for each. It is the audit trail for the suppressions embedded in the
source (cdk-nag `Metadata` blocks and `# pragma: allowlist secret` comments).

Findings **fixed** in code (CORS default, Docker `USER`/`HEALTHCHECK`, S3
versioning/logging, KMS encryption of secrets and log groups) are not listed
here — see git history on the `code-review` branch.

Each exception should be revisited whenever the affected resource changes or at
the next periodic security review.

---

## 1. Deferred — accepted residual risk

### 1.1 IAM wildcard on BatchJobRole S3 object access

- **Scanner / rule:** cdk-nag `AwsSolutions-IAM5`
- **Location:** `infrastructure/batch/cloudformation.yaml` — `BatchJobRole`
- **Finding:** IAM policy uses a wildcard in the resource ARN.
- **Why deferred:** Object-level S3 access inherently requires an object-path
  wildcard (`arn:aws:s3:::blitz-input-*/*`). The wildcard is confined to the
  object path *within* the specific Blitz input/output buckets — it does not
  widen the set of buckets — and the allowed actions are limited to
  `s3:GetObject`, `s3:GetObjectVersion`, `s3:PutObject`, and `s3:ListBucket`.
  A batch job must read/write arbitrary object keys the pipeline produces, so
  this is already least-privilege for the workload.
- **Residual risk:** Low. Compromise of the job role would allow read/write of
  objects only within the two named buckets.
- **Suppression:** `cdk_nag.rules_to_suppress` on the `BatchJobRole` resource.

### 1.2 AWS-managed service-role policies (managed-policy usage)

- **Scanner / rule:** cdk-nag `AwsSolutions-IAM4`
- **Locations:**
  - `infrastructure/batch/cloudformation.yaml` — `BatchServiceRole`
    (`AWSBatchServiceRole`)
  - `infrastructure/batch/cloudformation.yaml` — `BatchInstanceRole`
    (`AmazonEC2ContainerServiceforEC2Role`, `AmazonSSMManagedInstanceCore`)
  - `infrastructure/batch/cloudformation.yaml` — `BatchExecutionRole`
    (`AmazonECSTaskExecutionRolePolicy`)
  - `infrastructure/ec2-api/cloudformation.yaml` — `EC2InstanceRole`
    (`CloudWatchAgentServerPolicy`, `AmazonSSMManagedInstanceCore`)
- **Finding:** Roles attach AWS-managed policies rather than customer-managed
  inline policies.
- **Why deferred:** These are the AWS-recommended service-role policies for
  Batch, ECS-on-EC2, ECS task execution, the CloudWatch agent, and SSM Session
  Manager (which replaces inbound SSH). Re-implementing them as inline policies
  would duplicate AWS-maintained permission sets, drift out of date as the
  services evolve, and is more likely to *reduce* security than improve it.
- **Residual risk:** Low. Scope is bounded by each managed policy's documented
  permissions.
- **Suppression:** `cdk_nag.rules_to_suppress` on each role resource.

### 1.3 ECR `GetAuthorizationToken` wildcard

- **Scanner / rule:** cdk-nag `AwsSolutions-IAM5`
- **Locations:** `BatchExecutionRole` (batch) and `EC2InstanceRole` (ec2-api)
- **Finding:** `ecr:GetAuthorizationToken` uses `Resource: '*'`.
- **Why deferred:** Per AWS documentation, `ecr:GetAuthorizationToken` does not
  support resource-level permissions and **must** use `Resource: '*'`. Every
  other ECR action in the same policies is scoped to
  `arn:aws:ecr:<region>:<account>:repository/blitz-*`.
- **Residual risk:** Negligible. The token action only returns a short-lived
  registry auth token; it grants no data access on its own.
- **Suppression:** `cdk_nag.rules_to_suppress` on each role resource.

---

## 2. Suppressed — false positives

### 2.1 "Secret keyword" on an HTTP header name

- **Scanner / rule:** detect-secrets `SECRET-SECRET-KEYWORD`
- **Location:** `blitz/security/auth.py:8`
- **Finding:** Secret of type "Secret Keyword" detected.
- **Reality:** The flagged line assigns the *name* of an HTTP header (the
  `X-API-Key` header), not a credential. No secret value is present. See
  `blitz/security/auth.py:8` for the actual line.
- **Suppression:** inline `# pragma: allowlist secret` comment.

### 2.2 "Secret keyword" on an IAM action string

- **Scanner / rule:** detect-secrets `SECRET-SECRET-KEYWORD`
- **Location:** `infrastructure/ec2-api/cloudformation.yaml` — `secretsmanager:GetSecretValue`
- **Finding:** Secret of type "Secret Keyword" detected.
- **Reality:** The flagged token is the IAM action identifier
  `secretsmanager:GetSecretValue` in a policy statement — an API action name,
  not a secret. The actual API keys live in AWS Secrets Manager (CMK-encrypted),
  never in the template.
- **Suppression:** inline `# pragma: allowlist secret` comment.

---

## 3. Out of scope — vendored / generated code

These findings are in third-party vendored code or scanner-generated artifacts,
not in first-party project source. They are not suppressed in-code (the files
are not ours to annotate) and are excluded from remediation.

| Scanner / rule | Location | Note |
|---|---|---|
| bandit `B605` (shell injection) | `.ash/ash_output/converted/jupyter/…Whisper_Transcription_-_NeMo_Diarization…ipynb-converted.py:690` | ASH-generated conversion of a **vendored** whisper-diarization notebook. `.ash/` is gitignored. |
| checkov `CKV2_GHA_1` (GH Actions write-all) | `blitz/services/diarization/vendor/ctc_forced_aligner/.github/workflows/CI.yml` | Vendored upstream CI workflow (ctc-forced-aligner). |
| checkov `CKV2_GHA_1` (GH Actions write-all) | `blitz/services/diarization/vendor/whisper-diarization/.github/workflows/test_run.yml` | Vendored upstream CI workflow (whisper-diarization). |

**LOW/INFO findings** (bandit `B101` asserts, `B404`/`B603`/`B607` subprocess
usage with fixed argument arrays, `B110` try/except/pass, cdk-nag `AwsSolutions-S*`
informational notes) are below the actionable threshold and are not tracked
here. The subprocess calls in `blitz/server.py`, `blitz/diarize_endpoint.py`,
and `blitz/batch/transcribe_job.py` use list-form arguments without
`shell=True`, so no shell-injection path exists.

---

## Attestation

- [ ] Reviewed by: _______________
- [ ] Date: _______________
- [ ] Decision: PASS / PASS WITH CONDITIONS / FAIL
- [ ] Notes: _______________
