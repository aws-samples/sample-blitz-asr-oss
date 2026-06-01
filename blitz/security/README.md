# Blitz ASR — Security Layer

Production security for Blitz ASR: API key authentication, rate limiting, request validation, and audit logging.

## Quick Start

### 1. Add to server.py (after observability)

```python
from blitz.security.integration import integrate_security
security = integrate_security(app)
```

### 2. Configure via .env

```bash
# Authentication
BLITZ_AUTH_ENABLED=true
BLITZ_API_KEYS={"myapp": ["blitz_abc123"], "partner": ["blitz_def456"]}

# Rate Limiting
BLITZ_RATE_LIMIT_ENABLED=true
BLITZ_RATE_LIMIT_RPM=60          # 60 requests/min per client
BLITZ_RATE_LIMIT_BURST=10        # Allow 10 extra burst

# CORS
BLITZ_CORS_ORIGINS=http://localhost:8501,https://myapp.com
```

### 3. Use API keys in requests

```bash
# With auth enabled, all /transcribe requests need a key:
curl -X POST http://localhost:8000/transcribe \
  -H "X-API-Key: blitz_abc123" \
  -F "file=@audio.mp3" -F "format=json"

# Public endpoints don't need a key:
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

## Features

### 🔐 API Key Authentication
- **Header-based**: `X-API-Key` header
- **Multi-client**: Each client gets named keys
- **Key rotation**: Multiple valid keys per client (rotate without downtime)
- **Public exemptions**: `/health`, `/metrics` always accessible
- **Disable for dev**: Set `BLITZ_AUTH_ENABLED=false`

### ⏱️ Rate Limiting
- **Sliding window**: Per-client request counting
- **Configurable**: RPM + burst via env vars
- **Informational headers**: `X-RateLimit-Limit`, `X-RateLimit-Remaining`
- **429 response**: Includes `Retry-After` header
- **IP fallback**: Rate limits by IP if no API key

### 🛡️ Request Validation
- **File size limit**: Configurable max (default 500MB)
- **Content type check**: Only audio MIME types accepted
- **Magic byte verification**: Detects disguised files
- **Filename sanitization**: Prevents path traversal

### 📝 Audit Logging
- **Every request logged**: Client ID, timestamp, file hash, result
- **File hash only**: Audio content is NEVER stored in logs
- **Structured JSON**: Compatible with CloudWatch Insights queries

## Generating API Keys

```bash
# Generate a key locally
python3 -c "from blitz.security.auth import APIKeyAuth; print(APIKeyAuth.generate_key())"
# Output: blitz_a3Bx7K9mP2qR5tW8yZ1c4F6hJ0lN...
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `BLITZ_AUTH_ENABLED` | `true` | Enable API key authentication |
| `BLITZ_API_KEYS` | (empty) | API keys (JSON or comma-separated) |
| `BLITZ_RATE_LIMIT_ENABLED` | `true` | Enable rate limiting |
| `BLITZ_RATE_LIMIT_RPM` | `60` | Requests per minute per client |
| `BLITZ_RATE_LIMIT_BURST` | `10` | Burst above normal rate |
| `BLITZ_CORS_ORIGINS` | `*` | Allowed CORS origins |
| `MAX_UPLOAD_BYTES` | `524288000` | Max file size (500MB) |
| `BLITZ_ADMIN_KEY` | (empty) | Admin key for `/admin/generate-key` |

## Security Headers in Responses

Every response includes:
- `X-Request-ID` — Unique request correlation ID
- `X-Processing-Time` — Server-side processing duration
- `X-RateLimit-Limit` — Rate limit ceiling
- `X-RateLimit-Remaining` — Remaining requests in window

## Testing Without Auth (Development)

For local development, disable auth:
```bash
BLITZ_AUTH_ENABLED=false
BLITZ_RATE_LIMIT_ENABLED=false
```

## File Structure

```
blitz/security/
├── __init__.py       # Module exports
├── auth.py           # API key authentication
├── rate_limiter.py   # Sliding window rate limiter
├── validation.py     # Request validation (size, type, magic bytes)
├── audit.py          # Audit logging
├── integration.py    # One-call setup
└── README.md         # This file
```
