"""
Security module for Blitz ASR.

Provides:
- API key authentication (header-based)
- Rate limiting (per-client, sliding window)
- Request validation (file size, content type, abuse prevention)
- CORS configuration
- Audit logging (who transcribed what, when)
"""

from blitz.security.auth import APIKeyAuth, get_current_client
from blitz.security.rate_limiter import RateLimiter, RateLimitMiddleware
from blitz.security.validation import RequestValidator
from blitz.security.integration import integrate_security

__all__ = [
    "APIKeyAuth",
    "get_current_client",
    "RateLimiter",
    "RateLimitMiddleware",
    "RequestValidator",
    "integrate_security",
]
