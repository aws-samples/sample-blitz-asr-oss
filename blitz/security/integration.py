import os, logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from blitz.security.auth import get_auth_manager, APIKeyAuth
from blitz.security.rate_limiter import RateLimiter, RateLimitMiddleware
from blitz.security.validation import RequestValidator

logger = logging.getLogger("blitz.security")

def integrate_security(app: FastAPI) -> dict:
    # Deny cross-origin by default. Set BLITZ_CORS_ORIGINS to a comma-separated
    # allowlist (e.g. "https://app.example.com") to permit specific origins, or
    # "*" to allow any origin. Per the CORS spec, credentials are never sent
    # with a wildcard origin, so allow_credentials is disabled in that case.
    cors_origins = os.environ.get("BLITZ_CORS_ORIGINS", "").strip()
    origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
    allow_wildcard = origins == ["*"]
    allow_credentials = bool(origins) and not allow_wildcard
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=allow_credentials, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"], expose_headers=["X-Request-ID", "X-Processing-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining"])
    logger.info(f"CORS configured: origins={origins or 'none (cross-origin denied)'}, credentials={allow_credentials}")
    rate_limiter = RateLimiter()
    app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
    auth_manager = get_auth_manager()
    validator = RequestValidator()
    logger.info("Security layer initialized: auth + rate limiting + CORS + validation")
    return {"auth_manager": auth_manager, "rate_limiter": rate_limiter, "validator": validator}
