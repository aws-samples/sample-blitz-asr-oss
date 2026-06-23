import os, logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from blitz.security.auth import get_auth_manager, APIKeyAuth
from blitz.security.rate_limiter import RateLimiter, RateLimitMiddleware
from blitz.security.validation import RequestValidator

logger = logging.getLogger("blitz.security")

def integrate_security(app: FastAPI) -> dict:
    cors_origins = os.environ.get("BLITZ_CORS_ORIGINS", "*")
    origins = ["*"] if cors_origins == "*" else [o.strip() for o in cors_origins.split(",")]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"], expose_headers=["X-Request-ID", "X-Processing-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining"])
    logger.info(f"CORS configured: origins={origins}")
    rate_limiter = RateLimiter()
    app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
    auth_manager = get_auth_manager()
    validator = RequestValidator()
    logger.info("Security layer initialized: auth + rate limiting + CORS + validation")
    return {"auth_manager": auth_manager, "rate_limiter": rate_limiter, "validator": validator}
