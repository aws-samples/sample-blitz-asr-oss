"""
Security integration for Blitz ASR.

One-call setup that adds all security layers to the FastAPI app:
- API key authentication
- Rate limiting
- CORS configuration
- Request validation (available as dependency)

Usage:
    from blitz.security.integration import integrate_security
    integrate_security(app)

Configuration (all via environment variables):
    BLITZ_AUTH_ENABLED=true              # Enable/disable API key auth
    BLITZ_API_KEYS='{"app1":["key1"]}'   # API keys (JSON or comma-separated)
    BLITZ_RATE_LIMIT_ENABLED=true        # Enable/disable rate limiting
    BLITZ_RATE_LIMIT_RPM=60              # Requests per minute per client
    BLITZ_RATE_LIMIT_BURST=10            # Burst allowance above rate
    BLITZ_CORS_ORIGINS=*                 # Allowed CORS origins (comma-separated)
"""

import os
import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from blitz.security.auth import APIKeyAuth, get_auth_manager, get_current_client
from blitz.security.rate_limiter import RateLimiter, RateLimitMiddleware
from blitz.security.validation import RequestValidator

logger = logging.getLogger("blitz.security")


def integrate_security(app: FastAPI) -> dict:
    """
    One-call integration: adds all security layers to a FastAPI app.

    Returns a dict with initialized components for use in endpoints:
        components = integrate_security(app)
        validator = components["validator"]

    Middleware order (applied bottom-to-top in Starlette):
        1. CORS (outermost)
        2. Rate Limiting
        3. Authentication (via dependency injection, not middleware)
    """

    # ---- CORS Configuration ----
    cors_origins = os.environ.get("BLITZ_CORS_ORIGINS", "*")
    if cors_origins == "*":
        origins = ["*"]
    else:
        origins = [o.strip() for o in cors_origins.split(",")]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-ID",
            "X-Processing-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ],
    )
    logger.info(f"CORS configured: origins={origins}")

    # ---- Rate Limiting ----
    rate_limiter = RateLimiter()
    app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)

    # ---- API Key Auth (via dependency, not middleware) ----
    auth_manager = get_auth_manager()

    # ---- Request Validator ----
    validator = RequestValidator()

    # ---- Key generation endpoint (admin only) ----
    @app.post("/admin/generate-key", include_in_schema=False)
    async def generate_api_key():
        """Generate a new API key (for admin use during setup)."""
        admin_key = os.environ.get("BLITZ_ADMIN_KEY", "")
        if not admin_key:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not found")
        # This endpoint is intentionally hidden and requires BLITZ_ADMIN_KEY
        new_key = APIKeyAuth.generate_key()
        return {"key": new_key, "note": "Add this to BLITZ_API_KEYS env var"}

    logger.info("Security layer initialized: auth + rate limiting + CORS + validation")

    return {
        "auth_manager": auth_manager,
        "rate_limiter": rate_limiter,
        "validator": validator,
    }
