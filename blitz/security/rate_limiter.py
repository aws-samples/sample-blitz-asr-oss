"""
Rate limiting for Blitz ASR.

Implements a sliding window rate limiter:
- Per-client limiting (by API key or IP)
- Configurable requests per window
- Returns 429 Too Many Requests with Retry-After header
- In-memory storage (suitable for single-instance)

Configuration via env vars:
    BLITZ_RATE_LIMIT_RPM=60         # Requests per minute per client
    BLITZ_RATE_LIMIT_BURST=10       # Max burst above normal rate
    BLITZ_RATE_LIMIT_ENABLED=true   # Enable/disable rate limiting
"""

import os
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("blitz.security.rate_limiter")

# Endpoints exempt from rate limiting
EXEMPT_ENDPOINTS = {"/health", "/metrics", "/docs", "/openapi.json"}


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    enabled: bool = True
    requests_per_minute: int = 60
    burst_size: int = 10  # Additional burst allowance
    window_seconds: int = 60

    @classmethod
    def from_env(cls) -> "RateLimitConfig":
        return cls(
            enabled=os.environ.get("BLITZ_RATE_LIMIT_ENABLED", "true").lower() in ("true", "1"),
            requests_per_minute=int(os.environ.get("BLITZ_RATE_LIMIT_RPM", "60")),
            burst_size=int(os.environ.get("BLITZ_RATE_LIMIT_BURST", "10")),
            window_seconds=60,
        )


class RateLimiter:
    """
    Sliding window rate limiter.

    Tracks request timestamps per client and rejects requests
    that exceed the configured rate.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig.from_env()
        # client_id -> list of request timestamps
        self._windows: Dict[str, List[float]] = defaultdict(list)
        self._max_rate = self.config.requests_per_minute + self.config.burst_size

        if self.config.enabled:
            logger.info(
                f"Rate limiting ENABLED: {self.config.requests_per_minute} req/min "
                f"+ {self.config.burst_size} burst per client"
            )
        else:
            logger.info("Rate limiting DISABLED")

    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request (API key client or IP)."""
        # Try to get client ID from auth (set by auth middleware)
        client_id = getattr(request.state, "client_id", None)
        if client_id:
            return client_id

        # Fall back to IP address
        if request.client:
            return f"ip:{request.client.host}"
        return "unknown"

    def _clean_window(self, client_id: str, now: float):
        """Remove expired timestamps from the window."""
        cutoff = now - self.config.window_seconds
        self._windows[client_id] = [
            ts for ts in self._windows[client_id] if ts > cutoff
        ]

    def check_rate_limit(self, request: Request) -> Optional[JSONResponse]:
        """
        Check if request should be rate limited.

        Returns None if allowed, or a 429 JSONResponse if rate limited.
        """
        if not self.config.enabled:
            return None

        if request.url.path in EXEMPT_ENDPOINTS:
            return None

        client_id = self._get_client_id(request)
        now = time.time()

        # Clean old entries
        self._clean_window(client_id, now)

        # Check rate
        current_count = len(self._windows[client_id])
        if current_count >= self._max_rate:
            # Calculate retry-after
            oldest = self._windows[client_id][0] if self._windows[client_id] else now
            retry_after = int(self.config.window_seconds - (now - oldest)) + 1

            logger.warning(
                f"Rate limited client '{client_id}': {current_count}/{self._max_rate} "
                f"requests in window (retry after {retry_after}s)"
            )

            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "detail": f"Rate limit exceeded. Maximum {self.config.requests_per_minute} requests per minute.",
                    "retry_after_seconds": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.config.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(oldest + self.config.window_seconds)),
                },
            )

        # Allow request — record timestamp
        self._windows[client_id].append(now)

        return None

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for a client in the current window."""
        now = time.time()
        self._clean_window(client_id, now)
        current = len(self._windows[client_id])
        return max(0, self._max_rate - current)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that applies rate limiting to all requests."""

    def __init__(self, app, rate_limiter: Optional[RateLimiter] = None):
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter()

    async def dispatch(self, request: Request, call_next):
        # Check rate limit
        rejection = self.rate_limiter.check_rate_limit(request)
        if rejection:
            return rejection

        # Add rate limit headers to response
        response = await call_next(request)

        # Add informational headers
        if self.rate_limiter.config.enabled and request.url.path not in EXEMPT_ENDPOINTS:
            client_id = self.rate_limiter._get_client_id(request)
            remaining = self.rate_limiter.get_remaining(client_id)
            response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.config.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response
