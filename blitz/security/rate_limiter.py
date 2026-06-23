import os, time, logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("blitz.security.rate_limiter")
EXEMPT_ENDPOINTS = {"/health", "/metrics", "/docs", "/openapi.json"}

@dataclass
class RateLimitConfig:
    enabled: bool = True
    requests_per_minute: int = 60
    burst_size: int = 10
    window_seconds: int = 60
    @classmethod
    def from_env(cls):
        return cls(
            enabled=os.environ.get("BLITZ_RATE_LIMIT_ENABLED", "true").lower() in ("true", "1"),
            requests_per_minute=int(os.environ.get("BLITZ_RATE_LIMIT_RPM", "60")),
            burst_size=int(os.environ.get("BLITZ_RATE_LIMIT_BURST", "10")),
        )

class RateLimiter:
    def __init__(self, config=None):
        self.config = config or RateLimitConfig.from_env()
        self._windows: Dict[str, List[float]] = defaultdict(list)
        self._max_rate = self.config.requests_per_minute + self.config.burst_size
        if self.config.enabled:
            logger.info(f"Rate limiting ENABLED: {self.config.requests_per_minute} req/min + {self.config.burst_size} burst")
        else:
            logger.info("Rate limiting DISABLED")

    def _get_client_id(self, request: Request) -> str:
        client_id = getattr(request.state, "client_id", None)
        if client_id: return client_id
        return f"ip:{request.client.host}" if request.client else "unknown"

    def _clean_window(self, client_id, now):
        cutoff = now - self.config.window_seconds
        self._windows[client_id] = [ts for ts in self._windows[client_id] if ts > cutoff]

    def check_rate_limit(self, request: Request) -> Optional[JSONResponse]:
        if not self.config.enabled or request.url.path in EXEMPT_ENDPOINTS:
            return None
        client_id = self._get_client_id(request)
        now = time.time()
        self._clean_window(client_id, now)
        current = len(self._windows[client_id])
        if current >= self._max_rate:
            oldest = self._windows[client_id][0] if self._windows[client_id] else now
            retry_after = int(self.config.window_seconds - (now - oldest)) + 1
            logger.warning(f"Rate limited '{client_id}': {current}/{self._max_rate}")
            return JSONResponse(status_code=429, content={"error": "rate_limit_exceeded", "detail": f"Max {self.config.requests_per_minute} req/min.", "retry_after_seconds": retry_after}, headers={"Retry-After": str(retry_after), "X-RateLimit-Limit": str(self.config.requests_per_minute), "X-RateLimit-Remaining": "0"})
        self._windows[client_id].append(now)
        return None

    def get_remaining(self, client_id):
        self._clean_window(client_id, time.time())
        return max(0, self._max_rate - len(self._windows[client_id]))

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limiter=None):
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter()
    async def dispatch(self, request, call_next):
        rejection = self.rate_limiter.check_rate_limit(request)
        if rejection: return rejection
        response = await call_next(request)
        if self.rate_limiter.config.enabled and request.url.path not in EXEMPT_ENDPOINTS:
            client_id = self.rate_limiter._get_client_id(request)
            response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.config.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(self.rate_limiter.get_remaining(client_id))
        return response
