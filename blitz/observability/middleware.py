"""
FastAPI middleware for automatic observability instrumentation.

Automatically adds to every request:
- Unique request ID (X-Request-ID header)
- Request duration tracking
- Prometheus metrics recording
- Structured log entries (start/end of request)
- GPU metrics refresh (periodic)
"""

import time
import uuid
import logging
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from blitz.observability.metrics import METRICS, update_gpu_metrics
from blitz.observability.logging_config import set_request_context, clear_request_context

logger = logging.getLogger("blitz.middleware")

# Track GPU metrics refresh interval
_last_gpu_update = 0.0
_GPU_UPDATE_INTERVAL = 15.0  # seconds


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Middleware that instruments all requests with metrics and logging."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        global _last_gpu_update

        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        
        # Set request context for structured logging
        set_request_context(
            request_id=request_id,
            metadata={
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
            },
        )

        # Track in-progress requests
        METRICS.requests_in_progress.inc()
        start_time = time.time()

        # Log request start (only for transcribe endpoint to reduce noise)
        if request.url.path == "/transcribe":
            logger.info(
                f"Request started: {request.method} {request.url.path}",
                extra={"client_ip": request.client.host if request.client else "unknown"},
            )

        try:
            response = await call_next(request)
            duration = time.time() - start_time

            # Record metrics
            METRICS.requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code,
            ).inc()

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{duration:.3f}s"

            # Log request completion (for transcribe endpoint)
            if request.url.path == "/transcribe":
                logger.info(
                    f"Request completed: {response.status_code} in {duration:.3f}s",
                    extra={"processing_time_sec": duration},
                )

            return response

        except Exception as exc:
            duration = time.time() - start_time

            METRICS.requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=500,
            ).inc()

            METRICS.transcription_errors_total.labels(
                engine="unknown",
                error_type=type(exc).__name__,
            ).inc()

            logger.error(
                f"Request failed: {type(exc).__name__}: {str(exc)[:200]}",
                extra={"error_type": type(exc).__name__, "processing_time_sec": duration},
                exc_info=True,
            )
            raise

        finally:
            METRICS.requests_in_progress.dec()

            # Periodically update GPU metrics
            now = time.time()
            if now - _last_gpu_update > _GPU_UPDATE_INTERVAL:
                _last_gpu_update = now
                update_gpu_metrics()

            # Clear request context
            clear_request_context()
