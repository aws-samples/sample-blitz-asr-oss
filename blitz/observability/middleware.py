import time, uuid, logging
from starlette.middleware.base import BaseHTTPMiddleware
from blitz.observability.metrics import METRICS, update_gpu_metrics
from blitz.observability.logging_config import set_request_context, clear_request_context

logger = logging.getLogger("blitz.middleware")
_last_gpu_update = 0.0

class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        global _last_gpu_update
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        set_request_context(request_id)
        METRICS.requests_in_progress.inc()
        start = time.time()
        try:
            response = await call_next(request)
            METRICS.requests_total.labels(method=request.method, endpoint=request.url.path, status_code=response.status_code).inc()
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{time.time()-start:.3f}s"
            return response
        except Exception as exc:
            METRICS.transcription_errors_total.labels(engine="unknown", error_type=type(exc).__name__).inc()
            raise
        finally:
            METRICS.requests_in_progress.dec()
            now = time.time()
            if now - _last_gpu_update > 15.0:
                _last_gpu_update = now
                update_gpu_metrics()
            clear_request_context()
