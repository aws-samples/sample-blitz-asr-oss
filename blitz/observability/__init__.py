"""
Observability module for Blitz ASR.

Provides:
- Prometheus metrics (request counts, latencies, GPU metrics, tier distribution)
- Structured logging (JSON format for CloudWatch/ELK ingestion)
- Request tracing (correlation IDs)
"""

from blitz.observability.metrics import setup_metrics, METRICS
from blitz.observability.logging_config import setup_logging, get_request_logger
from blitz.observability.middleware import ObservabilityMiddleware

__all__ = [
    "setup_metrics",
    "METRICS",
    "setup_logging",
    "get_request_logger",
    "ObservabilityMiddleware",
]
