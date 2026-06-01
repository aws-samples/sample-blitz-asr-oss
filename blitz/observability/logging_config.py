"""
Structured JSON logging for Blitz ASR.

Outputs JSON-formatted logs compatible with:
- AWS CloudWatch Logs
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Datadog, Splunk, etc.

Each log entry includes:
- timestamp, level, message
- request_id (correlation ID for tracing)
- service metadata (instance_type, region)
- request-specific fields (audio_duration, language, engine, etc.)
"""

import logging
import json
import sys
import os
from datetime import datetime, timezone
from typing import Optional
from contextvars import ContextVar

# Context variable for request correlation ID
_request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_request_metadata_var: ContextVar[dict] = ContextVar("request_metadata", default={})


def set_request_context(request_id: str, metadata: Optional[dict] = None):
    """Set the current request context for structured logging."""
    _request_id_var.set(request_id)
    _request_metadata_var.set(metadata or {})


def clear_request_context():
    """Clear the current request context."""
    _request_id_var.set(None)
    _request_metadata_var.set({})


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def __init__(self):
        super().__init__()
        self.service_name = "blitz-asr"
        self.instance_type = os.environ.get("INSTANCE_TYPE", "g6.xlarge")
        self.region = os.environ.get("INSTANCE_REGION", "us-east-1")
        self.hostname = os.environ.get("HOSTNAME", "unknown")

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "instance_type": self.instance_type,
            "region": self.region,
            "hostname": self.hostname,
        }

        # Add request correlation ID if available
        request_id = _request_id_var.get()
        if request_id:
            log_entry["request_id"] = request_id

        # Add request metadata if available
        metadata = _request_metadata_var.get()
        if metadata:
            log_entry.update(metadata)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            log_entry["exception_type"] = (
                record.exc_info[0].__name__ if record.exc_info[0] else None
            )

        # Add any extra fields passed via `extra={}`
        for key in ["audio_duration", "language", "engine", "rtf", "confidence",
                    "error_type", "file_name", "file_size_mb", "client_ip",
                    "processing_time_sec", "tier", "vad_result", "batch_size"]:
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry, default=str)


def setup_logging(log_level: str = "INFO", json_format: bool = True):
    """
    Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_format: If True, output JSON logs. If False, use human-readable format.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        # Human-readable format for local development
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root_logger.addHandler(handler)

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("nemo").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("speechbrain").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    return root_logger


def get_request_logger(name: str = "blitz.request") -> logging.Logger:
    """Get a logger configured for request-level structured logging."""
    return logging.getLogger(name)
