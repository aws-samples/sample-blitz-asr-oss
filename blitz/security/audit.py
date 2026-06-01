"""
Audit logging for Blitz ASR.

Records every transcription request with:
- Client identity (API key client ID)
- Request metadata (timestamp, IP, file size, duration)
- Result metadata (language detected, tier used, processing time)
- File hash (SHA-256 of audio content — NOT the content itself)

Outputs structured JSON to a dedicated audit logger for compliance/governance.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger("blitz.audit")


@dataclass
class AuditEntry:
    """A single audit log entry for a transcription request."""

    # Request identity
    timestamp: float
    request_id: str
    client_id: str
    client_ip: str

    # Input metadata
    file_name: str
    file_size_bytes: int
    file_hash_sha256: str  # Hash of content, NOT the content
    audio_duration_sec: Optional[float] = None

    # Result metadata
    status: str = "success"  # success, error, rejected, rate_limited
    language_detected: Optional[str] = None
    engine_used: Optional[str] = None
    processing_time_sec: Optional[float] = None
    error_message: Optional[str] = None


def create_audit_entry(
    request_id: str,
    client_id: str,
    client_ip: str,
    file_name: str,
    file_content: bytes,
) -> AuditEntry:
    """Create a new audit entry with file hash."""
    return AuditEntry(
        timestamp=time.time(),
        request_id=request_id,
        client_id=client_id or "anonymous",
        client_ip=client_ip,
        file_name=file_name,
        file_size_bytes=len(file_content),
        file_hash_sha256=hashlib.sha256(file_content).hexdigest(),
    )


def log_audit_entry(entry: AuditEntry):
    """Write audit entry to the audit log."""
    logger.info(
        f"AUDIT: client={entry.client_id} status={entry.status} "
        f"engine={entry.engine_used} lang={entry.language_detected} "
        f"duration={entry.audio_duration_sec}s file={entry.file_name}",
        extra=asdict(entry),
    )
