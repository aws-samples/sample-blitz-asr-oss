import hashlib, logging, time
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger("blitz.audit")

@dataclass
class AuditEntry:
    timestamp: float
    request_id: str
    client_id: str
    client_ip: str
    file_name: str
    file_size_bytes: int
    file_hash_sha256: str
    audio_duration_sec: Optional[float] = None
    status: str = "success"
    language_detected: Optional[str] = None
    engine_used: Optional[str] = None
    processing_time_sec: Optional[float] = None

def create_audit_entry(request_id, client_id, client_ip, file_name, file_content):
    return AuditEntry(timestamp=time.time(), request_id=request_id, client_id=client_id or "anonymous", client_ip=client_ip, file_name=file_name, file_size_bytes=len(file_content), file_hash_sha256=hashlib.sha256(file_content).hexdigest())

def log_audit_entry(entry):
    logger.info(f"AUDIT: client={entry.client_id} status={entry.status} engine={entry.engine_used} lang={entry.language_detected} file={entry.file_name}", extra=asdict(entry))
