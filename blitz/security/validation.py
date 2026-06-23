import os, logging
from typing import Optional, Set
from fastapi import UploadFile, HTTPException

logger = logging.getLogger("blitz.security.validation")
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}
ALLOWED_MIME_TYPES = {"audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave", "audio/flac", "audio/x-flac", "audio/mp4", "audio/x-m4a", "audio/m4a", "audio/ogg", "audio/webm", "application/octet-stream"}

class RequestValidator:
    def __init__(self, max_file_size_bytes=None):
        self.max_file_size_bytes = max_file_size_bytes or int(os.environ.get("MAX_UPLOAD_BYTES", 524288000))
        logger.info(f"Request validator: max_size={self.max_file_size_bytes/(1024*1024):.0f}MB")

    def validate_file(self, file: UploadFile, file_content: bytes):
        file_size = len(file_content)
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file.")
        if file_size > self.max_file_size_bytes:
            raise HTTPException(status_code=413, detail=f"File too large: {file_size/(1024*1024):.1f}MB. Max: {self.max_file_size_bytes/(1024*1024):.0f}MB.")
        filename = file.filename or "unknown"
        ext = os.path.splitext(filename)[1].lower()
        if ext and ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=415, detail=f"Unsupported format: '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    def sanitize_filename(self, filename: Optional[str]) -> str:
        if not filename: return "unnamed_audio"
        filename = os.path.basename(filename)
        safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        return "".join(c if c in safe else "_" for c in filename)[:255] or "unnamed_audio"
