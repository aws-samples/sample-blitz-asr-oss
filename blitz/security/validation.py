"""
Request validation for Blitz ASR.

Validates incoming requests for:
- File size limits (prevent memory exhaustion)
- Content type validation (only audio files)
- Filename sanitization (prevent path traversal)
- Audio format verification (check magic bytes)
"""

import os
import logging
from typing import Optional, Set

from fastapi import UploadFile, HTTPException

logger = logging.getLogger("blitz.security.validation")

# Allowed audio MIME types
ALLOWED_MIME_TYPES: Set[str] = {
    "audio/mpeg",          # MP3
    "audio/mp3",           # MP3 alternate
    "audio/wav",           # WAV
    "audio/x-wav",         # WAV alternate
    "audio/wave",          # WAV alternate
    "audio/flac",          # FLAC
    "audio/x-flac",       # FLAC alternate
    "audio/mp4",           # M4A
    "audio/x-m4a",        # M4A alternate
    "audio/m4a",           # M4A alternate
    "audio/ogg",           # OGG
    "audio/webm",          # WebM audio
    "application/octet-stream",  # Allow generic (check extension instead)
}

# Allowed file extensions
ALLOWED_EXTENSIONS: Set[str] = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}

# Audio file magic bytes (first few bytes)
MAGIC_BYTES = {
    b"RIFF": "wav",
    b"ID3": "mp3",
    b"\xff\xfb": "mp3",
    b"\xff\xf3": "mp3",
    b"\xff\xf2": "mp3",
    b"fLaC": "flac",
    b"\x00\x00\x00": "m4a",  # partial ftyp
    b"OggS": "ogg",
    b"\x1aE\xdf\xa3": "webm",
}


class RequestValidator:
    """Validates transcription requests for security and correctness."""

    def __init__(
        self,
        max_file_size_bytes: Optional[int] = None,
        allowed_extensions: Optional[Set[str]] = None,
    ):
        self.max_file_size_bytes = max_file_size_bytes or int(
            os.environ.get("MAX_UPLOAD_BYTES", 524_288_000)  # 500MB default
        )
        self.allowed_extensions = allowed_extensions or ALLOWED_EXTENSIONS

        logger.info(
            f"Request validator: max_file_size={self.max_file_size_bytes/(1024*1024):.0f}MB, "
            f"allowed_formats={sorted(self.allowed_extensions)}"
        )

    def validate_file(self, file: UploadFile, file_content: bytes) -> None:
        """
        Validate an uploaded audio file.

        Raises HTTPException with appropriate error codes on failure.
        """
        # 1. Check file size
        file_size = len(file_content)
        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded. Please provide an audio file with content.",
            )

        if file_size > self.max_file_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large: {file_size/(1024*1024):.1f}MB. "
                    f"Maximum allowed: {self.max_file_size_bytes/(1024*1024):.0f}MB."
                ),
            )

        # 2. Check file extension
        filename = file.filename or "unknown"
        ext = os.path.splitext(filename)[1].lower()
        if ext and ext not in self.allowed_extensions:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file format: '{ext}'. "
                    f"Allowed formats: {', '.join(sorted(self.allowed_extensions))}"
                ),
            )

        # 3. Check content type (lenient — many clients send wrong MIME types)
        content_type = file.content_type or "application/octet-stream"
        if content_type not in ALLOWED_MIME_TYPES and ext not in self.allowed_extensions:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported content type: '{content_type}'. "
                    f"Allowed formats: MP3, WAV, M4A, FLAC, OGG, WebM"
                ),
            )

        # 4. Verify magic bytes (detect disguised files)
        if file_size >= 4:
            is_valid_audio = False
            header = file_content[:4]
            for magic, _ in MAGIC_BYTES.items():
                if header[:len(magic)] == magic:
                    is_valid_audio = True
                    break

            # Don't hard-fail on magic bytes — some valid audio has unusual headers
            if not is_valid_audio and ext not in (".m4a",):
                logger.warning(
                    f"File '{filename}' has unrecognized magic bytes "
                    f"(header: {header.hex()}). Proceeding anyway."
                )

    def sanitize_filename(self, filename: Optional[str]) -> str:
        """Sanitize filename to prevent path traversal and injection."""
        if not filename:
            return "unnamed_audio"

        # Remove path components
        filename = os.path.basename(filename)

        # Remove dangerous characters
        safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        sanitized = "".join(c if c in safe_chars else "_" for c in filename)

        # Limit length
        if len(sanitized) > 255:
            sanitized = sanitized[:255]

        return sanitized or "unnamed_audio"
