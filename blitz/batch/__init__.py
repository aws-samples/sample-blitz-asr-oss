"""
AWS Batch processing module for Blitz ASR.

This module provides batch transcription support for running Blitz in AWS Batch,
enabling event-driven audio transcription triggered by S3 uploads.

Components:
    - S3Handler: Download/upload utilities for S3
    - BatchConfig: Batch-specific configuration
    - transcribe_s3_file: Main transcription workflow

Usage:
    python -m blitz.batch.transcribe_job --input-bucket BUCKET --input-key KEY

Environment Variables:
    OUTPUT_BUCKET_NAME: S3 bucket for results (required)
    OUTPUT_PREFIX: S3 key prefix for outputs (default: transcripts/)
    TIER2_BACKEND: nemo or onnx (default: nemo)
    TIER2_PRECISION: fp32, bfloat16, or fp16 (default: bfloat16)
    SKIP_SILENCE: Skip VAD-detected silent files (default: true)
    INCLUDE_TIMESTAMPS: Include timestamps in JSON output (default: true)
"""

from .s3_handler import S3Handler
from .batch_config import BatchConfig

__all__ = ["S3Handler", "BatchConfig"]
