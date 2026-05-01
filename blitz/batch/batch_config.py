"""
Batch-specific configuration for AWS Batch processing.

Extends the base ServerConfig with batch-specific settings for S3 I/O,
output formatting, and batch processing optimization.
"""

import os

from pydantic import Field

# Import base config
from blitz.config import ServerConfig


class BatchConfig(ServerConfig):
    """
    Extended configuration for AWS Batch processing.

    Inherits all settings from ServerConfig and adds batch-specific
    configuration for S3 operations and output formatting.
    """

    # =========================================================================
    # S3 Configuration
    # =========================================================================

    output_bucket: str = Field(
        default="", description="S3 bucket for transcription results"
    )

    output_prefix: str = Field(
        default="transcripts/", description="S3 key prefix for output files"
    )

    # =========================================================================
    # Output Options
    # =========================================================================

    include_timestamps: bool = Field(
        default=True, description="Include segment timestamps in JSON output"
    )

    include_word_timestamps: bool = Field(
        default=False, description="Include word-level timestamps (slower, more data)"
    )

    include_metrics: bool = Field(
        default=True,
        description="Include processing metrics and cost estimates in output",
    )

    # =========================================================================
    # Processing Options
    # =========================================================================

    skip_silence: bool = Field(
        default=True, description="Skip processing for VAD-detected silent/empty files"
    )

    silence_threshold: float = Field(
        default=0.10,
        description="Files with speech ratio below this are considered silent",
        ge=0.0,
        le=1.0,
    )

    # =========================================================================
    # Batch Mode Flag
    # =========================================================================

    batch_mode: bool = Field(
        default=True, description="Running in AWS Batch mode (disables async queues)"
    )

    @classmethod
    def from_environment(cls) -> "BatchConfig":
        """
        Create BatchConfig from environment variables.

        Reads all standard ServerConfig env vars plus batch-specific vars:
            - OUTPUT_BUCKET_NAME: S3 bucket for results
            - OUTPUT_PREFIX: S3 key prefix (default: transcripts/)
            - SKIP_SILENCE: Skip silent files (default: true)
            - INCLUDE_TIMESTAMPS: Include timestamps (default: true)
            - INCLUDE_METRICS: Include metrics (default: true)

        Returns:
            BatchConfig instance with values from environment
        """
        # Read batch-specific env vars
        output_bucket = os.environ.get("OUTPUT_BUCKET_NAME", "")
        output_prefix = os.environ.get("OUTPUT_PREFIX", "transcripts/")
        skip_silence = os.environ.get("SKIP_SILENCE", "true").lower() == "true"
        include_timestamps = (
            os.environ.get("INCLUDE_TIMESTAMPS", "true").lower() == "true"
        )
        include_metrics = os.environ.get("INCLUDE_METRICS", "true").lower() == "true"
        include_word_timestamps = (
            os.environ.get("INCLUDE_WORD_TIMESTAMPS", "false").lower() == "true"
        )
        silence_threshold = float(os.environ.get("SILENCE_THRESHOLD", "0.10"))

        return cls(
            output_bucket=output_bucket,
            output_prefix=output_prefix,
            skip_silence=skip_silence,
            silence_threshold=silence_threshold,
            include_timestamps=include_timestamps,
            include_word_timestamps=include_word_timestamps,
            include_metrics=include_metrics,
            batch_mode=True,
        )

    def validate_batch_config(self) -> None:
        """
        Validate batch-specific configuration.

        Raises:
            ValueError: If required configuration is missing
        """
        if not self.output_bucket:
            raise ValueError(
                "OUTPUT_BUCKET_NAME environment variable is required. "
                "Set it to the S3 bucket for transcription results."
            )

    def get_output_key(self, input_key: str) -> str:
        """
        Generate output key from input key.

        Args:
            input_key: S3 key of input audio file

        Returns:
            S3 key for output JSON file

        Example:
            input_key: "audio/interview.mp3"
            output_key: "transcripts/interview.json"
        """
        # Extract filename without extension
        filename = os.path.basename(input_key)
        name_without_ext = os.path.splitext(filename)[0]

        # Ensure prefix ends with /
        prefix = self.output_prefix
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        return f"{prefix}{name_without_ext}.json"
