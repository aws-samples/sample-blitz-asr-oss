"""
API request and response schemas using Pydantic.

Provides type safety, validation, and automatic OpenAPI documentation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Common Models
# ============================================================================


class LanguageInfo(BaseModel):
    """Language detection information."""

    code: str = Field(
        ...,
        description="ISO 639-1 language code (e.g., 'en', 'zh', 'cy')",
        min_length=2,
        max_length=2,
    )
    prob: float = Field(
        ..., description="Confidence probability (0.0 to 1.0)", ge=0.0, le=1.0
    )


class ProcessingMetrics(BaseModel):
    """GPU processing and cost metrics."""

    processing_time_seconds: float = Field(
        ..., description="Total processing time in seconds", ge=0.0
    )
    audio_duration_seconds: float = Field(
        ..., description="Audio duration in seconds", ge=0.0
    )
    real_time_factor: float = Field(
        ..., description="Processing time / audio duration (lower is better)", ge=0.0
    )
    cost_per_1000_min_audio_usd: Dict[str, float] = Field(
        ...,
        description="Cost estimates per 1000 minutes of audio (on_demand, spot, savings_1yr, savings_3yr)",
    )
    cost_breakdown_percent: Dict[str, float] = Field(
        ...,
        description="Percentage breakdown of processing time (language_detection, transcription)",
    )
    instance_type: str = Field(default="g6.xlarge", description="AWS EC2 instance type")
    region: str = Field(default="us-east-1", description="AWS region")


class MetricsInfo(BaseModel):
    """Detailed metrics about processing pipeline."""

    device: str = Field(
        ..., description="Processing device (gpu or cpu)", pattern="^(gpu|cpu)$"
    )
    lang_detect_time_ms: float = Field(
        ..., description="Language detection time in milliseconds", ge=0.0
    )
    lang_detect_method: str = Field(
        ...,
        description="Language detection method used (e.g., 'whisper', 'whisper+voxlingua')",
    )
    lang_detect_model: str = Field(
        ...,
        description="Language detection model used (e.g., 'large-v3-turbo', 'large-v3-turbo+voxlingua107')",
    )
    whisper_lang_confidence: Optional[float] = Field(
        None,
        description="Whisper language detection confidence (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    voxlingua_lang_confidence: Optional[float] = Field(
        None,
        description="VoxLingua107 language detection confidence (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    whisper_detect_time_ms: float = Field(
        ..., description="Whisper language detection time in milliseconds", ge=0.0
    )
    voxlingua_detect_time_ms: float = Field(
        ..., description="VoxLingua107 language detection time in milliseconds", ge=0.0
    )
    transcribe_time_seconds: float = Field(
        ..., description="Transcription time in seconds", ge=0.0
    )
    processing_metrics: ProcessingMetrics = Field(
        ..., description="GPU processing and cost metrics"
    )


class TimestampSegment(BaseModel):
    """Timestamp segment for SRT/VTT generation."""

    start: float = Field(..., description="Segment start time in seconds", ge=0.0)
    end: float = Field(..., description="Segment end time in seconds", ge=0.0)
    segment: str = Field(..., description="Segment text content", min_length=1)


# ============================================================================
# Response Models
# ============================================================================


class TranscriptionResponse(BaseModel):
    """Full JSON transcription response with metadata."""

    filename: str = Field(..., description="Original audio filename")
    language: str = Field(
        ...,
        description="Detected language (ISO 639-1 code)",
        min_length=2,
        max_length=2,
    )
    language_confidence: float = Field(
        ..., description="Language detection confidence (0.0 to 1.0)", ge=0.0, le=1.0
    )
    engine: str = Field(
        ..., description="Transcription engine used (parakeet or whisper)"
    )
    duration_sec: float = Field(..., description="Audio duration in seconds", ge=0.0)
    top_languages: List[LanguageInfo] = Field(
        ..., description="Top 3 detected languages with confidence scores"
    )
    text: str = Field(..., description="Transcription text")
    chunk_approach: str = Field(
        default="direct", description="Processing approach (direct, chunked, temp_file)"
    )
    chunk_count: int = Field(default=1, description="Number of chunks processed", ge=1)
    chunk_size_sec: Optional[int] = Field(
        None, description="Chunk size in seconds (if chunked)", ge=1
    )
    metrics: MetricsInfo = Field(..., description="Detailed processing metrics")
    timestamps: Optional[Dict] = Field(
        None, description="Timestamp data with segment-level timing"
    )
    transcript: Optional[str] = Field(
        None, description="Formatted transcript with timestamps (if available)"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "filename": "audio.wav",
                "language": "en",
                "language_confidence": 0.9876,
                "engine": "parakeet",
                "duration_sec": 15.3,
                "top_languages": [
                    {"code": "en", "prob": 0.9876},
                    {"code": "es", "prob": 0.0054},
                    {"code": "fr", "prob": 0.0032},
                ],
                "text": "Hello world, this is a test transcription.",
                "chunk_approach": "direct",
                "chunk_count": 1,
                "metrics": {
                    "device": "gpu",
                    "lang_detect_time_ms": 120.5,
                    "lang_detect_method": "whisper",
                    "lang_detect_model": "large-v3-turbo",
                    "whisper_lang_confidence": 0.9876,
                    "voxlingua_lang_confidence": None,
                    "whisper_detect_time_ms": 120.5,
                    "voxlingua_detect_time_ms": 0.0,
                    "transcribe_time_seconds": 0.45,
                    "processing_metrics": {
                        "processing_time_seconds": 0.58,
                        "audio_duration_seconds": 15.3,
                        "real_time_factor": 0.038,
                        "cost_per_1000_min_audio_usd": {
                            "on_demand": 0.51,
                            "spot": 0.33,
                            "savings_1yr": 0.33,
                            "savings_3yr": 0.29,
                        },
                        "cost_breakdown_percent": {
                            "language_detection": 20.8,
                            "transcription": 77.6,
                        },
                        "instance_type": "g6.xlarge",
                        "region": "us-east-1",
                    },
                },
            }
        }


class SRTResponse(BaseModel):
    """SRT subtitle format response."""

    text: str = Field(..., description="SRT formatted subtitle content")
    format: str = Field(
        default="srt", description="Response format identifier", pattern="^srt$"
    )
    filename: str = Field(..., description="Original audio filename")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "text": "1\n00:00:00,000 --> 00:00:02,500\nHello world\n\n2\n00:00:02,500 --> 00:00:05,000\nHow are you?\n\n",
                "format": "srt",
                "filename": "audio.wav",
            }
        }


class HealthResponse(BaseModel):
    """Health check endpoint response."""

    status: str = Field(default="healthy", description="Health status")
    models: Dict[str, Dict[str, Any]] = Field(
        ..., description="Loaded models information"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "status": "healthy",
                "models": {
                    "parakeet": {"loaded": True, "device": "cuda"},
                    "whisper": {"loaded": True, "device": "cuda"},
                    "voxlingua": {"loaded": True, "device": "cuda"},
                },
            }
        }


class ModelsInfoResponse(BaseModel):
    """Models information endpoint response."""

    parakeet: Dict[str, Any] = Field(..., description="Parakeet model information")
    whisper: Dict[str, Any] = Field(..., description="Whisper model information")
    voxlingua: Dict[str, Any] = Field(..., description="VoxLingua107 model information")
    language_routing: Dict[str, Any] = Field(
        ..., description="Language routing configuration"
    )
    batch_config: Dict[str, Any] = Field(
        ..., description="Micro-batching configuration"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "parakeet": {
                    "model": "nvidia/parakeet-tdt-0.6b-v3",
                    "supported_languages": ["bg", "hr", "cs", "da", "nl", "en"],
                },
                "whisper": {
                    "model": "openai/whisper-large-v3-turbo",
                    "supported_languages": ["en", "zh", "de", "es", "ru", "ko"],
                },
                "voxlingua": {
                    "model": "speechbrain/lang-id-voxlingua107-ecapa",
                    "languages": 107,
                    "purpose": "fallback_language_detection",
                },
                "language_routing": {
                    "parakeet_count": 25,
                    "whisper_count": 99,
                    "total_supported": 99,
                },
                "batch_config": {"batch_timeout_ms": 15.0, "max_batch_size": 4},
            }
        }
