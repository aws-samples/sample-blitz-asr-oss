"""
Centralized configuration for Blitz transcription service.

All configuration is managed through environment variables with sensible defaults.
Uses Pydantic for type safety and validation.
"""

from typing import Dict, Set
from pydantic import Field, field_validator

# Try to import BaseSettings from pydantic-settings (Pydantic v2)
# Fall back to BaseModel if not available
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _MODEL_CONFIG = SettingsConfigDict(
        env_prefix="",  # Read directly from env vars
        case_sensitive=False,
        extra="ignore",  # Ignore unknown env vars
    )
except ImportError:
    from pydantic import BaseModel as BaseSettings

    _MODEL_CONFIG = {}
    import warnings

    warnings.warn(
        "pydantic-settings not installed; falling back to BaseModel "
        "(env var reading disabled)",
        stacklevel=2,
    )


class ServerConfig(BaseSettings):
    """Server configuration with validation and type safety."""

    # ============================================================================
    # Micro-Batching Configuration
    # ============================================================================

    batch_timeout_ms: float = Field(
        default=15.0,
        description="Wait up to this many milliseconds to gather a batch (0-15ms latency, 2x+ throughput)",
        ge=0.0,
        le=1000.0,
    )

    max_batch_size: int = Field(
        default=4,
        description="Maximum number of requests to batch together (higher = better GPU utilization but more memory)",
        ge=1,
        le=16,
    )

    # ============================================================================
    # Chunking Configuration for Long Audio
    # ============================================================================

    chunk_length_sec: int = Field(
        default=480,
        description="Chunk length in seconds for long audio (8 minutes default to prevent OOM on g6.xlarge)",
        ge=60,
        le=900,
    )

    chunk_overlap_sec: int = Field(
        default=1,
        description="Overlap between chunks in seconds to prevent word cutoffs",
        ge=0,
        le=10,
    )

    # ============================================================================
    # Language Detection Configuration
    # ============================================================================

    language_detection_model: str = Field(
        default="large-v3-turbo",
        description="Model to use for language detection: 'tiny' (saves ~8GB) or 'large-v3-turbo' (better accuracy, already loaded)",
    )

    primary_language_detector: str = Field(
        default="whisper",
        description="Primary language detection model: 'whisper' (GPU, more accurate) or 'voxlingua' (CPU, frees GPU)",
    )

    secondary_language_detector: str = Field(
        default="voxlingua",
        description="Secondary language detection model for low-confidence fallback",
    )

    voxlingua_confidence_threshold: float = Field(
        default=0.90,
        description="Minimum VoxLingua confidence to skip fallback (VoxLingua tends to produce lower confidence scores)",
        ge=0.0,
        le=1.0,
    )

    whisper_confidence_threshold: float = Field(
        default=0.70,
        description="Minimum Whisper confidence to skip fallback",
        ge=0.0,
        le=1.0,
    )

    # Legacy field for backward compatibility
    min_language_confidence: float = Field(
        default=0.70,
        description="[DEPRECATED] Use voxlingua_confidence_threshold or whisper_confidence_threshold instead",
        ge=0.0,
        le=1.0,
    )

    # ============================================================================
    # Efficiency Thresholds
    # ============================================================================

    short_audio_threshold_sec: int = Field(
        default=300,
        description="Audio shorter than this uses direct array processing (fastest, lowest memory)",
        ge=60,
        le=3600,
    )

    local_attention_threshold_sec: int = Field(
        default=1440,
        description="Audio longer than this switches to local attention mode (prevents OOM). Recommended: A10G=1440s, A100=3600s",
        ge=300,
        le=7200,
    )

    # Device configuration
    device: str = Field(
        default="cuda", description="Device to use for inference ('cuda' or 'cpu')"
    )

    # ============================================================================
    # Tier 1: Fastest (default: CTC-110M for English)
    # ============================================================================

    tier1_model: str = Field(
        default="nvidia/parakeet-tdt_ctc-110m",
        description="Tier 1 model (fastest). Empty string disables Tier 1.",
    )

    tier1_precision: str = Field(
        default="fp16",
        description="Tier 1 precision: 'fp32' or 'fp16'",
    )

    tier1_languages: str = Field(
        default="en",
        description="Languages routed to Tier 1 (comma-separated ISO 639-1). Empty string disables Tier 1.",
    )

    # ============================================================================
    # Tier 2: Fast Multilingual (default: Parakeet 0.6B for EU languages)
    # ============================================================================

    tier2_model: str = Field(
        default="nvidia/parakeet-tdt-0.6b-v3",
        description="Tier 2 model (fast multilingual)",
    )

    tier2_precision: str = Field(
        default="bfloat16",
        description="Tier 2 precision: 'fp32', 'bfloat16', or 'fp16'",
    )

    tier2_backend: str = Field(
        default="nemo",
        description="Tier 2 backend: 'nemo' (PyTorch/NeMo) or 'onnx' (ONNX Runtime)",
    )

    tier2_languages: str = Field(
        default="bg,hr,cs,da,nl,et,fi,fr,de,el,hu,it,lv,lt,mt,pl,pt,ro,sk,sl,es,sv,ru,uk",
        description="Languages routed to Tier 2 (comma-separated ISO 639-1, if not in Tier 1)",
    )

    tier2_onnx_model_path: str = Field(
        default="./models/parakeet",
        description="Path to ONNX model files (for tier2_backend='onnx')",
    )

    tier2_auto_download: bool = Field(
        default=True,
        description="Auto-download FP16 ONNX models from HuggingFace if missing",
    )

    # ============================================================================
    # Tier 3: Catch-all (default: Whisper for everything else)
    # ============================================================================

    tier3_model: str = Field(
        default="openai/whisper-large-v3-turbo",
        description="Tier 3 model (catch-all fallback for languages not in Tier 1 or 2)",
    )

    whisper_tiny_model_name: str = Field(
        default="openai/whisper-tiny",
        description="Whisper tiny model identifier (for language detection)",
    )

    voxlingua_model_name: str = Field(
        default="speechbrain/lang-id-voxlingua107-ecapa",
        description="VoxLingua107 model identifier for fallback language detection",
    )

    # ============================================================================
    # Model Revision Pinning (supply-chain hardening)
    # ============================================================================
    # Empty string = fetch latest from hub (convenient for dev).
    # Set to a commit SHA or tag in production to get reproducible, auditable model loads.

    tier1_model_revision: str = Field(
        default="",
        description="HuggingFace revision (commit SHA or tag) for tier1_model. Empty = latest.",
    )

    tier2_model_revision: str = Field(
        default="",
        description="HuggingFace revision for tier2_model (NeMo or ONNX repo). Empty = latest.",
    )

    tier3_model_revision: str = Field(
        default="",
        description="HuggingFace revision for tier3_model (Whisper primary). Empty = latest.",
    )

    whisper_tiny_revision: str = Field(
        default="",
        description="HuggingFace revision for whisper_tiny_model_name. Empty = latest.",
    )

    voxlingua_revision: str = Field(
        default="",
        description="HuggingFace revision for voxlingua_model_name. Empty = latest.",
    )

    silero_vad_revision: str = Field(
        default="",
        description="torch.hub commit SHA for snakers4/silero-vad. Empty = latest on master.",
    )

    # ============================================================================
    # GPU Instance Configuration
    # ============================================================================

    instance_type: str = Field(
        default="g6.xlarge", description="AWS EC2 instance type for cost calculations"
    )

    instance_region: str = Field(
        default="us-east-1", description="AWS region for cost calculations"
    )

    # Hourly GPU costs (USD) for g6.xlarge in us-east-1.
    # Source: AWS Pricing API (https://aws.amazon.com/ec2/pricing/reserved-instances/pricing/),
    # Linux/Shared/No Upfront. Last refreshed 2026-04-29 (API publicationDate).
    # Refresh with: python scripts/update_pricing.py
    # Spot is real-time and volatile; the default below is the upper end of a
    # recent 7-day us-east-1 sample; override per deployment as needed.
    gpu_price_on_demand: float = Field(
        default=0.8048, description="On-Demand hourly rate (USD)", ge=0.0
    )

    gpu_price_spot: float = Field(
        default=0.42,
        description="Spot instance hourly rate (USD, upper end of recent us-east-1 range)",
        ge=0.0,
    )

    gpu_price_savings_1yr: float = Field(
        default=0.5239,
        description="1-year Standard RI, No Upfront (USD/hr)",
        ge=0.0,
    )

    gpu_price_savings_3yr: float = Field(
        default=0.3694,
        description="3-year Standard RI, No Upfront (USD/hr)",
        ge=0.0,
    )

    # ============================================================================
    # Batch Mode Configuration
    # ============================================================================

    batch_mode: bool = Field(
        default=False,
        description="Running in AWS Batch mode (disables async queues, uses synchronous processing)",
    )

    max_upload_bytes: int = Field(
        default=500 * 1024 * 1024,
        description="Maximum accepted /transcribe upload body size in bytes",
        ge=1024,
    )

    # Pydantic v2 configuration for reading environment variables
    model_config = _MODEL_CONFIG

    @field_validator("language_detection_model")
    @classmethod
    def validate_language_detection_model(cls, v: str) -> str:
        """Validate language detection model choice."""
        v = v.lower()
        if v not in ["tiny", "large-v3-turbo"]:
            raise ValueError(
                f"language_detection_model must be 'tiny' or 'large-v3-turbo', got: {v}"
            )
        return v

    @field_validator("primary_language_detector", "secondary_language_detector")
    @classmethod
    def validate_language_detector(cls, v: str) -> str:
        """Validate language detector choice."""
        v = v.lower()
        if v not in ["voxlingua", "whisper"]:
            raise ValueError(
                f"language detector must be 'voxlingua' or 'whisper', got: {v}"
            )
        return v

    @field_validator("tier2_backend")
    @classmethod
    def validate_tier2_backend(cls, v: str) -> str:
        """Validate Tier 2 backend choice."""
        v = v.lower()
        if v not in ["nemo", "onnx"]:
            raise ValueError(f"tier2_backend must be 'nemo' or 'onnx', got: {v}")
        return v

    @field_validator("tier2_precision")
    @classmethod
    def validate_tier2_precision(cls, v: str) -> str:
        """Validate Tier 2 precision choice."""
        v = v.lower()
        if v not in ["fp32", "bfloat16", "fp16"]:
            raise ValueError(
                f"tier2_precision must be 'fp32', 'bfloat16', or 'fp16', got: {v}"
            )
        return v

    @field_validator("tier1_precision")
    @classmethod
    def validate_tier1_precision(cls, v: str) -> str:
        """Validate Tier 1 precision choice."""
        v = v.lower()
        if v not in ["fp32", "fp16"]:
            raise ValueError(f"tier1_precision must be 'fp32' or 'fp16', got: {v}")
        return v

    def get_tier1_languages(self) -> Set[str]:
        """Parse Tier 1 languages into a set. Empty string returns empty set."""
        if not self.tier1_languages.strip():
            return set()
        return {
            lang.strip() for lang in self.tier1_languages.split(",") if lang.strip()
        }

    def get_tier2_languages(self) -> Set[str]:
        """Parse Tier 2 languages into a set. Empty string returns empty set."""
        if not self.tier2_languages.strip():
            return set()
        return {
            lang.strip() for lang in self.tier2_languages.split(",") if lang.strip()
        }

    def get_gpu_pricing(self) -> Dict[str, float]:
        """Get GPU pricing dictionary for cost calculations."""
        return {
            "on_demand": self.gpu_price_on_demand,
            "spot": self.gpu_price_spot,
            "savings_1yr": self.gpu_price_savings_1yr,
            "savings_3yr": self.gpu_price_savings_3yr,
        }


# ============================================================================
# ISO 639-3 to ISO 639-1 Language Code Mapping
# ============================================================================
# VoxLingua107 uses ISO 639-3 codes (3-letter), but we need ISO 639-1 (2-letter)

VOXLINGUA_TO_ISO639_1: Dict[str, str] = {
    # Major European languages
    "eng": "en",
    "fra": "fr",
    "deu": "de",
    "spa": "es",
    "ita": "it",
    "por": "pt",
    "nld": "nl",
    "pol": "pl",
    "rus": "ru",
    "ukr": "uk",
    "ces": "cs",
    "slk": "sk",
    "ron": "ro",
    "bul": "bg",
    "hrv": "hr",
    "slv": "sl",
    "lit": "lt",
    "lav": "lv",
    "est": "et",
    "hun": "hu",
    "ell": "el",
    "fin": "fi",
    "swe": "sv",
    "dan": "da",
    "nor": "no",
    # Celtic languages
    "cym": "cy",  # Welsh - Key mapping for Welsh language support!
    "gle": "ga",  # Irish
    "gla": "gd",  # Scottish Gaelic
    "bre": "br",  # Breton
    # Asian languages
    "cmn": "zh",
    "zho": "zh",  # Chinese (Mandarin)
    "jpn": "ja",
    "kor": "ko",
    "tha": "th",
    "vie": "vi",
    "ind": "id",
    "msa": "ms",
    "tgl": "tl",
    # Other major languages
    "ara": "ar",
    "tur": "tr",
    "heb": "he",
    "fas": "fa",
    "hin": "hi",
    "ben": "bn",
    "tam": "ta",
    "tel": "te",
    "urd": "ur",
    "kan": "kn",
    "mal": "ml",
    "mar": "mr",
    "guj": "gu",
    "pan": "pa",
    # African languages
    "swa": "sw",
    "afr": "af",
    "amh": "am",
    "hau": "ha",
    "yor": "yo",
    "som": "so",
    # Other European
    "cat": "ca",
    "eus": "eu",
    "glg": "gl",
    "isl": "is",
    "mlt": "mt",
    "sqi": "sq",
    "mkd": "mk",
    "bos": "bs",
    "srp": "sr",
    "bel": "be",
    "aze": "az",
    "kat": "ka",
    "hye": "hy",
    "kaz": "kk",
    "uzb": "uz",
    "tgk": "tg",
    "mon": "mn",
    "nep": "ne",
    "sin": "si",
    "khm": "km",
    "lao": "lo",
    "mya": "my",
    "bod": "bo",
    # Additional
    "lat": "la",
    "yid": "yi",
    "oci": "oc",
    "fao": "fo",
    "hat": "ht",
    "pus": "ps",
    "tuk": "tk",
    "nno": "nn",
    "asm": "as",
    "tat": "tt",
    "lin": "ln",
    "bak": "ba",
    "jav": "jw",
    "sun": "su",
    "mlg": "mg",
    "san": "sa",
    "ltz": "lb",
    "sna": "sn",
}


# ============================================================================
# Global Configuration Instance
# ============================================================================

# Singleton configuration instance
config = ServerConfig()
