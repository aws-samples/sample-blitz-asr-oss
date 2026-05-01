"""
Language detection service with configurable primary/secondary detectors.

Provides accurate language detection with:
- Configurable primary detector (VoxLingua CPU or Whisper GPU)
- Automatic fallback to secondary detector for low-confidence cases
- Per-model confidence thresholds
- Top-K language ranking for observability
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import time
import torch
import numpy as np

from blitz.models.whisper import WhisperModel
from blitz.models.voxlingua import VoxLinguaModel


class DetectionResult:
    """Language detection result with metadata."""

    def __init__(
        self,
        language: str,
        confidence: float,
        top_languages: Dict[str, float],
        detection_time_ms: float,
        detection_method: str,
        detection_model: str,
        whisper_confidence: Optional[float] = None,
        voxlingua_confidence: Optional[float] = None,
        whisper_time_ms: float = 0.0,
        voxlingua_time_ms: float = 0.0,
    ):
        """
        Initialize detection result.

        Args:
            language: ISO 639-1 language code (e.g., "en", "cy")
            confidence: Detection confidence (0.0 to 1.0)
            top_languages: Dictionary of top-K languages with probabilities
            detection_time_ms: Total detection time in milliseconds
            detection_method: Detection method used ("hint", "voxlingua", "whisper",
                            "voxlingua+whisper", "whisper+voxlingua")
            detection_model: Model identifier used
            whisper_confidence: Whisper confidence (if Whisper was used)
            voxlingua_confidence: VoxLingua confidence (if VoxLingua was used)
            whisper_time_ms: Whisper detection time in milliseconds
            voxlingua_time_ms: VoxLingua detection time in milliseconds
        """
        self.language = language
        self.confidence = confidence
        self.top_languages = top_languages
        self.detection_time_ms = detection_time_ms
        self.detection_method = detection_method
        self.detection_model = detection_model
        self.whisper_confidence = whisper_confidence
        self.voxlingua_confidence = voxlingua_confidence
        self.whisper_time_ms = whisper_time_ms
        self.voxlingua_time_ms = voxlingua_time_ms

    def get_top_k(self, k: int = 3) -> List[Dict[str, Any]]:
        """
        Get top-K languages sorted by confidence.

        Args:
            k: Number of top languages to return (default: 3)

        Returns:
            List of dictionaries with 'code' and 'prob' keys
        """
        sorted_langs = sorted(
            self.top_languages.items(), key=lambda kv: kv[1], reverse=True
        )[:k]

        return [{"code": lang, "prob": prob} for lang, prob in sorted_langs]


class LanguageDetectionService:
    """Service for intelligent language detection with configurable primary/secondary."""

    def __init__(
        self,
        whisper_model: WhisperModel,
        voxlingua_model: Optional[VoxLinguaModel] = None,
        primary_detector: str = "voxlingua",
        secondary_detector: str = "whisper",
        voxlingua_threshold: float = 0.90,
        whisper_threshold: float = 0.70,
        min_confidence_threshold: float = 0.70,  # Legacy, used if per-model not set
    ):
        """
        Initialize language detection service.

        Args:
            whisper_model: Whisper model for detection/fallback (GPU-based)
            voxlingua_model: VoxLingua107 model for detection/fallback (CPU-based)
            primary_detector: Primary detection model ("voxlingua" or "whisper")
            secondary_detector: Secondary detection model for fallback
            voxlingua_threshold: Minimum VoxLingua confidence to skip fallback
            whisper_threshold: Minimum Whisper confidence to skip fallback
            min_confidence_threshold: Legacy threshold (deprecated)
        """
        self.whisper_model = whisper_model
        self.voxlingua_model = voxlingua_model
        self.primary_detector = primary_detector.lower()
        self.secondary_detector = secondary_detector.lower()

        # Per-model thresholds
        self.thresholds = {
            "voxlingua": voxlingua_threshold,
            "whisper": whisper_threshold,
        }

        # Legacy support
        self.min_confidence_threshold = min_confidence_threshold

        # Log configuration
        primary_device = "CPU" if self.primary_detector == "voxlingua" else "GPU"
        secondary_device = "CPU" if self.secondary_detector == "voxlingua" else "GPU"
        print(
            f"🔧 Language detection: {self.primary_detector} ({primary_device}) primary, "
            f"{self.secondary_detector} ({secondary_device}) fallback"
        )
        print(
            f"   Thresholds: voxlingua={voxlingua_threshold:.2f}, whisper={whisper_threshold:.2f}"
        )

    def _is_detector_available(self, detector: str) -> bool:
        """Check if a detector is available."""
        if detector == "voxlingua":
            return (
                self.voxlingua_model is not None and self.voxlingua_model.is_available
            )
        elif detector == "whisper":
            return self.whisper_model is not None
        return False

    def _run_voxlingua(
        self, audio: Union[np.ndarray, torch.Tensor], sample_rate: int
    ) -> Tuple[str, float, Dict[str, float], float]:
        """
        Run VoxLingua detection.

        Returns:
            Tuple of (language, confidence, top_languages, time_ms)
        """
        start = time.time()
        assert self.voxlingua_model is not None, "VoxLingua model not available"
        language, confidence, top_languages = self.voxlingua_model.detect_language(
            audio, sample_rate
        )
        time_ms = (time.time() - start) * 1000
        return language, confidence, top_languages, time_ms

    def _run_whisper(
        self, audio: Union[np.ndarray, torch.Tensor], sample_rate: int
    ) -> Tuple[str, float, Dict[str, float], float]:
        """
        Run Whisper detection.

        Returns:
            Tuple of (language, confidence, top_languages, time_ms)
        """
        start = time.time()
        language, confidence, top_languages = self.whisper_model.detect_language(
            audio, sample_rate
        )
        time_ms = (time.time() - start) * 1000
        return language, confidence, top_languages, time_ms

    def detect(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        language_hint: Optional[str] = None,
    ) -> DetectionResult:
        """
        Detect language with configurable primary/secondary strategy.

        Process:
        1. If language_hint provided, use it directly (skip detection)
        2. Run primary detector
        3. If confidence < threshold, run secondary detector as fallback
        4. Return best result with full metadata

        Args:
            audio: Audio array (numpy or torch tensor)
            sample_rate: Sample rate in Hz
            language_hint: Optional language code to skip detection (e.g., "en")

        Returns:
            DetectionResult with language, confidence, and metadata
        """
        start_time = time.time()

        # Handle language hint (skip detection)
        if language_hint:
            return DetectionResult(
                language=language_hint,
                confidence=1.0,
                top_languages={language_hint: 1.0},
                detection_time_ms=0.0,
                detection_method="hint",
                detection_model="hint",
                whisper_confidence=None,
                voxlingua_confidence=None,
                whisper_time_ms=0.0,
                voxlingua_time_ms=0.0,
            )

        # Initialize tracking variables
        whisper_confidence = None
        voxlingua_confidence = None
        whisper_time_ms = 0.0
        voxlingua_time_ms = 0.0

        # Determine which detector to run first
        primary = self.primary_detector
        secondary = self.secondary_detector

        # Check primary availability, swap if needed
        if not self._is_detector_available(primary):
            if self._is_detector_available(secondary):
                primary, secondary = secondary, primary
            else:
                # No detectors available, return default
                return DetectionResult(
                    language="en",
                    confidence=0.0,
                    top_languages={"en": 0.0},
                    detection_time_ms=0.0,
                    detection_method="none",
                    detection_model="none",
                )

        # Run primary detector
        if primary == "voxlingua":
            language, confidence, top_languages, voxlingua_time_ms = (
                self._run_voxlingua(audio, sample_rate)
            )
            voxlingua_confidence = confidence
            detection_method = "voxlingua"
            detection_model = "voxlingua107"
        else:
            language, confidence, top_languages, whisper_time_ms = self._run_whisper(
                audio, sample_rate
            )
            whisper_confidence = confidence
            detection_method = "whisper"
            detection_model = self.whisper_model.language_detection_model_name

        # Check if fallback is needed
        threshold = self.thresholds.get(primary, 0.70)

        if confidence < threshold and self._is_detector_available(secondary):
            # Run secondary detector
            if secondary == "voxlingua":
                sec_lang, sec_conf, sec_top, voxlingua_time_ms = self._run_voxlingua(
                    audio, sample_rate
                )
                voxlingua_confidence = sec_conf
            else:
                sec_lang, sec_conf, sec_top, whisper_time_ms = self._run_whisper(
                    audio, sample_rate
                )
                whisper_confidence = sec_conf

            # Use secondary result if higher confidence
            if sec_conf > confidence:
                language = sec_lang
                confidence = sec_conf
                top_languages = sec_top
                detection_method = f"{primary}+{secondary}"
                if secondary == "voxlingua":
                    detection_model = f"{self.whisper_model.language_detection_model_name}+voxlingua107"
                else:
                    detection_model = f"voxlingua107+{self.whisper_model.language_detection_model_name}"
            else:
                # Primary was better, but we still ran both
                detection_method = f"{primary}+{secondary}"

        # Calculate total detection time
        detection_time_ms = (time.time() - start_time) * 1000

        return DetectionResult(
            language=language,
            confidence=confidence,
            top_languages=top_languages,
            detection_time_ms=detection_time_ms,
            detection_method=detection_method,
            detection_model=detection_model,
            whisper_confidence=whisper_confidence,
            voxlingua_confidence=voxlingua_confidence,
            whisper_time_ms=whisper_time_ms,
            voxlingua_time_ms=voxlingua_time_ms,
        )

    def supports_language(self, language: str, model_type: str) -> bool:
        """
        Check if a model supports a language.

        Args:
            language: ISO 639-1 language code
            model_type: Model type ("whisper" or "parakeet")

        Returns:
            True if language is supported, False otherwise
        """
        if model_type == "whisper":
            return language in self.whisper_model.supported_languages
        # Parakeet support would need to be checked with the Parakeet model
        # This is just a placeholder for the service layer
        return False
