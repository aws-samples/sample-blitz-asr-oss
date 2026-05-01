"""
Parakeet ASR model wrapper with NeMo 2.4+ timestamp support.

Provides interface for NVIDIA Parakeet TDT 0.6B model with:
- Automatic timestamp extraction
- Attention mode switching for long audio
- Memory-efficient processing
"""

from typing import Dict, Optional, Tuple, Union
import torch
import numpy as np
import nemo.collections.asr as nemo_asr
from omegaconf import open_dict


class ParakeetModel:
    """Wrapper for NVIDIA Parakeet ASR model with timestamp support."""

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: Optional[str] = None,
        local_attention_threshold_sec: float = 1440.0,  # 24 minutes
    ):
        """
        Initialize Parakeet model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ("cuda" or "cpu"). Auto-detects if None.
            local_attention_threshold_sec: Switch to local attention after this duration
        """
        self.model_name = model_name
        self.local_attention_threshold_sec = local_attention_threshold_sec
        self._precision = "fp32"  # Track precision for autocast

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        if self.device == "cuda":
            self.model = self.model.to("cuda")

        # Enable timestamps in model configuration (NeMo 2.4+)
        self._enable_timestamps()

        print("✅ Parakeet loaded in FP32 with timestamps enabled (NeMo 2.4.0)")

    def set_precision(self, precision: str) -> None:
        """
        Set model precision and enable autocast for mixed precision inference.

        Args:
            precision: "fp32", "bfloat16", or "fp16"
        """
        self._precision = precision
        if precision != "fp32":
            print(f"✅ Parakeet precision set to {precision} (using autocast)")

    def _enable_timestamps(self) -> None:
        """Enable timestamp extraction in NeMo model configuration."""
        try:
            with open_dict(self.model.cfg.decoding):
                self.model.cfg.decoding.compute_timestamps = True
                self.model.cfg.decoding.preserve_alignments = True
                self.model.cfg.decoding.segment_seperators = [".", "?", "!", ","]
            self.model.change_decoding_strategy(self.model.cfg.decoding)
        except Exception as e:
            print(f"⚠️  Could not configure timestamps: {e}")

    def set_attention_mode(self, duration_sec: float) -> None:
        """
        Switch attention mode based on audio duration.

        Args:
            duration_sec: Audio duration in seconds
                - Short audio (< threshold): Use full attention (default)
                - Long audio (> threshold): Use local attention to manage memory
        """
        try:
            if duration_sec > self.local_attention_threshold_sec:
                self.model.change_attention_model(
                    self_attention_model="rel_pos_local_attn",
                    att_context_size=[256, 256],
                )
            else:
                # Switch back to full attention for short audio
                self.model.change_attention_model(self_attention_model="rel_pos")
        except Exception:
            # Not all model variants support attention switching
            pass

    def transcribe(
        self,
        audio: Union[str, np.ndarray, torch.Tensor, list],
        timestamps: bool = True,
        batch_size: int = 1,
    ) -> Tuple[str, Optional[Dict]]:
        """
        Transcribe audio with optional timestamp extraction.

        Args:
            audio: Audio input (file path, array, or list of paths/arrays)
            timestamps: Whether to extract timestamps (default: True)
            batch_size: Batch size for processing (default: 1)

        Returns:
            Tuple of (transcript_text, timestamps_dict or None)
            - transcript_text: Transcribed text
            - timestamps_dict: Dict with 'char', 'word', 'segment' timestamp lists
                              None if timestamps=False or unavailable
        """
        # Ensure audio is a list for NeMo API
        if not isinstance(audio, list):
            audio = [audio]

        # Run transcription with autocast for mixed precision
        if self._precision != "fp32" and self.device == "cuda":
            dtype = torch.bfloat16 if self._precision == "bfloat16" else torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype):
                results = self.model.transcribe(
                    audio, batch_size=batch_size, timestamps=timestamps
                )
        else:
            results = self.model.transcribe(
                audio, batch_size=batch_size, timestamps=timestamps
            )

        # Extract first result
        result = results[0] if results else None

        # Extract text from Hypothesis object
        text = getattr(result, "text", str(result))

        # Extract timestamps if requested and available
        timestamp_data = None
        if timestamps and result is not None and hasattr(result, "timestamp"):
            ts = result.timestamp or {}
            timestamp_data = {
                "char": ts.get("char", []),
                "word": ts.get("word", []),
                "segment": ts.get("segment", []),
            }

        return text, timestamp_data

    @property
    def supported_languages(self) -> set:
        """
        Get set of supported language codes (ISO 639-1).

        Language support varies by model:
        - parakeet-tdt_ctc-110m: English only (fast, 110M params)
        - parakeet-tdt-0.6b-v3: 25 European languages (600M params)

        Non-European languages (e.g., Chinese, Japanese, Korean) should use Whisper.
        """
        # CTC-110M is English-only (fastest model)
        if "ctc-110m" in self.model_name.lower():
            return {"en"}

        # 0.6B v3 supports 25 European languages
        return {
            "bg",  # Bulgarian
            "hr",  # Croatian
            "cs",  # Czech
            "da",  # Danish
            "nl",  # Dutch
            "en",  # English
            "et",  # Estonian
            "fi",  # Finnish
            "fr",  # French
            "de",  # German
            "el",  # Greek
            "hu",  # Hungarian
            "it",  # Italian
            "lv",  # Latvian
            "lt",  # Lithuanian
            "mt",  # Maltese
            "pl",  # Polish
            "pt",  # Portuguese
            "ro",  # Romanian
            "ru",  # Russian
            "sk",  # Slovak
            "sl",  # Slovenian
            "es",  # Spanish
            "sv",  # Swedish
            "uk",  # Ukrainian
        }
