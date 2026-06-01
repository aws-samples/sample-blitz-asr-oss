"""
Model wrappers for Parakeet, Whisper, and VoxLingua107.

Provides clean interfaces for model loading, language detection, and transcription.
Supports both NeMo (PyTorch) and ONNX backends for Parakeet.
"""

from typing import Any

from .parakeet import ParakeetModel
from .whisper import WhisperModel
from .voxlingua import VoxLinguaModel
from .parakeet_factory import (
    create_parakeet_model,
    create_tier2_model_from_config,
    create_parakeet_ctc_model,
    create_tier1_model_from_config,
    ParakeetModelProtocol,
)

# Conditional import for ONNX model
# Declare with Any type for conditional assignment
ParakeetONNXModel: Any
ONNX_ASR_AVAILABLE: bool

try:
    from .parakeet_onnx import ParakeetONNXModel, ONNX_ASR_AVAILABLE
except ImportError:
    ParakeetONNXModel = None
    ONNX_ASR_AVAILABLE = False

__all__ = [
    # NeMo models
    "ParakeetModel",
    "WhisperModel",
    "VoxLinguaModel",
    # Factory functions
    "create_parakeet_model",
    "create_tier2_model_from_config",
    "create_parakeet_ctc_model",
    "create_tier1_model_from_config",
    "ParakeetModelProtocol",
    # ONNX model (optional)
    "ParakeetONNXModel",
    "ONNX_ASR_AVAILABLE",
]
