"""
Factory for creating Parakeet ASR models.

Provides a unified interface to create either NeMo-based or ONNX-based
Parakeet models based on configuration.
"""

import logging
from typing import Optional, Protocol, Dict, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ParakeetModelProtocol(Protocol):
    """Protocol defining the interface for Parakeet models."""

    def transcribe(
        self,
        audio: Union[str, np.ndarray, list],
        timestamps: bool = True,
        batch_size: int = 1,
    ) -> Tuple[str, Optional[Dict]]:
        """Transcribe audio with optional timestamps."""
        ...

    def set_attention_mode(self, duration_sec: float) -> None:
        """Set attention mode based on audio duration."""
        ...

    @property
    def supported_languages(self) -> set:
        """Get set of supported language codes."""
        ...


def create_parakeet_model(
    backend: str = "nemo",
    precision: str = "bfloat16",
    model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
    onnx_model_path: str = "./models/parakeet",
    auto_download: bool = True,
    device: Optional[str] = None,
    local_attention_threshold_sec: float = 1440.0,
) -> ParakeetModelProtocol:
    """
    Factory function to create a Parakeet ASR model.

    Creates either a NeMo-based model (default) or an ONNX-based model
    depending on the backend parameter.

    Args:
        backend: Model backend - "nemo" (PyTorch/NeMo) or "onnx" (ONNX Runtime)
        precision: Precision mode for NeMo backend - "fp32", "bfloat16", "fp16"
        model_name: HuggingFace model name for NeMo backend
        onnx_model_path: Path to ONNX model files for ONNX backend
        auto_download: Auto-download ONNX models from HuggingFace if missing
        device: Device to use - "cuda" or "cpu" (auto-detects if None)
        local_attention_threshold_sec: Duration threshold for local attention (NeMo only)

    Returns:
        A Parakeet model instance (either ParakeetModel or ParakeetONNXModel)

    Raises:
        ValueError: If backend is not "nemo" or "onnx"
        ImportError: If required packages are not installed

    Examples:
        # Create NeMo model with BFloat16 (default)
        model = create_parakeet_model()

        # Create NeMo model with FP32
        model = create_parakeet_model(precision="fp32")

        # Create ONNX model with FP16
        model = create_parakeet_model(backend="onnx")

        # Create ONNX model with FP32
        model = create_parakeet_model(backend="onnx", precision="fp32")
    """
    backend = backend.lower()

    if backend == "nemo":
        return _create_nemo_model(
            model_name=model_name,
            precision=precision,
            device=device,
            local_attention_threshold_sec=local_attention_threshold_sec,
        )
    elif backend == "onnx":
        return _create_onnx_model(
            model_path=onnx_model_path,
            precision=precision,
            device=device,
            auto_download=auto_download,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Must be 'nemo' or 'onnx'")


def _create_nemo_model(
    model_name: str,
    precision: str,
    device: Optional[str],
    local_attention_threshold_sec: float,
) -> ParakeetModelProtocol:
    """Create NeMo-based Parakeet model."""
    from blitz.models.parakeet import ParakeetModel

    logger.info(f"Creating NeMo Parakeet model: {model_name}")
    logger.info(f"Precision: {precision}")

    model = ParakeetModel(
        model_name=model_name,
        device=device,
        local_attention_threshold_sec=local_attention_threshold_sec,
    )

    # Apply precision conversion
    if precision != "fp32":
        import torch

        if precision == "bfloat16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
        else:
            raise ValueError(f"Unknown precision: {precision}")

        if hasattr(model, "model"):
            model.model = model.model.to(dtype=dtype)
            logger.info(f"Converted model to {precision}")

        # Set precision for autocast in transcribe method
        if hasattr(model, "set_precision"):
            model.set_precision(precision)

    return model


def _create_onnx_model(
    model_path: str, precision: str, device: Optional[str], auto_download: bool
) -> ParakeetModelProtocol:
    """Create ONNX-based Parakeet model."""
    from blitz.models.parakeet_onnx import ParakeetONNXModel, ONNX_ASR_AVAILABLE

    if not ONNX_ASR_AVAILABLE:
        raise ImportError(
            "ONNX backend requires onnx-asr package. "
            "Install with: pip install onnx-asr onnxruntime-gpu"
        )

    # Map precision to quantization parameter
    if precision == "fp16":
        quantization = "fp16"
    else:
        # For fp32 or bfloat16, use no quantization (FP32 ONNX)
        quantization = None

    logger.info(f"Creating ONNX Parakeet model from: {model_path}")
    logger.info(f"Quantization: {quantization or 'fp32'}")

    return ParakeetONNXModel(
        model_path=model_path,
        quantization=quantization,
        device=device,
        auto_download=auto_download,
    )


def create_tier2_model_from_config(config) -> ParakeetModelProtocol:
    """
    Create Tier 2 model (Parakeet 0.6B) from a ServerConfig instance.

    Tier 2 is the fast multilingual model for European languages.

    Args:
        config: ServerConfig instance with tier2_* configuration

    Returns:
        A Parakeet model instance for Tier 2
    """
    return create_parakeet_model(
        backend=config.tier2_backend,
        precision=config.tier2_precision,
        model_name=config.tier2_model,
        onnx_model_path=config.tier2_onnx_model_path,
        auto_download=config.tier2_auto_download,
        device=config.device,
        local_attention_threshold_sec=config.local_attention_threshold_sec,
    )


def create_parakeet_ctc_model(
    precision: str = "fp16",
    model_name: str = "nvidia/parakeet-tdt_ctc-110m",
    device: Optional[str] = None,
    local_attention_threshold_sec: float = 1440.0,
) -> ParakeetModelProtocol:
    """
    Create fast English-only Parakeet CTC model.

    The CTC-110M model is optimized for English transcription with:
    - RTFx: 5345 (vs 3332 for 0.6B) - 60% faster
    - WER: 7.49% (vs 6.32% for 0.6B) - slight accuracy tradeoff
    - Memory: ~0.5GB (vs ~2.5GB for 0.6B)

    Args:
        precision: Precision mode - "fp32" or "fp16"
        model_name: HuggingFace model identifier
        device: Device to use - "cuda" or "cpu" (auto-detects if None)
        local_attention_threshold_sec: Duration threshold for local attention

    Returns:
        A Parakeet CTC model instance (English-only)

    Examples:
        # Create CTC model with FP16 (default, fastest)
        model = create_parakeet_ctc_model()

        # Create CTC model with FP32 (higher precision)
        model = create_parakeet_ctc_model(precision="fp32")
    """
    return _create_nemo_model(
        model_name=model_name,
        precision=precision,
        device=device,
        local_attention_threshold_sec=local_attention_threshold_sec,
    )


def create_tier1_model_from_config(config) -> ParakeetModelProtocol:
    """
    Create Tier 1 model (Parakeet CTC-110M) from a ServerConfig instance.

    Tier 1 is the fastest model for English transcription.

    Args:
        config: ServerConfig instance with tier1_* configuration

    Returns:
        A Parakeet CTC model instance (English-only)
    """
    return create_parakeet_ctc_model(
        precision=config.tier1_precision,
        model_name=config.tier1_model,
        device=config.device,
        local_attention_threshold_sec=config.local_attention_threshold_sec,
    )
