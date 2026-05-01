"""
ONNX-based Parakeet ASR model with FP16 support.

Provides interface for Parakeet TDT 0.6B model using ONNX Runtime with:
- FP16 quantized models (50% smaller, faster inference)
- CUDA ExecutionProvider for GPU acceleration
- Compatible API with NeMo-based ParakeetModel
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Try to import onnx_asr - it's optional
try:
    import onnx_asr

    ONNX_ASR_AVAILABLE = True
except ImportError:
    ONNX_ASR_AVAILABLE = False
    logger.warning("onnx-asr not installed. ONNX backend will not be available.")


class ParakeetONNXModel:
    """
    ONNX-based Parakeet ASR model with FP16 support.

    This model uses the onnx-asr library to run inference with FP16 quantized
    ONNX models, providing 50% smaller footprint and faster inference on
    modern GPUs.

    Attributes:
        model_path: Path to the directory containing ONNX model files
        quantization: Quantization mode ("fp16" or None for FP32)
        device: Device to run inference on ("cuda" or "cpu")
    """

    def __init__(
        self,
        model_path: str = "./models/parakeet",
        quantization: Optional[str] = "fp16",
        device: Optional[str] = None,
        auto_download: bool = True,
        revision: Optional[str] = None,
    ):
        """
        Initialize ONNX-based Parakeet model.

        Args:
            model_path: Path to directory containing ONNX model files
            quantization: "fp16" for half-precision or None for FP32
            device: Device to use ("cuda" or "cpu"). Auto-detects if None.
            auto_download: If True, download models from HuggingFace if missing
            revision: HuggingFace revision (commit SHA/tag) to pin downloads
                to a specific snapshot. None or empty = latest on default branch.
        """
        if not ONNX_ASR_AVAILABLE:
            raise ImportError(
                "onnx-asr package is required for ONNX backend. "
                "Install with: pip install onnx-asr"
            )

        self.model_path = Path(model_path)
        self.quantization = quantization
        self.revision = revision or None

        # Determine device
        if device is None:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        # Check if models exist, download if needed
        if auto_download and not self._models_exist():
            self._download_models()

        # Validate model files exist
        self._validate_model_files()

        # Set up providers based on device
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # Load the model
        logger.info(f"Loading Parakeet ONNX model from {self.model_path}")
        logger.info(f"Quantization: {quantization}, Device: {self.device}")

        self.model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v3",
            str(self.model_path),
            quantization=quantization,
            providers=providers,
            cpu_preprocessing=(self.device == "cpu"),
        )

        precision_str = "FP16" if quantization == "fp16" else "FP32"
        print(f"✅ Parakeet ONNX loaded in {precision_str} mode")

    def _models_exist(self) -> bool:
        """Check if required model files exist."""
        if self.quantization == "fp16":
            required_files = [
                "encoder-model.fp16.onnx",
                "decoder_joint-model.fp16.onnx",
                "config.json",
                "vocab.txt",
                "nemo128.onnx",
            ]
        else:
            required_files = [
                "encoder-model.onnx",
                "decoder_joint-model.onnx",
                "config.json",
                "vocab.txt",
                "nemo128.onnx",
            ]

        for filename in required_files:
            if not (self.model_path / filename).exists():
                return False
        return True

    def _validate_model_files(self) -> None:
        """Validate that all required model files are present."""
        if self.quantization == "fp16":
            encoder_file = "encoder-model.fp16.onnx"
            decoder_file = "decoder_joint-model.fp16.onnx"
        else:
            encoder_file = "encoder-model.onnx"
            decoder_file = "decoder_joint-model.onnx"

        required_files = {
            encoder_file: "Encoder model",
            decoder_file: "Decoder model",
            "config.json": "Model config",
            "vocab.txt": "Vocabulary",
            "nemo128.onnx": "Feature extractor",
        }

        missing = []
        for filename, description in required_files.items():
            if not (self.model_path / filename).exists():
                missing.append(f"  - {filename} ({description})")

        if missing:
            raise FileNotFoundError(
                f"Missing required ONNX model files in {self.model_path}:\n"
                + "\n".join(missing)
                + "\n\n"
                "Run 'python scripts/download_fp16_models.py' to download models."
            )

    def _download_models(self) -> None:
        """Download FP16 models from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface-hub is required for model download. "
                "Install with: pip install huggingface-hub"
            )

        self.model_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading Parakeet ONNX models to {self.model_path}...")

        # FP16 models from grikdotnet repo
        fp16_repo = "grikdotnet/parakeet-tdt-0.6b-fp16"
        fp16_files = ["encoder-model.fp16.onnx", "decoder_joint-model.fp16.onnx"]

        # Supporting files from FP32 repo
        fp32_repo = "istupakov/parakeet-tdt-0.6b-v3-onnx"
        support_files = ["config.json", "vocab.txt", "nemo128.onnx"]

        # Revision is config-driven (TIER2_MODEL_REVISION); see SECURITY.md.
        try:
            for filename in fp16_files:
                logger.info(f"  Downloading {filename}...")
                hf_hub_download(  # nosec B615
                    repo_id=fp16_repo,
                    filename=filename,
                    local_dir=str(self.model_path),
                    revision=self.revision,
                )

            for filename in support_files:
                logger.info(f"  Downloading {filename}...")
                hf_hub_download(  # nosec B615
                    repo_id=fp32_repo,
                    filename=filename,
                    local_dir=str(self.model_path),
                    revision=self.revision,
                )

            logger.info("✅ Model download complete!")

        except Exception as e:
            raise RuntimeError(
                f"Failed to download models: {e}\n"
                "Please download manually from:\n"
                f"  - https://huggingface.co/{fp16_repo}\n"
                f"  - https://huggingface.co/{fp32_repo}"
            )

    def set_attention_mode(self, duration_sec: float) -> None:
        """
        Set attention mode based on audio duration.

        Note: ONNX models have fixed attention patterns. This method is
        provided for API compatibility with ParakeetModel but has no effect.

        Args:
            duration_sec: Audio duration in seconds (unused)
        """
        # ONNX models have fixed attention - this is a no-op for compatibility
        pass

    def transcribe(
        self,
        audio: Union[str, np.ndarray, List[np.ndarray]],
        timestamps: bool = True,
        batch_size: int = 1,
    ) -> Tuple[str, Optional[Dict]]:
        """
        Transcribe audio with optional timestamp extraction.

        Args:
            audio: Audio input - file path, numpy array, or list of arrays
                   Arrays should be 16kHz mono float32
            timestamps: Whether to extract timestamps (default: True)
            batch_size: Batch size for processing (currently ignored,
                       processes one at a time)

        Returns:
            Tuple of (transcript_text, timestamps_dict or None)
            - transcript_text: Transcribed text
            - timestamps_dict: Dict with 'char', 'word', 'segment' timestamp lists
                              Currently returns None (timestamp extraction TBD)
        """
        # Handle list input - process first item
        if isinstance(audio, list):
            if len(audio) == 0:
                return "", None
            audio = audio[0]

        # Handle file path - load and process as numpy array for proper chunking
        if isinstance(audio, str):
            print(f"🔊 ONNX loading file: {audio}")
            try:
                # Load audio file to numpy array
                audio_data, file_sr = sf.read(audio, dtype="float32")

                # Convert stereo to mono if needed
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1).astype(np.float32)

                # Resample to 16kHz if needed
                sample_rate = 16000
                if file_sr != sample_rate:
                    import torchaudio
                    import torch

                    audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
                    audio_data = (
                        torchaudio.functional.resample(
                            audio_tensor, file_sr, sample_rate
                        )
                        .squeeze()
                        .numpy()
                    )

                duration_sec = len(audio_data) / sample_rate
                print(
                    f"🔊 ONNX loaded file: {len(audio_data)} samples, {duration_sec:.1f}s"
                )

                # Apply same chunking logic as numpy arrays
                max_chunk_sec = 30
                overlap_sec = 1

                if duration_sec > max_chunk_sec:
                    text = self._transcribe_chunked(
                        audio_data, sample_rate, max_chunk_sec, overlap_sec
                    )
                else:
                    text = self._transcribe_single(audio_data, sample_rate)

                print(f"🔊 ONNX transcribed {len(text)} chars")
            except Exception as e:
                logger.exception("ONNX recognize failed: %s: %r", type(e).__name__, e)
                raise
        elif isinstance(audio, np.ndarray):
            # onnx_asr's recognize() expects file paths, not numpy arrays
            # Save to temp file and process
            sample_rate = 16000
            duration_sec = len(audio) / sample_rate
            print(
                f"🔊 ONNX processing numpy array: shape={audio.shape}, dtype={audio.dtype}, duration={duration_sec:.1f}s"
            )

            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # ONNX model has max input length limitation
            # Testing shows anything >30s can fail - use very conservative limit
            max_chunk_sec = 30  # Very conservative - 30s chunks work reliably
            overlap_sec = 1  # 1 second overlap between chunks

            # ALWAYS chunk - even short audio seems to fail randomly
            if duration_sec > max_chunk_sec:
                # Chunk long audio
                text = self._transcribe_chunked(
                    audio, sample_rate, max_chunk_sec, overlap_sec
                )
            else:
                # Process directly
                text = self._transcribe_single(audio, sample_rate)

            print(f"🔊 ONNX transcribed {len(text)} chars")
        else:
            raise TypeError(
                f"audio must be str, np.ndarray, or list, got {type(audio)}"
            )

        # TODO: Implement timestamp extraction for ONNX backend
        # The onnx-asr library may provide timestamps in a different format
        # that needs to be adapted to match NeMo's format
        timestamp_data: Optional[Dict[str, List]] = None
        if timestamps:
            # Placeholder for future timestamp implementation
            timestamp_data = {"char": [], "word": [], "segment": []}

        return text, timestamp_data

    def transcribe_batch(
        self, audio_list: List[Union[str, np.ndarray]], timestamps: bool = True
    ) -> List[Tuple[str, Optional[Dict]]]:
        """
        Transcribe a batch of audio files.

        Args:
            audio_list: List of audio inputs (file paths or arrays)
            timestamps: Whether to extract timestamps

        Returns:
            List of (transcript_text, timestamps_dict) tuples
        """
        results = []
        for audio in audio_list:
            result = self.transcribe(audio, timestamps=timestamps)
            results.append(result)
        return results

    def _transcribe_single(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe a single audio segment that fits within ONNX limits."""
        # Pad to 160-sample boundary (10ms frames)
        chunk_size = 160
        remainder = len(audio) % chunk_size
        if remainder != 0:
            pad_samples = chunk_size - remainder
            audio = np.pad(audio, (0, pad_samples), mode="constant", constant_values=0)

        duration_sec = len(audio) / sample_rate
        print(
            f"🔊 ONNX _transcribe_single: {len(audio)} samples, {duration_sec:.1f}s at {sample_rate}Hz"
        )

        # Create temp file
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        try:
            os.close(fd)
            sf.write(temp_path, audio, sample_rate)
            result = self.model.recognize(temp_path)

            # onnx_asr may return a generator - consume it
            if hasattr(result, "__iter__") and not isinstance(result, str):
                result = list(result)
                text = result[0] if result else ""
            else:
                text = result
            return text
        except Exception as e:
            logger.exception("ONNX recognize failed: %s: %r", type(e).__name__, e)
            raise
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _transcribe_chunked(
        self,
        audio: np.ndarray,
        sample_rate: int,
        max_chunk_sec: float,
        overlap_sec: float,
    ) -> str:
        """Transcribe long audio by chunking to avoid ONNX encoder limits."""
        chunk_samples = int(max_chunk_sec * sample_rate)
        overlap_samples = int(overlap_sec * sample_rate)
        stride = chunk_samples - overlap_samples

        total_samples = len(audio)
        num_chunks = (total_samples - overlap_samples) // stride + 1
        print(
            f"🔊 ONNX chunking {total_samples / sample_rate:.1f}s audio into {num_chunks} chunks of {max_chunk_sec}s"
        )

        transcripts = []
        start = 0
        chunk_idx = 0

        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            chunk = audio[start:end]
            chunk_idx += 1

            print(
                f"🔊 ONNX chunk {chunk_idx}: samples {start}-{end} ({len(chunk) / sample_rate:.1f}s)"
            )

            try:
                chunk_text = self._transcribe_single(chunk, sample_rate)
                transcripts.append(chunk_text)
            except Exception as e:
                print(f"❌ ONNX chunk {chunk_idx} failed: {e}")
                # Continue with remaining chunks
                transcripts.append("")

            start += stride
            if end >= total_samples:
                break

        # Join transcripts with space
        merged = " ".join(t.strip() for t in transcripts if t.strip())
        return merged

    @property
    def supported_languages(self) -> set:
        """
        Get set of supported language codes (ISO 639-1).

        Parakeet TDT 0.6B v3 supports 25 European languages.
        Non-European languages (e.g., Chinese, Japanese, Korean) should use Whisper.
        """
        return {
            # 25 European languages supported by Parakeet TDT 0.6B v3
            # Must match PARAKEET_SUPPORTED_LANGUAGES in Dockerfile
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

    @property
    def backend(self) -> str:
        """Return the backend type."""
        return "onnx"

    @property
    def precision(self) -> str:
        """Return the precision mode."""
        return self.quantization if self.quantization else "fp32"
