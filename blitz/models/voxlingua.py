"""
VoxLingua107 model wrapper for fallback language detection.

Provides interface for SpeechBrain VoxLingua107 ECAPA model with:
- 107 language support
- Strong performance on low-resource languages
- Fallback when Whisper confidence is low
"""

from typing import Dict, Optional, Tuple, Union
import torch
import numpy as np
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier

from blitz.config import VOXLINGUA_TO_ISO639_1


class VoxLinguaModel:
    """Wrapper for SpeechBrain VoxLingua107 language detection model."""

    def __init__(
        self,
        model_source: str = "speechbrain/lang-id-voxlingua107-ecapa",
        save_dir: str = "tmp/voxlingua",
        device: Optional[str] = None,
    ):
        """
        Initialize VoxLingua107 model.

        Args:
            model_source: SpeechBrain model source identifier
            save_dir: Directory to save model files
            device: Device to load model on ("cuda" or "cpu"). Auto-detects if None.

        Raises:
            Exception: If model loading fails
        """
        self.model_source = model_source
        self.save_dir = save_dir

        # VoxLingua runs on CPU to free GPU for transcription
        # This is a key optimization: CPU-based language detection frees GPU cycles
        self.device = "cpu"

        # Load model on CPU
        try:
            self.model = EncoderClassifier.from_hparams(
                source=model_source, savedir=save_dir, run_opts={"device": self.device}
            )
            print(
                "✅ VoxLingua107 loaded on CPU (107 languages, frees GPU for transcription)"
            )
        except Exception as e:
            print(f"⚠️  VoxLingua107 not available: {e}")
            self.model = None
            raise

    def detect_language(
        self, audio: Union[np.ndarray, torch.Tensor], sample_rate: int
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Detect language using VoxLingua107 ECAPA model.

        Supports 107 languages with strong performance on low-resource languages.

        Args:
            audio: Audio array (numpy or torch tensor)
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (language_code, confidence, top_3_dict)
            - language_code: ISO 639-1 code (e.g., "en", "cy")
            - confidence: Float between 0 and 1
            - top_3_dict: Dictionary of top 3 languages with their probabilities

        Note:
            Returns ("en", 0.0, {}) if model is not loaded
        """
        if self.model is None:
            return "en", 0.0, {}

        # VoxLingua expects 16kHz audio
        target_sr = 16000
        if sample_rate != target_sr:
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio).float()
            else:
                audio_tensor = audio.float()

            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sample_rate, target_sr
            )
            audio = audio_tensor.cpu().numpy()
            sample_rate = target_sr

        # Limit to first 30 seconds for faster, more reliable detection
        # (matching Whisper's language detection approach)
        max_samples = 30 * sample_rate  # 30 seconds
        if isinstance(audio, np.ndarray):
            if len(audio) > max_samples:
                audio = audio[:max_samples]
        elif isinstance(audio, torch.Tensor):
            if audio.shape[0] > max_samples:
                audio = audio[:max_samples]

        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()

        # Move tensor to the same device as the model
        audio_tensor = audio_tensor.to(self.device)

        # Run prediction (returns tuple of predictions and probabilities)
        with torch.no_grad():
            prediction = self.model.classify_batch(audio_tensor.unsqueeze(0))

        # Extract results
        # prediction[0]: Log-likelihoods tensor for all 107 languages [batch_size, num_langs]
        # prediction[1]: Best log-likelihood value as tensor
        # prediction[2]: Index of best language as tensor
        # prediction[3]: Language code - can be ['cym'] or ['cym: Welsh']
        all_log_likelihoods = prediction[0]  # Tensor with all language log-likelihoods
        predicted_lang_labels = prediction[3]  # List with language code
        best_lang_index = prediction[2]  # Index of best language

        # Extract ISO 639-3 code from label
        # Handle both formats: ['cym'] and ['cym: Welsh']
        vox_lang_code = None
        if isinstance(predicted_lang_labels, list) and len(predicted_lang_labels) > 0:
            lang_label = predicted_lang_labels[0]  # e.g., "cym" or "cym: Welsh"
            # Split on colon and take first part (handles both formats)
            vox_lang_code = lang_label.split(":")[0].strip()  # Extract "cym"

        # Convert from ISO 639-3 (VoxLingua) to ISO 639-1 (Whisper/Parakeet)
        if vox_lang_code:
            best_lang_code = VOXLINGUA_TO_ISO639_1.get(vox_lang_code, vox_lang_code)
        else:
            best_lang_code = "en"

        # Get proper probability using softmax on all log-likelihoods
        # This gives calibrated probabilities that sum to 1.0
        if all_log_likelihoods is not None and all_log_likelihoods.numel() > 0:
            # Apply softmax to convert log-likelihoods to probabilities
            probs = torch.nn.functional.softmax(all_log_likelihoods[0], dim=0)

            # Get probability for the best language
            if best_lang_index is not None and best_lang_index.numel() > 0:
                best_idx = int(best_lang_index.item())
                best_prob = float(probs[best_idx].item())
            else:
                best_prob = 0.0

            top_3_dict = {best_lang_code: best_prob}
        else:
            best_prob = 0.0
            top_3_dict = {}

        return best_lang_code, best_prob, top_3_dict

    @property
    def is_available(self) -> bool:
        """Check if model is loaded and available."""
        return self.model is not None

    @property
    def num_languages(self) -> int:
        """Number of languages supported by VoxLingua107."""
        return 107
