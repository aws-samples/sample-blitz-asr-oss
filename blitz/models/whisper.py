"""
Whisper model wrapper for language detection and transcription.

Provides interface for OpenAI Whisper models with:
- Language detection using efficient single decoding step
- Transcription with forced language
- Optional separate tiny model for language detection
"""

from typing import Collection, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from blitz.utils.audio import resample_audio


class WhisperModel:
    """Wrapper for OpenAI Whisper models with language detection."""

    def __init__(
        self,
        primary_model_name: str = "openai/whisper-large-v3-turbo",
        language_detection_model: str = "large-v3-turbo",
        device: Optional[str] = None,
    ):
        """
        Initialize Whisper model(s).

        Args:
            primary_model_name: Main model for transcription
            language_detection_model: Which model to use for language detection
                - "tiny": Load separate tiny model (faster, less accurate)
                - "large-v3-turbo": Use primary model (slower, more accurate)
            device: Device to load model on ("cuda" or "cpu"). Auto-detects if None.
        """
        self.primary_model_name = primary_model_name
        self.language_detection_model_name = language_detection_model

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load primary model (always needed for transcription)
        self.processor = WhisperProcessor.from_pretrained(primary_model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            primary_model_name
        ).to(self.device)
        print(f"✅ Whisper {primary_model_name} loaded as primary model")

        # Conditionally load separate tiny model for language detection
        if language_detection_model == "tiny":
            self.lang_detect_processor = WhisperProcessor.from_pretrained(
                "openai/whisper-tiny"
            )
            self.lang_detect_model = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-tiny"
            ).to(self.device)
            print("✅ Whisper tiny loaded for language detection")
        else:
            # Use primary model for language detection
            self.lang_detect_processor = self.processor
            self.lang_detect_model = self.model
            print("✅ Using Whisper large-v3-turbo for language detection")

    def detect_language(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        possible_languages: Optional[Collection[str]] = None,
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Detect language using fast single decoding step.

        Args:
            audio: Audio array (numpy or torch tensor)
            sample_rate: Sample rate in Hz
            possible_languages: Optional set of language codes to restrict detection

        Returns:
            Tuple of (language_code, confidence, top_3_dict)
            - language_code: ISO 639-1 code (e.g., "en", "cy")
            - confidence: Float between 0 and 1
            - top_3_dict: Dictionary of top 3 languages with their probabilities
        """
        # Convert to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Resample to 16kHz if needed (Whisper requirement)
        audio, sample_rate = resample_audio(audio, sample_rate, target_sr=16000)

        # Process audio to features
        inputs = self.lang_detect_processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        )

        # Move to device
        input_features = inputs.input_features.to(self.device)

        # Get language probabilities
        language_probs = self._detect_language_batch(
            self.lang_detect_model,
            self.lang_detect_processor,
            input_features,
            possible_languages,
        )

        # Extract top language and probabilities
        probs = language_probs[0]  # First (and only) item in batch

        # Convert token format "<|en|>" to "en"
        probs_cleaned = {lang[2:-2]: prob for lang, prob in probs.items()}

        # Get top 3 languages
        sorted_langs = sorted(probs_cleaned.items(), key=lambda x: x[1], reverse=True)
        top_3 = dict(sorted_langs[:3])

        # Best language
        best_lang = sorted_langs[0][0]
        best_prob = sorted_langs[0][1]

        return best_lang, best_prob, top_3

    def _detect_language_batch(
        self,
        model: WhisperForConditionalGeneration,
        processor: WhisperProcessor,
        input_features: torch.Tensor,
        possible_languages: Optional[Collection[str]] = None,
    ) -> List[Dict[str, float]]:
        """
        Language detection using a single decoding step of HF Whisper.
        Fast and memory-efficient - only uses encoder + one decoder step.

        Args:
            model: Whisper model to use
            processor: Whisper processor
            input_features: Preprocessed audio features
            possible_languages: Optional language code restriction

        Returns:
            List of dictionaries mapping language tokens to probabilities
        """
        tokenizer = processor.tokenizer

        # All language tokens (and only them) are 6 characters long, e.g. "<|en|>"
        language_tokens = [
            t for t in tokenizer.additional_special_tokens if len(t) == 6
        ]

        if possible_languages is not None:
            language_tokens = [
                t for t in language_tokens if t[2:-2] in possible_languages
            ]
            if len(language_tokens) < len(possible_languages):
                raise RuntimeError(
                    f"Some languages in {possible_languages} did not have associated language tokens"
                )

        language_token_ids = tokenizer.convert_tokens_to_ids(language_tokens)

        # 50258 is the token for "transcribe" in Whisper
        decoder_input_ids = torch.tensor(
            [[50258] for _ in range(input_features.shape[0])],
            device=input_features.device,
            dtype=torch.long,
        )

        with torch.no_grad():
            logits = model(
                input_features,
                decoder_input_ids=decoder_input_ids,
            ).logits

        # Mask out all non-language tokens
        mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
        mask[language_token_ids] = False
        logits[:, :, mask] = -float("inf")

        output_probs = logits.softmax(dim=-1).cpu()

        return [
            {
                lang: output_probs[input_idx, 0, token_id].item()
                for token_id, lang in zip(language_token_ids, language_tokens)
            }
            for input_idx in range(logits.shape[0])
        ]

    def transcribe(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        language: str,
        timestamps: bool = False,
    ) -> Tuple[str, Optional[Dict]]:
        """
        Transcribe audio with Whisper.

        Args:
            audio: Audio array (numpy or torch tensor)
            sample_rate: Sample rate in Hz
            language: ISO 639-1 language code to force
            timestamps: Whether to generate timestamps (default: False)

        Returns:
            Tuple of (transcript_text, timestamps_dict or None)
            - transcript_text: Transcribed text
            - timestamps_dict: Dict with 'segment' timestamp list (if timestamps=True)
                              None if timestamps=False
        """
        # Convert to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Resample to 16kHz if needed (Whisper requirement)
        audio, sample_rate = resample_audio(audio, sample_rate, target_sr=16000)

        # Process audio
        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt")

        # Generate forced decoder IDs for language
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        )

        # Run generation
        with torch.no_grad():
            result = self.model.generate(
                input_features=inputs.input_features.to(
                    device=self.model.device, dtype=self.model.dtype
                ),
                forced_decoder_ids=forced_decoder_ids,
            )

        # Decode result
        text = self.processor.batch_decode(result, skip_special_tokens=True)[0]

        # Timestamps not implemented yet - would require chunking and segment tracking
        timestamp_data = None
        if timestamps:
            # TODO: Implement chunked processing with timestamp tracking
            pass

        return text, timestamp_data

    @property
    def supported_languages(self) -> set:
        """
        Get set of supported language codes (ISO 639-1).

        Whisper supports 99 languages including most world languages.
        """
        return {
            # Major languages
            "en",
            "zh",
            "de",
            "es",
            "ru",
            "ko",
            "fr",
            "ja",
            "pt",
            "tr",
            "pl",
            "ca",
            "nl",
            "ar",
            "sv",
            "it",
            "id",
            "hi",
            "fi",
            "vi",
            "he",
            "uk",
            "el",
            "ms",
            "cs",
            "ro",
            "da",
            "hu",
            "ta",
            "no",
            "th",
            "ur",
            "hr",
            "bg",
            "lt",
            "la",
            "mi",
            "ml",
            "cy",
            "sk",
            "te",
            "fa",
            "lv",
            "bn",
            "sr",
            "az",
            "sl",
            "kn",
            "et",
            "mk",
            "br",
            "eu",
            "is",
            "hy",
            "ne",
            "mn",
            "bs",
            "kk",
            "sq",
            "sw",
            "gl",
            "mr",
            "pa",
            "si",
            "km",
            "sn",
            "yo",
            "so",
            "af",
            "oc",
            "ka",
            "be",
            "tg",
            "sd",
            "gu",
            "am",
            "yi",
            "lo",
            "uz",
            "fo",
            "ht",
            "ps",
            "tk",
            "nn",
            "mt",
            "sa",
            "lb",
            "my",
            "bo",
            "tl",
            "mg",
            "as",
            "tt",
            "haw",
            "ln",
            "ha",
            "ba",
            "jw",
            "su",
        }
