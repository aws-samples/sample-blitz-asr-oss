"""
Transcription service orchestrating the complete ASR workflow.

Coordinates:
- Language detection (Whisper + VoxLingua)
- Model routing (Parakeet vs Whisper)
- Audio preprocessing and optimization
- Result formatting and metrics collection
"""

from typing import Dict, Optional, Tuple, Union
import time
import torch
import numpy as np

from blitz.models.parakeet_factory import ParakeetModelProtocol
from blitz.models.whisper import WhisperModel
from blitz.services.language_detection import LanguageDetectionService, DetectionResult
from blitz.utils.audio import (
    get_audio_duration,
    chunk_audio,
    merge_chunked_transcripts,
    save_audio_to_temp,
    cleanup_temp_file,
)


class TranscriptionResult:
    """Transcription result with metadata."""

    def __init__(
        self,
        text: str,
        language: str,
        language_confidence: float,
        engine: str,
        duration_sec: float,
        timestamps: Optional[Dict] = None,
        chunk_approach: str = "direct",
        chunk_count: int = 1,
        chunk_size_sec: Optional[int] = None,
        transcription_time_sec: float = 0.0,
    ):
        """
        Initialize transcription result.

        Args:
            text: Transcribed text
            language: Detected language (ISO 639-1 code)
            language_confidence: Language detection confidence
            engine: Engine used ("parakeet" or "whisper")
            duration_sec: Audio duration in seconds
            timestamps: Optional timestamp data with char/word/segment timing
            chunk_approach: Processing approach ("direct", "chunked", "temp_file")
            chunk_count: Number of chunks processed
            chunk_size_sec: Chunk size in seconds (if chunked)
            transcription_time_sec: Transcription time in seconds
        """
        self.text = text
        self.language = language
        self.language_confidence = language_confidence
        self.engine = engine
        self.duration_sec = duration_sec
        self.timestamps = timestamps
        self.chunk_approach = chunk_approach
        self.chunk_count = chunk_count
        self.chunk_size_sec = chunk_size_sec
        self.transcription_time_sec = transcription_time_sec


class TranscriptionService:
    """Service for intelligent transcription with automatic model routing."""

    def __init__(
        self,
        parakeet_model: ParakeetModelProtocol,
        whisper_model: WhisperModel,
        language_detection_service: LanguageDetectionService,
        short_audio_threshold_sec: float = 600.0,  # 10 minutes
        chunk_length_sec: int = 1200,  # 20 minutes
        chunk_overlap_sec: int = 30,  # 30 seconds
    ):
        """
        Initialize transcription service.

        Args:
            parakeet_model: Parakeet model for efficient transcription
            whisper_model: Whisper model for multilingual fallback
            language_detection_service: Language detection service
            short_audio_threshold_sec: Threshold for direct vs temp file processing
            chunk_length_sec: Maximum chunk length for long audio
            chunk_overlap_sec: Overlap between chunks
        """
        self.parakeet_model = parakeet_model
        self.whisper_model = whisper_model
        self.language_detection_service = language_detection_service
        self.short_audio_threshold_sec = short_audio_threshold_sec
        self.chunk_length_sec = chunk_length_sec
        self.chunk_overlap_sec = chunk_overlap_sec

    def transcribe(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        language_hint: Optional[str] = None,
        timestamps: bool = True,
    ) -> Tuple[TranscriptionResult, DetectionResult]:
        """
        Transcribe audio with intelligent model routing.

        Process:
        1. Detect language (or use hint)
        2. Route to optimal model:
           - Parakeet: High performance for 25 supported languages
           - Whisper: Multilingual fallback for 99+ languages
        3. Apply optimizations:
           - Direct array processing for short audio
           - Temp file processing for long audio
           - Chunking for very long audio (>20 min)
           - Attention mode switching for Parakeet

        Args:
            audio: Audio array (numpy or torch tensor)
            sample_rate: Sample rate in Hz
            language_hint: Optional language code to skip detection
            timestamps: Whether to extract timestamps (default: True)

        Returns:
            Tuple of (TranscriptionResult, DetectionResult)
        """
        # Calculate duration for routing decisions
        duration_sec = get_audio_duration(audio, sample_rate)

        # Detect language
        detection_result = self.language_detection_service.detect(
            audio, sample_rate, language_hint
        )

        # Choose backend: Parakeet vs Whisper
        language = detection_result.language
        use_parakeet = language in self.parakeet_model.supported_languages
        use_whisper = not use_parakeet

        # Determine engine
        if use_parakeet:
            engine = "parakeet"
        elif use_whisper:
            engine = "whisper"
        else:
            # Fallback to Whisper with English
            engine = "whisper"
            language = "en"

        # Transcribe with selected engine
        start_time = time.time()

        if engine == "parakeet":
            text, timestamp_data, chunk_approach, chunk_count, chunk_size = (
                self._transcribe_with_parakeet(
                    audio, sample_rate, duration_sec, timestamps
                )
            )
        else:
            text, timestamp_data, chunk_approach, chunk_count, chunk_size = (
                self._transcribe_with_whisper(audio, sample_rate, language, timestamps)
            )

        transcription_time_sec = time.time() - start_time

        # Build result
        transcription_result = TranscriptionResult(
            text=text,
            language=language,
            language_confidence=detection_result.confidence,
            engine=engine,
            duration_sec=duration_sec,
            timestamps=timestamp_data,
            chunk_approach=chunk_approach,
            chunk_count=chunk_count,
            chunk_size_sec=chunk_size,
            transcription_time_sec=transcription_time_sec,
        )

        return transcription_result, detection_result

    def _transcribe_with_parakeet(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        duration_sec: float,
        timestamps: bool,
    ) -> Tuple[str, Optional[Dict], str, int, Optional[int]]:
        """
        Transcribe with Parakeet model.

        Returns:
            Tuple of (text, timestamps, chunk_approach, chunk_count, chunk_size_sec)
        """
        # Set attention mode based on duration
        self.parakeet_model.set_attention_mode(duration_sec)

        # Check if chunking is needed
        if duration_sec > self.chunk_length_sec:
            # Chunk long audio to prevent OOM
            # Note: Timestamps disabled for chunked audio (merging timestamps is complex)
            return self._transcribe_parakeet_chunked(audio, sample_rate, duration_sec)

        # Determine processing approach
        if duration_sec > self.short_audio_threshold_sec:
            chunk_approach = "temp_file"
        else:
            chunk_approach = "direct"

        # Transcribe
        text, timestamp_data = self.parakeet_model.transcribe(
            audio, timestamps=timestamps
        )

        return text, timestamp_data, chunk_approach, 1, None

    def _transcribe_parakeet_chunked(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        duration_sec: float,
    ) -> Tuple[str, None, str, int, int]:
        """
        Transcribe long audio with chunking (Parakeet).

        Returns:
            Tuple of (text, None, "chunked", chunk_count, chunk_size_sec)
        """
        # Convert to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Chunk audio
        chunks = chunk_audio(
            audio, sample_rate, self.chunk_length_sec, self.chunk_overlap_sec
        )

        # Process each chunk
        chunk_results = []
        for audio_chunk, start_sample, end_sample in chunks:
            # Save chunk to temp file
            temp_file = save_audio_to_temp(audio_chunk, sample_rate)
            try:
                text, _ = self.parakeet_model.transcribe(
                    temp_file,
                    timestamps=False,  # Disabled for chunked audio
                )
                chunk_results.append((text, start_sample, end_sample))
            finally:
                cleanup_temp_file(temp_file)

        # Merge chunk results
        merged_text = merge_chunked_transcripts(
            chunk_results, self.chunk_overlap_sec * sample_rate, sample_rate
        )

        return merged_text, None, "chunked", len(chunks), self.chunk_length_sec

    def _transcribe_with_whisper(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        language: str,
        timestamps: bool,
    ) -> Tuple[str, Optional[Dict], str, int, Optional[int]]:
        """
        Transcribe with Whisper model.

        Returns:
            Tuple of (text, timestamps, chunk_approach, chunk_count, chunk_size_sec)
        """
        # Convert to torch tensor if needed
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()

        # Check if chunking is needed
        if get_audio_duration(audio, sample_rate) > self.chunk_length_sec:
            # Chunk long audio
            return self._transcribe_whisper_chunked(
                audio_tensor, sample_rate, language, timestamps
            )

        # Direct transcription
        text, timestamp_data = self.whisper_model.transcribe(
            audio_tensor, sample_rate, language, timestamps
        )

        return text, timestamp_data, "direct", 1, None

    def _transcribe_whisper_chunked(
        self,
        audio_tensor: torch.Tensor,
        sample_rate: int,
        language: str,
        timestamps: bool,
    ) -> Tuple[str, Optional[Dict], str, int, int]:
        """
        Transcribe long audio with chunking (Whisper).

        Returns:
            Tuple of (text, timestamps_dict, "chunked", chunk_count, chunk_size_sec)
        """
        # Chunk audio
        chunks = chunk_audio(
            audio_tensor, sample_rate, self.chunk_length_sec, self.chunk_overlap_sec
        )

        # Process each chunk
        chunk_results = []
        chunk_timestamps = []

        for audio_chunk, start_sample, end_sample in chunks:
            # Transcribe chunk
            text, _ = self.whisper_model.transcribe(
                audio_chunk,
                sample_rate,
                language,
                timestamps=False,  # Individual chunk timestamps not merged
            )
            chunk_results.append((text, start_sample, end_sample))

            # Store chunk timing if timestamps enabled
            if timestamps and text.strip():
                chunk_timestamps.append(
                    {
                        "start": start_sample / sample_rate,
                        "end": end_sample / sample_rate,
                        "segment": text,
                    }
                )

        # Merge chunk results
        merged_text = merge_chunked_transcripts(
            chunk_results, self.chunk_overlap_sec * sample_rate, sample_rate
        )

        # Create timestamp data from chunks if requested
        timestamp_data = None
        if timestamps and chunk_timestamps:
            timestamp_data = {"segment": chunk_timestamps, "word": [], "char": []}

        return (
            merged_text,
            timestamp_data,
            "chunked",
            len(chunks),
            self.chunk_length_sec,
        )
