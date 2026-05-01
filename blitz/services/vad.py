"""
Voice Activity Detection (VAD) Service

Provides fast silence detection using Silero VAD before language detection.
Optimized for minimal latency and memory usage.
"""

from typing import Dict
import time
import numpy as np
import torch
from pathlib import Path


class VADService:
    """
    Silence detection service using Silero VAD.

    Analyzes audio to determine if it contains meaningful speech
    before expensive language detection and transcription.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize VAD service.

        Args:
            device: Device to run VAD on ("cpu" or "cuda")
                   Default "cpu" - VAD is fast enough on CPU
        """
        self.device = device
        self.sample_rate = 16000
        self._model = None
        self._utils = None

    def _load_model(self):
        """Lazy load VAD model on first use."""
        if self._model is None:
            print("🔧 Loading Silero VAD model...")
            from torch.hub import load as torch_hub_load

            self._model, self._utils = torch_hub_load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
            )
            self._model = self._model.to(self.device)
            print(f"✅ Silero VAD loaded on {self.device}")

    def detect_silence(
        self,
        audio: np.ndarray,
        sample_rate: int,
        threshold: float = 0.5,
        max_analysis_sec: float = 60.0,
    ) -> Dict:
        """
        Detect if audio is mostly silent.

        Args:
            audio: Audio array (mono, float32 or int16)
            sample_rate: Audio sample rate
            threshold: VAD probability threshold (0.0-1.0)
                      Higher = stricter (less sensitive to noise)
            max_analysis_sec: Only analyze first N seconds for long audio

        Returns:
            Dictionary with:
                - is_silent: bool - True if <10% of audio is speech
                - speech_ratio: float - Fraction of audio containing speech (0.0-1.0)
                - speech_duration_sec: float - Seconds of speech detected
                - total_duration_sec: float - Total audio duration analyzed
                - vad_time_ms: float - VAD processing time in milliseconds
                - method: str - Detection method used
        """
        start_time = time.time()

        # Lazy load model
        self._load_model()

        # Ensure float32 [-1, 1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Resample if needed
        if sample_rate != self.sample_rate:
            audio = self._resample(audio, sample_rate, self.sample_rate)

        # Limit analysis duration for very long files
        max_samples = int(max_analysis_sec * self.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        total_duration_sec = len(audio) / self.sample_rate

        # Get speech timestamps from Silero VAD
        assert self._utils is not None, "_load_model() must be called first"
        get_speech_timestamps = self._utils[0]

        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio).to(self.device)

            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                self._model,
                sampling_rate=self.sample_rate,
                threshold=threshold,
                min_speech_duration_ms=250,  # Ignore very short bursts
                min_silence_duration_ms=100,
                return_seconds=False,  # Return in samples
            )

        # Calculate speech duration
        speech_samples = 0
        for ts in speech_timestamps:
            speech_samples += ts["end"] - ts["start"]

        speech_duration_sec = speech_samples / self.sample_rate
        speech_ratio = (
            speech_duration_sec / total_duration_sec if total_duration_sec > 0 else 0.0
        )

        # Consider silent if <10% speech
        is_silent = speech_ratio < 0.10

        vad_time_ms = (time.time() - start_time) * 1000

        return {
            "is_silent": is_silent,
            "speech_ratio": speech_ratio,
            "speech_duration_sec": speech_duration_sec,
            "total_duration_sec": total_duration_sec,
            "vad_time_ms": vad_time_ms,
            "method": "silero_vad",
            "num_speech_segments": len(speech_timestamps),
            "threshold": threshold,
        }

    def detect_speech(
        self,
        audio: np.ndarray,
        sample_rate: int,
        threshold: float = 0.5,
        max_analysis_sec: float = 60.0,
    ) -> Dict:
        """
        Detect speech in audio. Alias for detect_silence().

        Args:
            audio: Audio array (mono, float32 or int16)
            sample_rate: Audio sample rate
            threshold: VAD probability threshold (0.0-1.0)
            max_analysis_sec: Only analyze first N seconds for long audio

        Returns:
            Dictionary with speech detection results
        """
        return self.detect_silence(audio, sample_rate, threshold, max_analysis_sec)

    def detect_silence_from_file(
        self, file_path: Path, threshold: float = 0.5, max_analysis_sec: float = 60.0
    ) -> Dict:
        """
        Detect silence directly from audio file.

        Args:
            file_path: Path to audio file
            threshold: VAD probability threshold
            max_analysis_sec: Only analyze first N seconds

        Returns:
            Same dictionary as detect_silence()
        """
        import soundfile as sf

        # Read audio file
        audio, sample_rate = sf.read(file_path, dtype="float32")

        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        return self.detect_silence(audio, sample_rate, threshold, max_analysis_sec)

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        import torchaudio.functional as F

        audio_tensor = torch.from_numpy(audio)
        resampled = F.resample(audio_tensor, orig_sr, target_sr)
        return resampled.numpy()

    def get_speech_segments(
        self, audio: np.ndarray, sample_rate: int, threshold: float = 0.5
    ) -> list:
        """
        Get list of speech segments with timestamps.

        Args:
            audio: Audio array
            sample_rate: Audio sample rate
            threshold: VAD threshold

        Returns:
            List of dictionaries with 'start' and 'end' times in seconds
        """
        self._load_model()

        # Ensure float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        # Resample if needed
        if sample_rate != self.sample_rate:
            audio = self._resample(audio, sample_rate, self.sample_rate)

        assert self._utils is not None, "_load_model() must be called first"
        get_speech_timestamps = self._utils[0]

        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio).to(self.device)

            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                self._model,
                sampling_rate=self.sample_rate,
                threshold=threshold,
                return_seconds=True,  # Return in seconds
            )

        return speech_timestamps


# Optional: Fast energy-based detection for very short files
class FastEnergyVAD:
    """
    Ultra-fast energy-based VAD for short audio (<5 seconds).

    Uses RMS energy instead of neural network - 100x faster than Silero
    but less accurate. Good for quick pre-filtering.
    """

    @staticmethod
    def detect_silence(
        audio: np.ndarray,
        sample_rate: int,
        energy_threshold: float = 0.01,
        frame_length_ms: int = 20,
    ) -> Dict:
        """
        Fast energy-based silence detection.

        Args:
            audio: Audio array (mono)
            sample_rate: Sample rate
            energy_threshold: RMS energy threshold (0.0-1.0)
            frame_length_ms: Frame size in milliseconds

        Returns:
            Dictionary with is_silent, speech_ratio, etc.
        """
        start_time = time.time()

        # Ensure float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        # Calculate frame length
        frame_length = int(sample_rate * frame_length_ms / 1000)

        # Calculate RMS energy per frame
        num_frames = len(audio) // frame_length
        speech_frames = 0

        for i in range(num_frames):
            frame = audio[i * frame_length : (i + 1) * frame_length]
            rms = np.sqrt(np.mean(frame**2))
            if rms > energy_threshold:
                speech_frames += 1

        speech_ratio = speech_frames / num_frames if num_frames > 0 else 0.0
        is_silent = speech_ratio < 0.10

        total_duration_sec = len(audio) / sample_rate
        speech_duration_sec = speech_ratio * total_duration_sec
        vad_time_ms = (time.time() - start_time) * 1000

        return {
            "is_silent": is_silent,
            "speech_ratio": speech_ratio,
            "speech_duration_sec": speech_duration_sec,
            "total_duration_sec": total_duration_sec,
            "vad_time_ms": vad_time_ms,
            "method": "energy_rms",
            "threshold": energy_threshold,
        }
