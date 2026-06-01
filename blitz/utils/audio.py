"""
Audio processing utilities for Blitz transcription service.

Functions for audio loading, resampling, chunking, and duration calculation.
"""

import logging
import os
import tempfile
import uuid
from typing import List, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)


def get_audio_duration(audio: Union[np.ndarray, torch.Tensor], sr: int) -> float:
    """
    Calculate audio duration in seconds.

    Args:
        audio: Audio array (numpy or torch tensor)
        sr: Sample rate in Hz

    Returns:
        Duration in seconds
    """
    if isinstance(audio, torch.Tensor):
        return audio.shape[0] / sr
    return len(audio) / sr


def resample_audio(
    audio: Union[np.ndarray, torch.Tensor], source_sr: int, target_sr: int = 16000
) -> Tuple[np.ndarray, int]:
    """
    Resample audio to target sample rate.

    Args:
        audio: Audio array (numpy or torch tensor)
        source_sr: Source sample rate in Hz
        target_sr: Target sample rate in Hz (default: 16000 for NeMo)

    Returns:
        Tuple of (resampled_audio_array, target_sample_rate)
    """
    if source_sr == target_sr:
        # No resampling needed
        if isinstance(audio, torch.Tensor):
            return audio.numpy(), target_sr
        return audio, target_sr

    # Convert to torch tensor if needed
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.from_numpy(audio).float()
    else:
        audio_tensor = audio.float()

    # Resample using torchaudio
    audio_tensor = torchaudio.functional.resample(audio_tensor, source_sr, target_sr)

    # Convert back to numpy
    return audio_tensor.numpy(), target_sr


def save_audio_to_temp(audio: Union[np.ndarray, torch.Tensor], sr: int) -> str:
    """
    Save audio array to temporary WAV file for NeMo processing.

    NeMo Parakeet expects 16kHz audio - automatically resamples if needed.

    Args:
        audio: Audio array (numpy or torch tensor)
        sr: Sample rate in Hz

    Returns:
        Path to temporary WAV file

    Note:
        Caller is responsible for cleanup using cleanup_temp_file()
    """
    # Resample to 16kHz if needed (NeMo Parakeet requirement)
    audio_array, final_sr = resample_audio(audio, sr, target_sr=16000)

    # Create temporary file
    tmp_dir = tempfile.gettempdir()
    tmp_file = os.path.join(tmp_dir, f"audio-{uuid.uuid4().hex}.wav")

    # Write audio to file
    sf.write(tmp_file, audio_array, final_sr)

    return tmp_file


def cleanup_temp_file(path: str) -> None:
    """
    Safely remove temporary file.

    Args:
        path: Path to temporary file

    Note:
        Silently handles errors (file already deleted, permission issues, etc.)
    """
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError as exc:
        logger.debug("cleanup_temp_file(%s) ignored: %s", path, exc)


def chunk_audio(
    audio: Union[np.ndarray, torch.Tensor],
    sr: int,
    chunk_length_sec: int,
    overlap_sec: int,
) -> List[Tuple[Union[np.ndarray, torch.Tensor], int, int]]:
    """
    Split long audio into overlapping chunks to prevent GPU OOM.

    Args:
        audio: Audio array (numpy or torch tensor)
        sr: Sample rate in Hz
        chunk_length_sec: Length of each chunk in seconds
        overlap_sec: Overlap between chunks in seconds

    Returns:
        List of tuples: (chunk_audio, start_sample, end_sample)
        - chunk_audio: Audio chunk (same type as input)
        - start_sample: Starting sample index in original audio
        - end_sample: Ending sample index in original audio

    Example:
        >>> audio = np.random.randn(48000)  # 3 seconds at 16kHz
        >>> chunks = chunk_audio(audio, sr=16000, chunk_length_sec=2, overlap_sec=1)
        >>> # Returns 2 chunks: [0:32000] and [16000:48000]
    """
    chunk_samples = chunk_length_sec * sr
    overlap_samples = overlap_sec * sr
    stride = chunk_samples - overlap_samples

    chunks = []
    total_samples = audio.shape[0] if isinstance(audio, torch.Tensor) else len(audio)

    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = audio[start:end]
        chunks.append((chunk, start, end))

        # Move to next chunk
        start += stride

        # If we're within one overlap of the end, we're done
        if end >= total_samples:
            break

    return chunks


def merge_chunked_transcripts(
    chunk_results: List[Tuple[str, int, int]], overlap_samples: int, sr: int
) -> str:
    """
    Merge transcripts from overlapping chunks.

    Handles overlap regions by concatenating transcripts with space separation.

    Args:
        chunk_results: List of (transcript, start_sample, end_sample) tuples
        overlap_samples: Number of overlapping samples between chunks
        sr: Sample rate in Hz

    Returns:
        Merged transcript as a single string

    Note:
        Simple concatenation strategy - in overlap regions, the second chunk's
        text takes precedence. Works well for most cases but may duplicate some
        words at chunk boundaries.

    Example:
        >>> chunks = [
        ...     ("Hello world", 0, 16000),
        ...     ("world how are", 8000, 24000),
        ...     ("are you", 16000, 32000)
        ... ]
        >>> merge_chunked_transcripts(chunks, overlap_samples=8000, sr=16000)
        'Hello world world how are are you'
    """
    if not chunk_results:
        return ""

    if len(chunk_results) == 1:
        return chunk_results[0][0]

    # Simple concatenation strategy: take full text from each chunk
    # In overlap regions, the second chunk's text takes precedence
    merged_text = chunk_results[0][0]

    for i in range(1, len(chunk_results)):
        text = chunk_results[i][0]

        # Simple space-join strategy
        if merged_text and text:
            merged_text = merged_text.rstrip() + " " + text.lstrip()
        elif text:
            merged_text = text

    return merged_text
