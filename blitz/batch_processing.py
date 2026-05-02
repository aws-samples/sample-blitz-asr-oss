"""
Engine-specific batch processing functions for Parakeet and Whisper.

These functions contain the core batch processing logic that is used by
the generic BatchWorker class. They handle model-specific optimizations
like chunking, attention mode switching, and timestamp extraction.
"""

import logging
from typing import Dict, List

import torch
import torchaudio

logger = logging.getLogger(__name__)

from blitz.utils.audio import (
    chunk_audio,
    merge_chunked_transcripts,
    save_audio_to_temp,
    cleanup_temp_file,
)


def aggressive_cleanup(*tensors_to_delete):
    """
    Aggressively cleanup GPU memory to prevent memory leaks.

    Performs:
    1. Explicit deletion of provided tensors
    2. CUDA synchronization to ensure operations complete
    3. GPU cache clearing
    4. Python garbage collection

    Args:
        *tensors_to_delete: Variable number of tensors/objects to explicitly delete
    """
    import gc

    # Delete provided tensors explicitly
    for tensor in tensors_to_delete:
        if tensor is not None:
            del tensor

    # Ensure all CUDA operations complete before cleanup
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Force Python garbage collection
    gc.collect()


def process_parakeet_batch(
    jobs: List[Dict],
    parakeet_model,
    short_audio_threshold_sec: int,
    chunk_length_sec: int,
    chunk_overlap_sec: int,
    local_attention_threshold_sec: int,
) -> List[Dict]:
    """
    Process a batch of Parakeet transcription jobs.

    Implements three processing strategies based on audio duration:
    1. Direct array processing for short audio (<10 min)
    2. Temp file processing for long audio (>10 min)
    3. Chunked processing for very long audio (>8 min chunks)

    Args:
        jobs: List of job dictionaries with keys: audio, sr, duration_sec, timestamps
        parakeet_model: Loaded Parakeet ASR model
        short_audio_threshold_sec: Threshold for direct array vs temp file
        chunk_length_sec: Maximum chunk length for very long audio
        chunk_overlap_sec: Overlap between chunks
        local_attention_threshold_sec: Threshold for switching to local attention

    Returns:
        List of result dictionaries with keys: text, timestamps, chunk_approach, chunk_size_sec, chunk_count
    """
    # Extract batch data
    audio_batch = [j["audio"] for j in jobs]
    sr_batch = [j["sr"] for j in jobs]
    duration_batch = [j["duration_sec"] for j in jobs]
    timestamps_batch = [j.get("timestamps", False) for j in jobs]

    # Resample all audio to 16kHz if needed (NeMo Parakeet requirement)
    target_sr = 16000
    for i in range(len(audio_batch)):
        if sr_batch[i] != target_sr:
            audio_tensor = torch.from_numpy(audio_batch[i]).float()
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sr_batch[i], target_sr
            )
            audio_batch[i] = audio_tensor.numpy()
            sr_batch[i] = target_sr

    # Set attention mode based on longest audio
    max_duration = max(duration_batch)
    _set_parakeet_attention_mode(
        parakeet_model, max_duration, local_attention_threshold_sec
    )

    # Check if any audio needs chunking (very long files)
    needs_chunking = any(d > chunk_length_sec for d in duration_batch)

    with torch.inference_mode():
        if needs_chunking:
            # Process each item individually with chunking if needed
            results = []
            for i, (audio, sr, duration) in enumerate(
                zip(audio_batch, sr_batch, duration_batch)
            ):
                if duration > chunk_length_sec:
                    # Chunk this long audio to prevent OOM
                    chunks = chunk_audio(audio, sr, chunk_length_sec, chunk_overlap_sec)
                    logger.info(
                        "Chunking %.1fs audio into %d chunks",
                        duration, len(chunks),
                    )

                    # Process each chunk (no timestamps)
                    chunk_results = []
                    for audio_chunk, start, end in chunks:
                        # Save chunk to temp file
                        temp_file = save_audio_to_temp(audio_chunk, sr)
                        try:
                            # Wrapper returns (text, timestamps_dict)
                            text, _ = parakeet_model.transcribe(
                                [temp_file],
                                batch_size=1,
                                timestamps=False,  # Disabled for chunked audio
                            )
                            chunk_results.append((text, start, end))
                        finally:
                            cleanup_temp_file(temp_file)

                        # Aggressive cleanup after each chunk
                        aggressive_cleanup(audio_chunk)

                    # Merge chunk results
                    merged_text = merge_chunked_transcripts(
                        chunk_results, chunk_overlap_sec * sr, sr
                    )
                    results.append(
                        {
                            "text": merged_text,
                            "timestamps": None,  # Timestamps not supported for chunked audio
                            "chunk_approach": "chunked",
                            "chunk_size_sec": chunk_length_sec,
                            "chunk_count": len(chunks),
                        }
                    )
                else:
                    # Process normally without chunking
                    use_timestamps = timestamps_batch[i]
                    temp_file = save_audio_to_temp(audio, sr)
                    try:
                        # Wrapper returns (text, timestamps_dict)
                        text, timestamps_data = parakeet_model.transcribe(
                            [temp_file], batch_size=1, timestamps=use_timestamps
                        )

                        results.append(
                            {
                                "text": text,
                                "timestamps": timestamps_data,
                                "chunk_approach": "temp_file_single",
                                "chunk_size_sec": None,
                                "chunk_count": 1,
                            }
                        )
                    finally:
                        cleanup_temp_file(temp_file)
                        aggressive_cleanup(audio, temp_file)

        elif any(d > short_audio_threshold_sec for d in duration_batch):
            # Long audio but not chunking - process each file individually
            results = []
            for i, (audio, sr) in enumerate(zip(audio_batch, sr_batch)):
                use_timestamps = timestamps_batch[i]
                temp_file = save_audio_to_temp(audio, sr)
                try:
                    # Wrapper returns (text, timestamps_dict)
                    text, timestamps_data = parakeet_model.transcribe(
                        [temp_file], batch_size=1, timestamps=use_timestamps
                    )

                    results.append(
                        {
                            "text": text,
                            "timestamps": timestamps_data,
                            "chunk_approach": "temp_file_batch",
                            "chunk_size_sec": None,
                            "chunk_count": 1,
                        }
                    )
                finally:
                    cleanup_temp_file(temp_file)
                    aggressive_cleanup(audio, temp_file)
        else:
            # Direct array processing for short audio
            results = []
            for i, audio in enumerate(audio_batch):
                use_timestamps = timestamps_batch[i]
                # Wrapper returns (text, timestamps_dict)
                text, timestamps_data = parakeet_model.transcribe(
                    [audio],  # Pass as single-element list
                    batch_size=1,
                    timestamps=use_timestamps,
                )

                results.append(
                    {
                        "text": text,
                        "timestamps": timestamps_data,
                        "chunk_approach": "direct_array",
                        "chunk_size_sec": None,
                        "chunk_count": 1,
                    }
                )

    return results


def process_whisper_batch(
    jobs: List[Dict], whisper_model, whisper_processor
) -> List[Dict]:
    """
    Process a batch of Whisper transcription jobs.

    Whisper processes individually (no true batching) and handles long audio
    by chunking into 30-second segments with 1-second overlap.

    Args:
        jobs: List of job dictionaries with keys: audio_tensor, sr, target_lang, duration_sec, timestamps
        whisper_model: Loaded Whisper model
        whisper_processor: Whisper processor for tokenization

    Returns:
        List of result dictionaries with keys: text, timestamps, chunk_approach, chunk_size_sec, chunk_count
    """
    WHISPER_CHUNK_SEC = 30
    sample_rate = 16000

    with torch.inference_mode():
        results = []
        for job in jobs:
            # Whisper processes individually
            audio_tensor = job["audio_tensor"]
            target_lang = job["target_lang"]
            duration_sec = job["duration_sec"]

            # Resample if needed
            if job["sr"] != sample_rate:
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor, job["sr"], sample_rate
                )

            # Encode language token
            forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(
                language=target_lang, task="transcribe"
            )

            timestamps_enabled = job.get("timestamps", False)
            timestamp_data = None

            if duration_sec <= WHISPER_CHUNK_SEC:
                # Short audio - process directly
                inputs = whisper_processor(
                    audio_tensor.numpy(), sampling_rate=sample_rate, return_tensors="pt"
                )

                result = whisper_model.generate(
                    input_features=inputs.input_features.to(
                        device=whisper_model.device, dtype=whisper_model.dtype
                    ),
                    forced_decoder_ids=forced_decoder_ids,
                )

                text = whisper_processor.batch_decode(result, skip_special_tokens=True)[
                    0
                ]

                # Create basic segment timestamp for short audio
                if timestamps_enabled and text.strip():
                    timestamp_data = {
                        "segment": [
                            {"start": 0.0, "end": duration_sec, "segment": text}
                        ],
                        "word": [],
                        "char": [],
                    }

                results.append(
                    {
                        "text": text,
                        "timestamps": timestamp_data,
                        "chunk_approach": "direct",
                        "chunk_size_sec": None,
                        "chunk_count": 1,
                    }
                )

            else:
                # Long audio - chunk into 30-second segments with 1-second overlap
                chunk_samples = WHISPER_CHUNK_SEC * sample_rate
                overlap_samples = 1 * sample_rate  # 1 second overlap

                chunks = []
                chunk_timestamps = []

                for start_sample in range(
                    0, audio_tensor.shape[0], chunk_samples - overlap_samples
                ):
                    end_sample = min(
                        start_sample + chunk_samples, audio_tensor.shape[0]
                    )
                    chunk = audio_tensor[start_sample:end_sample]

                    # Calculate chunk timing
                    start_time = start_sample / sample_rate
                    end_time = end_sample / sample_rate

                    # Process chunk
                    inputs = whisper_processor(
                        chunk.numpy(), sampling_rate=sample_rate, return_tensors="pt"
                    )

                    result = whisper_model.generate(
                        input_features=inputs.input_features.to(
                            device=whisper_model.device, dtype=whisper_model.dtype
                        ),
                        forced_decoder_ids=forced_decoder_ids,
                    )

                    chunk_text = whisper_processor.batch_decode(
                        result, skip_special_tokens=True
                    )[0]
                    chunks.append(chunk_text)

                    # Store chunk timestamp
                    if timestamps_enabled and chunk_text.strip():
                        chunk_timestamps.append(
                            {
                                "start": start_time,
                                "end": end_time,
                                "segment": chunk_text,
                            }
                        )

                    # Stop if we've reached the end
                    if end_sample >= audio_tensor.shape[0]:
                        break

                # Concatenate all chunks with space
                text = " ".join(chunks)

                # Aggregate timestamps if enabled
                if timestamps_enabled and chunk_timestamps:
                    timestamp_data = {
                        "segment": chunk_timestamps,
                        "word": [],
                        "char": [],
                    }

                results.append(
                    {
                        "text": text,
                        "timestamps": timestamp_data,
                        "chunk_approach": "chunked",
                        "chunk_size_sec": WHISPER_CHUNK_SEC,
                        "chunk_count": len(chunks),
                    }
                )

    return results


def _set_parakeet_attention_mode(
    parakeet_model, duration_sec: float, local_attention_threshold_sec: int
):
    """
    Set Parakeet attention mode based on audio duration.

    For very long audio (>24 min by default), switches to local attention
    to prevent GPU OOM.

    Note: This only works with NeMo models. ONNX models don't support
    dynamic attention mode switching.

    Args:
        parakeet_model: Loaded Parakeet ASR model
        duration_sec: Audio duration in seconds
        local_attention_threshold_sec: Threshold for switching to local attention
    """
    # ONNX models don't have cfg attribute - skip attention mode setting
    if not hasattr(parakeet_model, "cfg"):
        if duration_sec > local_attention_threshold_sec:
            logger.warning(
                "ONNX backend doesn't support local attention mode for %.1fs audio",
                duration_sec,
            )
        return

    from omegaconf import open_dict

    if duration_sec > local_attention_threshold_sec:
        # Very long audio - use local attention to prevent OOM
        with open_dict(parakeet_model.cfg.encoder):
            for layer_cfg in parakeet_model.cfg.encoder.get("layers", []):
                if hasattr(layer_cfg, "self_attention") and hasattr(
                    layer_cfg.self_attention, "att_context_size"
                ):
                    layer_cfg.self_attention.att_context_size = [
                        64,
                        64,
                    ]  # Local attention window
        logger.info("Switched to local attention for %.1fs audio", duration_sec)
    else:
        # Normal audio - use full attention
        with open_dict(parakeet_model.cfg.encoder):
            for layer_cfg in parakeet_model.cfg.encoder.get("layers", []):
                if hasattr(layer_cfg, "self_attention") and hasattr(
                    layer_cfg.self_attention, "att_context_size"
                ):
                    layer_cfg.self_attention.att_context_size = [
                        -1,
                        -1,
                    ]  # Full attention
