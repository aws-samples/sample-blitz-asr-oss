#!/usr/bin/env python3
"""
AWS Batch entry point for Blitz ASR.

Processes a single audio file from S3, transcribes it using Blitz's
3-tier ASR pipeline, and uploads the results to S3.

Usage:
    python -m blitz.batch.transcribe_job --input-bucket BUCKET --input-key KEY

Environment Variables:
    OUTPUT_BUCKET_NAME: S3 bucket for results (required)
    OUTPUT_PREFIX: S3 key prefix for outputs (default: transcripts/)
    TIER1_MODEL: Tier 1 model (default: nvidia/parakeet-tdt_ctc-110m)
    TIER1_LANGUAGES: Languages for Tier 1 (default: en)
    TIER2_BACKEND: nemo or onnx (default: nemo)
    TIER2_PRECISION: fp32, bfloat16, fp16 (default: bfloat16)
    SKIP_SILENCE: Skip silent files (default: true)
    INCLUDE_TIMESTAMPS: Include timestamps (default: true)
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf

if TYPE_CHECKING:
    from blitz.app_state import ApplicationState
    from blitz.batch.batch_config import BatchConfig
    from blitz.batch.s3_handler import S3Handler

from blitz.utils.audio import (
    chunk_audio,
    cleanup_temp_file,
    merge_chunked_transcripts,
    save_audio_to_temp,
)

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_audio_from_file(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load audio file from local path, with FFmpeg fallback for unsupported formats.

    Handles:
    - Native formats via soundfile (WAV, FLAC, OGG)
    - Other formats via FFmpeg (M4A, MP3, etc.)
    - Stereo to mono conversion
    - Float32 dtype conversion

    Args:
        file_path: Path to local audio file

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    filename = os.path.basename(file_path)

    # Try soundfile first (supports WAV, FLAC, OGG)
    try:
        audio, sr = sf.read(file_path)
        logger.info(
            f"Loaded {filename} via soundfile: sr={sr}, shape={audio.shape}, dtype={audio.dtype}"
        )
    except Exception as e:
        # soundfile failed - use FFmpeg to convert to WAV
        logger.info(f"soundfile failed for {filename}: {e}, trying FFmpeg...")

        # Use NamedTemporaryFile to avoid race conditions (mktemp is deprecated)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_out_path = tmp_file.name

        try:
            # Convert to WAV using FFmpeg
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    file_path,
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-f",
                    "wav",
                    tmp_out_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            # Load converted WAV
            audio, sr = sf.read(tmp_out_path)
            logger.info(
                f"FFmpeg converted {filename}: sr={sr}, shape={audio.shape}, dtype={audio.dtype}"
            )

        finally:
            # Clean up temp file
            if os.path.exists(tmp_out_path):
                os.unlink(tmp_out_path)

    # Convert stereo to mono if needed
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Ensure float32 dtype (NeMo requires float32)
    audio = audio.astype("float32")

    logger.info(
        f"Final audio: sr={sr}, samples={len(audio)}, duration={len(audio) / sr:.2f}s"
    )
    return audio, sr


def calculate_processing_costs(
    total_processing_time: float,
    audio_duration_sec: float,
    lang_detect_time_ms: float,
    transcribe_time_seconds: float,
) -> Dict[str, Any]:
    """
    Calculate processing metrics and estimated GPU costs.

    Args:
        total_processing_time: Total time from request to response (seconds)
        audio_duration_sec: Duration of audio file (seconds)
        lang_detect_time_ms: Time spent on language detection (milliseconds)
        transcribe_time_seconds: Time spent on transcription (seconds)

    Returns:
        Dictionary with processing metrics and cost estimates
    """
    # G6.xlarge pricing
    G6_XLARGE_PRICING = {
        "on_demand": 0.8048,
        "spot": 0.35,
        "savings_1yr": 0.4890,
        "savings_3yr": 0.3215,
    }

    # Calculate real-time factor (RTF)
    rtf = total_processing_time / audio_duration_sec if audio_duration_sec > 0 else 0

    # Calculate cost per 1000 minutes of audio
    processing_time_for_1000_min = 60000 * rtf
    processing_hours_for_1000_min = processing_time_for_1000_min / 3600

    costs_per_1000_min = {
        "on_demand": processing_hours_for_1000_min * G6_XLARGE_PRICING["on_demand"],
        "spot": processing_hours_for_1000_min * G6_XLARGE_PRICING["spot"],
        "savings_1yr": processing_hours_for_1000_min * G6_XLARGE_PRICING["savings_1yr"],
        "savings_3yr": processing_hours_for_1000_min * G6_XLARGE_PRICING["savings_3yr"],
    }

    # Calculate percentage breakdown
    lang_detect_seconds = lang_detect_time_ms / 1000
    lang_detect_pct = (
        (lang_detect_seconds / total_processing_time * 100)
        if total_processing_time > 0
        else 0
    )
    transcribe_pct = (
        (transcribe_time_seconds / total_processing_time * 100)
        if total_processing_time > 0
        else 0
    )

    return {
        "total_time_sec": round(total_processing_time, 3),
        "rtf": round(rtf, 4),
        "lang_detect_ms": round(lang_detect_time_ms, 2),
        "transcribe_sec": round(transcribe_time_seconds, 3),
        "cost_per_1000_min": {
            "on_demand": round(costs_per_1000_min["on_demand"], 2),
            "spot": round(costs_per_1000_min["spot"], 2),
            "savings_1yr": round(costs_per_1000_min["savings_1yr"], 2),
            "savings_3yr": round(costs_per_1000_min["savings_3yr"], 2),
        },
        "breakdown_pct": {
            "lang_detect": round(lang_detect_pct, 1),
            "transcribe": round(transcribe_pct, 1),
        },
    }


def transcribe_s3_file(
    input_bucket: str,
    input_key: str,
    output_bucket: str,
    output_prefix: str,
    config: "BatchConfig",
    state: "ApplicationState",
    s3_handler: "S3Handler",
) -> Dict[str, Any]:
    """
    Main transcription workflow for batch processing.

    1. Download audio from S3
    2. Load audio with FFmpeg fallback
    3. Run VAD (skip if silence detected and config.skip_silence)
    4. Detect language
    5. Route to Parakeet or Whisper
    6. Calculate metrics
    7. Format output JSON
    8. Upload results to S3

    Args:
        input_bucket: S3 bucket containing input audio
        input_key: S3 key of input audio file
        output_bucket: S3 bucket for results
        output_prefix: S3 key prefix for outputs
        config: Batch configuration
        state: Application state with models and services
        s3_handler: S3 handler for downloads/uploads

    Returns:
        Dictionary with transcription results and metadata
    """
    # Validate required services are initialized
    if state.language_detection_service is None:
        raise RuntimeError("language_detection_service not initialized")
    if state.tier2_model is None:
        raise RuntimeError("tier2_model not initialized")
    if state.whisper_model is None:
        raise RuntimeError("whisper_model not initialized")

    start_time = time.time()
    filename = os.path.basename(input_key)

    logger.info(f"Processing s3://{input_bucket}/{input_key}")

    # Download audio from S3
    local_path = s3_handler.download_audio(input_bucket, input_key)

    try:
        # Load audio
        audio, sample_rate = load_audio_from_file(local_path)
        duration_sec = len(audio) / sample_rate

        # Run VAD to check for silence
        vad_result = None
        if state.vad_service:
            vad_result = state.vad_service.detect_speech(audio, sample_rate)
            logger.info(f"VAD: speech_ratio={vad_result.get('speech_ratio', 0):.2%}")

            # Skip processing if silent
            if config.skip_silence:
                speech_ratio = vad_result.get("speech_ratio", 1.0)
                if speech_ratio < config.silence_threshold:
                    logger.info(
                        f"Skipping silent file (speech_ratio={speech_ratio:.2%} < {config.silence_threshold:.2%})"
                    )
                    return _create_silence_result(
                        input_bucket,
                        input_key,
                        filename,
                        duration_sec,
                        sample_rate,
                        vad_result,
                    )

        # Language detection
        lang_detect_start = time.time()
        detection_result = state.language_detection_service.detect(
            audio, sample_rate, language_hint=None
        )
        lang_detect_time_ms = (time.time() - lang_detect_start) * 1000

        language = detection_result.language
        confidence = detection_result.confidence
        method = detection_result.detection_method
        top_languages = detection_result.top_languages

        logger.info(f"Language: {language} (conf={confidence:.2%}, method={method})")

        # Determine engine: 3-tier routing
        # Tier 1 (CTC-110M, fastest) -> Tier 2 (Parakeet 0.6B) -> Tier 3 (Whisper)
        tier1_langs = config.get_tier1_languages()
        tier2_langs = config.get_tier2_languages()

        if tier1_langs and state.tier1_model is not None and language in tier1_langs:
            engine = "tier1"
        elif tier2_langs and language in tier2_langs:
            engine = "tier2"
        else:
            engine = "tier3"

        if engine == "tier1":
            logger.info(f"Engine: tier1 ({config.tier1_model})")
        elif engine == "tier2":
            logger.info(f"Engine: tier2 ({config.tier2_model})")
        else:
            logger.info(f"Engine: tier3 ({config.tier3_model})")

        # Transcribe
        transcribe_start = time.time()
        if engine == "tier1":
            text, timestamps = _transcribe_with_parakeet(
                audio, sample_rate, duration_sec, config, state, use_tier1=True
            )
        elif engine == "tier2":
            text, timestamps = _transcribe_with_parakeet(
                audio, sample_rate, duration_sec, config, state, use_tier1=False
            )
        else:
            text, timestamps = _transcribe_with_whisper(
                audio, sample_rate, language, config, state
            )
        transcribe_time = time.time() - transcribe_start

        # Calculate total processing time and costs
        total_time = time.time() - start_time
        metrics = calculate_processing_costs(
            total_time, duration_sec, lang_detect_time_ms, transcribe_time
        )

        # Build transcription dict (with optional timestamps)
        transcription_data: Dict[str, Any] = {
            "text": text,
            "language": {
                "code": language,
                "confidence": round(confidence, 4),
                "method": method,
                "top_languages": {
                    k: round(v, 4) for k, v in (top_languages or {}).items()
                },
            },
            "engine": engine,
        }
        if config.include_timestamps and timestamps:
            transcription_data["timestamps"] = timestamps

        # Build result JSON
        result: Dict[str, Any] = {
            "input": {"bucket": input_bucket, "key": input_key, "filename": filename},
            "transcription": transcription_data,
            "audio": {
                "duration_sec": round(duration_sec, 2),
                "sample_rate": sample_rate,
            },
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Add VAD info
        if vad_result:
            result["vad"] = {
                "is_silent": False,
                "speech_ratio": round(vad_result.get("speech_ratio", 1.0), 4),
            }

        # Add metrics if requested
        if config.include_metrics:
            result["metrics"] = metrics

        # Generate output key
        output_key = config.get_output_key(input_key)

        # Upload result to S3
        result["output"] = {"bucket": output_bucket, "key": output_key}
        s3_handler.upload_json(result, output_bucket, output_key)

        logger.info(
            f"Transcription complete: {len(text)} chars, RTF={metrics['rtf']:.4f}"
        )
        return result

    finally:
        # Clean up downloaded file
        if os.path.exists(local_path):
            os.unlink(local_path)


def _transcribe_with_parakeet(
    audio: np.ndarray,
    sample_rate: int,
    duration_sec: float,
    config: "BatchConfig",
    state: "ApplicationState",
    use_tier1: bool = False,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Transcribe using Parakeet model (Tier 1 CTC-110M or Tier 2 0.6B).

    Args:
        audio: Audio samples as numpy array
        sample_rate: Audio sample rate
        duration_sec: Audio duration in seconds
        config: Batch configuration
        state: Application state with models
        use_tier1: If True, use Tier 1 CTC-110M (fastest)

    Returns:
        Tuple of (text, timestamps_dict)
    """
    import torch
    import torchaudio

    # Select model based on tier
    if use_tier1:
        model = state.tier1_model
        if model is None:
            raise RuntimeError("Tier 1 model requested but not loaded")
    else:
        model = state.tier2_model
        if model is None:
            raise RuntimeError("Tier 2 model not loaded")

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        waveform = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        audio = resampler(waveform).squeeze(0).numpy()
        sample_rate = 16000

    # Set attention mode for long audio
    model.set_attention_mode(duration_sec)

    # Determine approach based on duration
    if duration_sec <= config.short_audio_threshold_sec:
        # Direct transcription for short audio
        output = model.transcribe(audio, timestamps=config.include_timestamps)

        # Extract text and timestamps
        if output:
            text = output[0].text if hasattr(output[0], "text") else str(output[0])
            timestamps = None
            if config.include_timestamps and hasattr(output[0], "timestamp"):
                ts_data = output[0].timestamp
                if ts_data and "segment" in ts_data:
                    timestamps = {"segment": ts_data["segment"]}
            return text, timestamps
        return "", None

    # Chunked transcription for longer audio to prevent GPU OOM
    logger.info(
        f"Using chunked transcription: {duration_sec:.1f}s audio, "
        f"{config.chunk_length_sec}s chunks, {config.chunk_overlap_sec}s overlap"
    )

    chunks = chunk_audio(
        audio, sample_rate, config.chunk_length_sec, config.chunk_overlap_sec
    )
    logger.info(f"Split audio into {len(chunks)} chunks")

    chunk_results = []
    for i, (audio_chunk, start_sample, end_sample) in enumerate(chunks):
        # Save chunk to temp file for NeMo processing
        temp_file = save_audio_to_temp(audio_chunk, sample_rate)
        try:
            # Transcribe chunk without timestamps (merging timestamps across chunks is complex)
            output = model.transcribe(temp_file, timestamps=False)
            if output:
                text = output[0].text if hasattr(output[0], "text") else str(output[0])
                chunk_results.append((text, start_sample, end_sample))
                logger.debug(f"Chunk {i + 1}/{len(chunks)}: {len(text)} chars")
        finally:
            cleanup_temp_file(temp_file)

    # Merge chunk transcripts
    merged_text = merge_chunked_transcripts(
        chunk_results, config.chunk_overlap_sec * sample_rate, sample_rate
    )
    logger.info(f"Merged {len(chunk_results)} chunks into {len(merged_text)} chars")

    # No timestamps for chunked transcription (would require complex alignment)
    return merged_text, None


def _transcribe_with_whisper(
    audio: np.ndarray,
    sample_rate: int,
    language: str,
    config: "BatchConfig",
    state: "ApplicationState",
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Transcribe using Whisper model."""
    import torch

    if state.whisper_model is None:
        raise RuntimeError(
            "Whisper model not loaded (tier 3). Caller must validate before invoking."
        )

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        import torchaudio

        waveform = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        audio = resampler(waveform).squeeze(0).numpy()

    # Process audio through Whisper
    audio_tensor = torch.from_numpy(audio).float()

    # Use Whisper model's transcribe method
    # Note: After resampling, sample_rate is always 16000
    text, timestamps_data = state.whisper_model.transcribe(
        audio_tensor,
        sample_rate=16000,
        language=language,
        timestamps=config.include_timestamps,
    )

    return text, timestamps_data


def _create_silence_result(
    input_bucket: str,
    input_key: str,
    filename: str,
    duration_sec: float,
    sample_rate: int,
    vad_result: Dict,
) -> Dict[str, Any]:
    """Create result for silent/empty audio file."""
    return {
        "input": {"bucket": input_bucket, "key": input_key, "filename": filename},
        "transcription": {
            "text": "",
            "language": {"code": "unknown", "confidence": 0.0},
            "engine": "none",
        },
        "audio": {"duration_sec": round(duration_sec, 2), "sample_rate": sample_rate},
        "vad": {
            "is_silent": True,
            "speech_ratio": round(vad_result.get("speech_ratio", 0.0), 4),
        },
        "status": "skipped_silence",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main():
    """CLI entry point for AWS Batch."""
    parser = argparse.ArgumentParser(
        description="Blitz ASR Batch Transcription Job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
    OUTPUT_BUCKET_NAME    S3 bucket for results (required)
    OUTPUT_PREFIX         S3 key prefix for outputs (default: transcripts/)
    TIER1_LANGUAGES       Languages for Tier 1 (default: en)
    TIER2_BACKEND         nemo or onnx (default: nemo)
    TIER2_PRECISION       fp32, bfloat16, fp16 (default: bfloat16)
    SKIP_SILENCE          Skip silent files (default: true)
    INCLUDE_TIMESTAMPS    Include timestamps (default: true)
        """,
    )
    parser.add_argument(
        "--input-bucket", "-b", required=True, help="S3 bucket containing input audio"
    )
    parser.add_argument(
        "--input-key", "-k", required=True, help="S3 key (path) to input audio file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 70)
    logger.info("Blitz ASR Batch Transcription Job")
    logger.info("=" * 70)

    # Import after argument parsing to speed up --help
    from blitz.batch.batch_config import BatchConfig
    from blitz.batch.s3_handler import S3Handler
    from blitz.app_state import ApplicationState

    # Load configuration
    config = BatchConfig.from_environment()
    config.validate_batch_config()

    logger.info(f"Input: s3://{args.input_bucket}/{args.input_key}")
    logger.info(f"Output: s3://{config.output_bucket}/{config.output_prefix}")
    logger.info(
        f"Tier 2 Backend: {config.tier2_backend}, Precision: {config.tier2_precision}"
    )

    # Initialize S3 handler
    s3_handler = S3Handler()

    # Initialize application state (models and services)
    logger.info("Initializing models...")
    state = ApplicationState.create_with_defaults()
    logger.info("Models initialized")

    # Run transcription
    try:
        result = transcribe_s3_file(
            input_bucket=args.input_bucket,
            input_key=args.input_key,
            output_bucket=config.output_bucket,
            output_prefix=config.output_prefix,
            config=config,
            state=state,
            s3_handler=s3_handler,
        )

        logger.info("=" * 70)
        logger.info(f"Status: {result['status']}")
        if result["status"] == "success":
            logger.info(f"Text length: {len(result['transcription']['text'])} chars")
            logger.info(f"Language: {result['transcription']['language']['code']}")
            logger.info(f"Engine: {result['transcription']['engine']}")
            if "metrics" in result:
                logger.info(f"RTF: {result['metrics']['rtf']:.4f}")
        logger.info(
            f"Output: s3://{result['output']['bucket']}/{result['output']['key']}"
        )
        logger.info("=" * 70)

        sys.exit(0)

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
