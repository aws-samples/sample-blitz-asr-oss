"""
Integrated server using modular architecture.

This server demonstrates the refactored codebase using:
- ApplicationState for dependency injection
- Service-based architecture for business logic
- Generic batch workers with engine-specific processors
- Pydantic schemas for type safety
- Centralized configuration

Original server.py: 1,706 lines
Integrated server: ~400 lines (76% reduction)
"""

import asyncio
import io
import os
import pathlib
import re
import subprocess
import tempfile
import time
import uuid
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from blitz import ApplicationState, get_app_state, create_app
from blitz.schemas.api import TranscriptionResponse, SRTResponse, HealthResponse, ModelsInfoResponse
from blitz.utils.audio import get_audio_duration
from blitz.utils.formatting import generate_srt_content, generate_vtt_content
from blitz.banner import print_banner, print_startup_info


# Create FastAPI app with integrated lifespan management
app = create_app()


_UNSAFE_FILENAME_CHARS = re.compile(r'[\r\n"\\/]')


def _safe_attachment_filename(name: Optional[str], fallback: str) -> str:
    """Sanitize a user-supplied filename for use in Content-Disposition headers.

    Strips CR/LF/quote/backslash/slash to prevent header injection and path
    traversal when the uploaded filename is reflected back to the client.
    """
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", (name or "").strip())
    return cleaned or fallback


# ============================================================================
# Helper Functions
# ============================================================================

# G6.xlarge pricing (g6.xlarge with NVIDIA L4 GPU) - high end of ranges
G6_XLARGE_PRICING = {
    "on_demand": 0.8048,      # On-Demand rate (us-east-1)
    "spot": 0.35,             # Spot (upper end: 0.20-0.35/hr range)
    "savings_1yr": 0.4890,    # 1-yr Standard RI (us-east-1)
    "savings_3yr": 0.3215,    # 3-yr Standard RI (us-east-1)
}


def calculate_processing_costs(
    total_processing_time: float,
    audio_duration_sec: float,
    lang_detect_time_ms: float,
    transcribe_time_seconds: float
) -> dict:
    """
    Calculate processing metrics and estimated GPU costs.

    Args:
        total_processing_time: Total time from request to response (seconds)
        audio_duration_sec: Duration of audio file (seconds)
        lang_detect_time_ms: Time spent on language detection (milliseconds)
        transcribe_time_seconds: Time spent on transcription (seconds)

    Returns:
        Dictionary with processing metrics and cost estimates including:
        - Real-time factor (RTF)
        - Cost per 1000 minutes for all pricing models
        - Cost breakdown percentages
    """
    # Calculate real-time factor (RTF): processing_time / audio_duration
    rtf = total_processing_time / audio_duration_sec if audio_duration_sec > 0 else 0

    # Calculate processing time needed for 1000 minutes (60,000 seconds) of audio
    processing_time_for_1000_min = 60000 * rtf
    processing_hours_for_1000_min = processing_time_for_1000_min / 3600

    # Calculate cost per 1000 minutes of audio for each pricing model
    costs_per_1000_min = {
        "on_demand": processing_hours_for_1000_min * G6_XLARGE_PRICING["on_demand"],
        "spot": processing_hours_for_1000_min * G6_XLARGE_PRICING["spot"],
        "savings_1yr": processing_hours_for_1000_min * G6_XLARGE_PRICING["savings_1yr"],
        "savings_3yr": processing_hours_for_1000_min * G6_XLARGE_PRICING["savings_3yr"],
    }

    # Calculate percentage breakdown of processing time
    lang_detect_seconds = lang_detect_time_ms / 1000
    lang_detect_pct = (lang_detect_seconds / total_processing_time * 100) if total_processing_time > 0 else 0
    transcribe_pct = (transcribe_time_seconds / total_processing_time * 100) if total_processing_time > 0 else 0

    return {
        "total_processing_time_sec": round(total_processing_time, 3),
        "audio_duration_sec": round(audio_duration_sec, 3),
        "real_time_factor": round(rtf, 3),
        "language_detection_time_ms": round(lang_detect_time_ms, 2),
        "transcription_time_sec": round(transcribe_time_seconds, 3),
        "cost_per_1000_min_audio_usd": {
            "on_demand": round(costs_per_1000_min["on_demand"], 2),
            "spot": round(costs_per_1000_min["spot"], 2),
            "savings_1yr": round(costs_per_1000_min["savings_1yr"], 2),
            "savings_3yr": round(costs_per_1000_min["savings_3yr"], 2),
        },
        "cost_breakdown_percent": {
            "language_detection": round(lang_detect_pct, 2),
            "transcription": round(transcribe_pct, 2),
        },
        "instance_type": "g6.xlarge",
        "region": "us-east-1"
    }


async def load_audio_file(file: UploadFile) -> tuple[np.ndarray, int]:
    """
    Load audio file from upload, with FFmpeg fallback for unsupported formats.

    Handles:
    - Native formats via soundfile (WAV, FLAC, OGG)
    - Other formats via FFmpeg (M4A, MP3, etc.)
    - Stereo to mono conversion
    - Float32 dtype conversion

    Args:
        file: Uploaded audio file

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    data = await file.read()

    # Try soundfile first (supports WAV, FLAC, OGG)
    try:
        audio, sr = sf.read(io.BytesIO(data))
        print(f"📁 Loaded {file.filename} via soundfile: sr={sr}, shape={audio.shape}, dtype={audio.dtype}")
    except Exception as e:
        # soundfile failed - use FFmpeg to convert to WAV
        print(f"📁 soundfile failed for {file.filename}: {e}, trying FFmpeg...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_in:
            tmp_in.write(data)
            tmp_in_path = tmp_in.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            tmp_out_path = tmp_out.name

        try:
            # Convert to WAV using FFmpeg
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", tmp_in_path,
                    "-ar", "16000",
                    "-ac", "1",
                    "-f", "wav",
                    tmp_out_path
                ],
                check=True,
                capture_output=True,
                text=True
            )

            # Load converted WAV
            audio, sr = sf.read(tmp_out_path)
            print(f"📁 FFmpeg converted {file.filename}: sr={sr}, shape={audio.shape}, dtype={audio.dtype}")

        finally:
            # Clean up temp files
            pathlib.Path(tmp_in_path).unlink(missing_ok=True)
            pathlib.Path(tmp_out_path).unlink(missing_ok=True)

    # Convert stereo to mono if needed
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Ensure float32 dtype (NeMo requires float32)
    audio = audio.astype('float32')

    print(f"📁 Final audio: sr={sr}, samples={len(audio)}, duration={len(audio)/sr:.2f}s, dtype={audio.dtype}")
    return audio, sr


# ============================================================================
# API Routes
# ============================================================================

@app.post("/transcribe", response_model=None)
async def transcribe(
    file: UploadFile = File(...),
    format: str = Form("text"),
    language_hint: Optional[str] = Form(None),
    state: ApplicationState = Depends(get_app_state)
):
    """
    Intelligent ASR endpoint with automatic language detection and micro-batching.

    Process:
    1) Load and preprocess audio (stereo to mono, measure duration)
    2) Detect language with Whisper encoder (unless language_hint provided)
    3) Route to optimal model via async queue:
       - Parakeet: High performance for 25 supported languages
       - Whisper: Multilingual fallback for 99+ languages
    4) Batch workers gather requests within timeout window for efficient GPU utilization
    5) Return response in requested format (text/json/srt/vtt)

    Args:
        file: Audio file (WAV, MP3, FLAC, M4A, etc.)
        format: Response format - "text" (default), "json", "srt", "vtt"
        language_hint: Skip detection if language is known (e.g., "en", "zh")
        state: Application state (injected)

    Returns:
        Response format depends on 'format' parameter:
        - "text"/"json": Full JSON with text, timestamps, language, confidence, metrics
        - "srt": SRT subtitle file format
        - "vtt": WebVTT subtitle file format
    """
    # Timestamps enabled (NeMo 2.4+ native support)
    timestamps = True

    try:
        # Start total timing
        start_time = time.time()

        # ---- Load and preprocess audio ----
        audio, sr = await load_audio_file(file)
        duration_sec = get_audio_duration(audio, sr)

        # ---- Language detection FIRST (unless hint provided) ----
        lang_detect_start = time.time()

        detection_result = state.language_detection_service.detect(
            audio=audio,
            sample_rate=sr,
            language_hint=language_hint
        )

        lang_detect_time_ms = (time.time() - lang_detect_start) * 1000
        best_lang_code = detection_result.language

        # ---- Voice Activity Detection (ONLY if low language confidence) ----
        # If language confidence is LOW, audio might be silent/unclear → run VAD
        # If language confidence is HIGH, audio is clearly speech → skip VAD
        vad_result = None
        vad_skipped = False

        if detection_result.confidence < state.config.min_language_confidence:
            # Low confidence - run VAD to check for silence
            vad_result = state.vad_service.detect_silence(
                audio=audio,
                sample_rate=sr,
                threshold=0.5,  # VAD threshold
                max_analysis_sec=60.0  # Only analyze first 60 seconds
            )

            # ---- Early return if silence detected ----
            if vad_result["is_silent"]:
                total_processing_time = time.time() - start_time

                # Return minimal response for silent audio
                response_data = {
                    "text": "",  # Empty text for silent audio
                    "language": {
                        "code": best_lang_code,
                        "confidence": detection_result.confidence,
                        "top_languages": detection_result.top_languages,
                        "detection_method": detection_result.detection_method,
                        "detection_model": detection_result.detection_model
                    },
                    "vad": {
                        "is_silent": vad_result["is_silent"],
                        "speech_ratio": vad_result["speech_ratio"],
                        "speech_duration_sec": vad_result["speech_duration_sec"],
                        "vad_time_ms": vad_result["vad_time_ms"],
                        "method": vad_result["method"],
                        "num_speech_segments": vad_result["num_speech_segments"]
                    },
                    "engine": "none",  # No engine used
                    "duration_sec": duration_sec,
                    "timestamps": None,
                    "metrics": {
                        "total_processing_time_sec": round(total_processing_time, 3),
                        "audio_duration_sec": round(duration_sec, 3),
                        "real_time_factor": round(total_processing_time / duration_sec if duration_sec > 0 else 0, 3),
                        "language_detection_time_ms": round(lang_detect_time_ms, 2),
                        "transcription_time_sec": 0.0,  # Skipped
                        "estimated_gpu_cost_usd": 0.0,  # No GPU used
                        "gpu_instance": "none",
                        "gpu_cost_per_hour_usd": 0.0,
                        "chunk_approach": "skipped_silent",
                        "chunk_size_sec": None,
                        "chunk_count": 0
                    }
                }

                # Return in requested format
                if format == "srt" or format == "vtt":
                    # No timestamps for silent audio - return error
                    raise HTTPException(
                        status_code=400,
                        detail=f"{format.upper()} format not available for silent audio (no speech detected)"
                    )
                else:
                    return response_data
        else:
            # High confidence - skip VAD, proceed directly to transcription
            vad_skipped = True

        # ---- 3-Tier Model Routing ----
        # Tier 1: Fastest (CTC-110M, default: English)
        # Tier 2: Fast multilingual (Parakeet 0.6B, default: EU languages)
        # Tier 3: Catch-all (Whisper, everything else)
        tier1_langs = state.config.get_tier1_languages()
        tier2_langs = state.config.get_tier2_languages()

        if tier1_langs and state.tier1_model is not None and best_lang_code in tier1_langs:
            engine = "tier1"
        elif tier2_langs and best_lang_code in tier2_langs:
            engine = "tier2"
        else:
            # Tier 3: Whisper catch-all
            engine = "tier3"

        # ---- Submit job to queue and wait for result ----
        transcribe_start = time.time()

        # Create job
        job_id = str(uuid.uuid4())
        job = {
            "job_id": job_id,
            "engine": engine,
            "audio": audio,
            "sr": sr,
            "duration_sec": duration_sec,
            "timestamps": timestamps,
        }

        # Add engine-specific fields for Tier 3 (Whisper)
        if engine == "tier3":
            job["audio_tensor"] = torch.from_numpy(audio).float()
            job["target_lang"] = best_lang_code

        # Create Future for result
        result_future = asyncio.Future()
        state.pending_results[job_id] = result_future

        # Submit to queue
        await state.transcription_queue.put(job)

        # Wait for result (with timeout)
        try:
            result = await asyncio.wait_for(result_future, timeout=300.0)  # 5 minute timeout
        except asyncio.TimeoutError:
            # Cleanup
            state.pending_results.pop(job_id, None)
            raise HTTPException(status_code=504, detail="Transcription timeout after 5 minutes")

        # Extract result
        transcript = result["text"]
        timestamp_data = result.get("timestamps")
        chunk_approach = result.get("chunk_approach", "unknown")
        chunk_size_sec = result.get("chunk_size_sec")
        chunk_count = result.get("chunk_count", 1)

        # Cleanup
        state.pending_results.pop(job_id, None)

        # Capture transcription timing
        transcribe_time_seconds = time.time() - transcribe_start

        # ---- Calculate metrics ----
        total_processing_time = time.time() - start_time

        metrics = calculate_processing_costs(
            total_processing_time,
            duration_sec,
            lang_detect_time_ms,
            transcribe_time_seconds
        )

        # ---- Build response ----
        # Top-3 languages for observability
        TOP_K = 3
        top_k_languages = dict(
            sorted(
                detection_result.top_languages.items(),
                key=lambda x: x[1],
                reverse=True
            )[:TOP_K]
        )

        # Build VAD info (or null if skipped due to high language confidence)
        vad_info = None
        if vad_result is not None:
            vad_info = {
                "is_silent": vad_result["is_silent"],
                "speech_ratio": vad_result["speech_ratio"],
                "speech_duration_sec": vad_result["speech_duration_sec"],
                "vad_time_ms": vad_result["vad_time_ms"],
                "method": vad_result["method"],
                "num_speech_segments": vad_result["num_speech_segments"]
            }
        else:
            # VAD was skipped (high language confidence)
            vad_info = {
                "is_silent": False,
                "speech_ratio": None,
                "speech_duration_sec": None,
                "vad_time_ms": 0.0,
                "method": "skipped_high_confidence",
                "num_speech_segments": None
            }

        # Base response data
        response_data = {
            "text": transcript,
            "language": {
                "code": best_lang_code,
                "confidence": detection_result.confidence,
                "top_languages": top_k_languages,
                "detection_method": detection_result.detection_method,
                "detection_model": detection_result.detection_model
            },
            "vad": vad_info,
            "engine": engine,
            "duration_sec": duration_sec,
            "timestamps": timestamp_data,
            "metrics": {
                **metrics,
                "chunk_approach": chunk_approach,
                "chunk_size_sec": chunk_size_sec,
                "chunk_count": chunk_count
            }
        }

        # ---- Return in requested format ----
        if format == "srt":
            if not timestamp_data or not timestamp_data.get("segment"):
                raise HTTPException(
                    status_code=400,
                    detail="SRT format requires timestamps, but timestamps are not available for this transcription"
                )

            srt_content = generate_srt_content(timestamp_data["segment"])
            safe_name = _safe_attachment_filename(file.filename, "transcript")
            return Response(
                content=srt_content,
                media_type="text/plain",
                headers={
                    "Content-Disposition": f'attachment; filename="{safe_name}.srt"'
                }
            )

        elif format == "vtt":
            if not timestamp_data or not timestamp_data.get("segment"):
                raise HTTPException(
                    status_code=400,
                    detail="VTT format requires timestamps, but timestamps are not available for this transcription"
                )

            vtt_content = generate_vtt_content(timestamp_data["segment"])
            safe_name = _safe_attachment_filename(file.filename, "transcript")
            return Response(
                content=vtt_content,
                media_type="text/vtt",
                headers={
                    "Content-Disposition": f'attachment; filename="{safe_name}.vtt"'
                }
            )

        else:
            # Default: JSON response
            return response_data

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health(state: ApplicationState = Depends(get_app_state)):
    """Health check endpoint with model and configuration information."""
    models = {
        "tier2": {
            "loaded": True,
            "device": state.config.device,
            "model": state.config.tier2_model
        },
        "tier3": {
            "loaded": True,
            "device": state.config.device,
            "model": state.config.tier3_model
        },
        "voxlingua": {
            "loaded": True,
            "device": "cpu"
        }
    }

    # Add Tier 1 model info if enabled
    if state.tier1_model is not None:
        models["tier1"] = {
            "loaded": True,
            "device": state.config.device,
            "model": state.config.tier1_model,
            "languages": list(state.config.get_tier1_languages())
        }

    return {
        "status": "healthy",
        "models": models
    }


@app.get("/models", response_model=ModelsInfoResponse)
async def models_info(state: ApplicationState = Depends(get_app_state)):
    """Get detailed information about loaded models and configuration."""
    tier2_languages = state.tier2_model.supported_languages
    whisper_languages = state.whisper_model.supported_languages

    response = {
        "tier2": {
            "model": state.config.tier2_model,
            "languages": sorted(tier2_languages),
            "configured_languages": sorted(state.config.get_tier2_languages()),
            "optimizations": {
                "local_attention_threshold_sec": state.config.local_attention_threshold_sec,
                "short_audio_threshold_sec": state.config.short_audio_threshold_sec
            }
        },
        "tier3": {
            "model": state.config.tier3_model,
            "languages": sorted(whisper_languages),
            "features": ["language_detection", "multilingual_transcription", "catch_all"]
        }
    }

    # Add Tier 1 model info if enabled
    if state.tier1_model is not None:
        tier1_languages = state.tier1_model.supported_languages
        response["tier1"] = {
            "model": state.config.tier1_model,
            "languages": sorted(tier1_languages),
            "configured_languages": sorted(state.config.get_tier1_languages()),
            "performance": {
                "rtfx": 5345,
                "wer_percent": 7.49,
                "params_millions": 110,
                "speedup_vs_tier2": "60%"
            }
        }

    return response


if __name__ == "__main__":
    import uvicorn

    # Display branded banner
    print_banner()
    print_startup_info(host="0.0.0.0", port=8000)

    uvicorn.run(app, host="0.0.0.0", port=8000)
