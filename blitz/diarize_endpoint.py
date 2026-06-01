"""
Standalone /diarize endpoint for speaker-labeled transcription.

Uses NeMo's native TitaNet model for speaker embeddings — no additional
dependencies beyond nemo_toolkit[asr] which is already installed.

Import this in server.py with:
    from blitz.diarize_endpoint import register_diarize_endpoint
    register_diarize_endpoint(app)
"""

import io
import time
import asyncio
import uuid
import logging
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse

from blitz.app_state import ApplicationState, get_app_state
from blitz.services.diarization import (
    DiarizationService,
    format_diarized_json,
    format_diarized_transcript,
)

logger = logging.getLogger("blitz.diarize")

# Singleton diarization service (loaded on first request for lazy init)
_diarization_service: Optional[DiarizationService] = None


def _get_diarization_service() -> DiarizationService:
    global _diarization_service
    if _diarization_service is None:
        logger.info("Initializing diarization service (first request)...")
        _diarization_service = DiarizationService(device="cuda")
    return _diarization_service


def register_diarize_endpoint(app: FastAPI):
    """Register the /diarize endpoint on the FastAPI app."""

    @app.post("/diarize")
    async def diarize_endpoint(
        file: UploadFile = File(...),
        format: str = Form(default="json"),
        num_speakers: Optional[int] = Form(default=None),
        min_speakers: Optional[int] = Form(default=None),
        max_speakers: Optional[int] = Form(default=None),
        state: ApplicationState = Depends(get_app_state),
    ):
        """
        Transcribe audio with speaker diarization.

        Combines blitz transcription (with word timestamps) and NeMo TitaNet
        speaker embeddings + clustering to produce a speaker-labeled transcript.

        Parameters:
            file: Audio file (MP3, WAV, M4A, FLAC)
            format: Output format - "json" or "text" (default: json)
            num_speakers: Exact number of speakers (if known, improves accuracy)
            min_speakers: Minimum expected speakers
            max_speakers: Maximum expected speakers (default: 10)

        Returns:
            JSON with transcription + speaker-labeled segments
        """
        start_time = time.time()

        # Check diarization availability
        diar_service = _get_diarization_service()
        if not diar_service.is_available:
            raise HTTPException(
                status_code=503,
                detail="Diarization model not available. Check server logs for loading errors."
            )

        # Read audio file
        try:
            content = await file.read()
            audio_data, sample_rate = sf.read(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read audio file: {str(e)}")

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import torchaudio
            waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            audio_data = waveform.squeeze(0).numpy()
            sample_rate = 16000

        duration_sec = len(audio_data) / sample_rate
        logger.info(f"Diarize request: {duration_sec:.1f}s audio, file={file.filename}")

        # Step 1: Run transcription (using existing blitz pipeline)
        transcribe_start = time.time()
        try:
            # Detect language
            detection_result = state.language_detection_service.detect(audio_data, sample_rate, None)
            best_lang_code = detection_result.language
            lang_detect_time_ms = detection_result.detection_time_ms

            # Determine engine (same routing logic as /transcribe)
            tier1_langs = state.config.get_tier1_languages()
            tier2_langs = state.config.get_tier2_languages()

            if tier1_langs and state.tier1_model is not None and best_lang_code in tier1_langs:
                engine = "tier1"
            elif tier2_langs and best_lang_code in tier2_langs:
                engine = "tier2"
            else:
                engine = "tier3"

            # Submit to batch queue
            job = {
                "id": str(uuid.uuid4()),
                "engine": engine,
                "audio": audio_data,
                "sample_rate": sample_rate,
                "timestamps": True,  # Need word timestamps for merge
            }
            if engine == "tier3":
                job["audio_tensor"] = torch.from_numpy(audio_data).float()
                job["target_lang"] = best_lang_code

            result_future = asyncio.Future()
            state.pending_results[job["id"]] = result_future
            await state.transcription_queue.put(job)
            transcription_result = await asyncio.wait_for(result_future, timeout=300)
            del state.pending_results[job["id"]]

        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Transcription timed out")
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

        transcribe_time = time.time() - transcribe_start

        # Extract text and word timestamps
        text = transcription_result.get("text", "")
        word_timestamps = None
        timestamps_data = transcription_result.get("timestamps")
        if timestamps_data and isinstance(timestamps_data, dict):
            word_timestamps = timestamps_data.get("word", [])

        # Step 2: Run diarization
        diar_start = time.time()
        try:
            diar_result = diar_service.diarize(
                audio_data,
                sample_rate=sample_rate,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        except Exception as e:
            logger.error(f"Diarization failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Diarization failed: {str(e)}")
        diar_time = time.time() - diar_start

        # Step 3: Merge transcription with diarization
        if word_timestamps:
            merged_segments = diar_service.merge_with_transcription(diar_result, word_timestamps)
        else:
            # No word timestamps — return diarization segments without text
            merged_segments = diar_result.segments

        total_time = time.time() - start_time

        # Build response
        if format == "text":
            transcript = format_diarized_transcript(merged_segments)
            return JSONResponse(content={"transcript": transcript})

        response = {
            "text": text,
            "language": {
                "code": best_lang_code,
                "confidence": detection_result.confidence,
                "detection_method": detection_result.method,
            },
            "engine": engine,
            "duration_sec": round(duration_sec, 3),
            "diarization": {
                "num_speakers": diar_result.num_speakers,
                "processing_time_sec": diar_result.processing_time_sec,
                "method": diar_result.method,
                "segments": format_diarized_json(merged_segments),
                "transcript": format_diarized_transcript(merged_segments),
            },
            "metrics": {
                "total_processing_time_sec": round(total_time, 3),
                "transcription_time_sec": round(transcribe_time, 3),
                "diarization_time_sec": round(diar_time, 3),
                "language_detection_time_ms": round(lang_detect_time_ms, 2),
                "audio_duration_sec": round(duration_sec, 3),
            },
        }

        return JSONResponse(content=response)

    logger.info("Registered /diarize endpoint (NeMo TitaNet + clustering)")
