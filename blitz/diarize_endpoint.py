"""
/diarize endpoint using whisper-diarization (MSDD-based).

Calls the proven whisper-diarization pipeline:
  faster-whisper → NeMo MSDD → CTC forced alignment → punctuation
  
Returns speaker-labeled transcripts with timestamps.
"""

import io
import os
import time
import json
import uuid
import logging
import tempfile
import subprocess
from typing import Optional

import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse

logger = logging.getLogger("blitz.diarize")

DIARIZE_SCRIPT = "/tmp/whisper-diarization/diarize_parallel.py"


def parse_srt_with_speakers(srt_path: str) -> list:
    """Parse SRT file with speaker labels into structured segments."""
    segments = []
    if not os.path.exists(srt_path):
        return segments
    
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            # Line 1: index, Line 2: timestamps, Line 3+: text with speaker
            time_line = lines[1]
            text = " ".join(lines[2:])
            
            # Parse timestamps: "00:00:07,200 --> 00:00:14,080"
            try:
                parts = time_line.split(" --> ")
                start = srt_time_to_seconds(parts[0].strip())
                end = srt_time_to_seconds(parts[1].strip())
            except (IndexError, ValueError):
                continue
            
            # Parse speaker from text: "Speaker 0: actual text"
            speaker = "Speaker 0"
            actual_text = text
            if ": " in text and text.startswith("Speaker"):
                speaker_part, actual_text = text.split(": ", 1)
                speaker = speaker_part.strip()
            
            segments.append({
                "speaker": speaker,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": actual_text.strip(),
            })
    
    return segments


def srt_time_to_seconds(time_str: str) -> float:
    """Convert SRT timestamp to seconds. Format: HH:MM:SS,mmm"""
    time_str = time_str.replace(",", ".")
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def register_diarize_endpoint(app: FastAPI):
    """Register the /diarize endpoint."""

    @app.post("/diarize")
    async def diarize_endpoint(
        file: UploadFile = File(...),
        format: str = Form(default="json"),
        language: Optional[str] = Form(default=None),
    ):
        """
        Transcribe audio with speaker diarization.
        
        Uses whisper-diarization pipeline (faster-whisper + NeMo MSDD)
        for state-of-the-art speaker detection and word alignment.
        
        Parameters:
            file: Audio file (MP3, WAV, M4A, FLAC)
            format: "json" (structured) or "text" (plain transcript)
            language: Language code (optional, auto-detected if omitted)
        
        Returns:
            Speaker-labeled transcript with timestamps
        """
        start_time = time.time()
        
        # Verify script exists
        if not os.path.exists(DIARIZE_SCRIPT):
            raise HTTPException(
                status_code=503,
                detail="Diarization pipeline not installed. Run: git clone https://github.com/MahmoudAshraf97/whisper-diarization /tmp/whisper-diarization"
            )
        
        # Save uploaded file to temp location
        temp_dir = tempfile.mkdtemp(prefix="blitz_diar_")
        audio_ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
        temp_audio = os.path.join(temp_dir, f"input{audio_ext}")
        
        try:
            content = await file.read()
            with open(temp_audio, "wb") as f:
                f.write(content)
            
            # Get audio duration
            try:
                audio_data, sample_rate = sf.read(temp_audio)
                if len(audio_data.shape) > 1:
                    duration_sec = len(audio_data) / sample_rate
                else:
                    duration_sec = len(audio_data) / sample_rate
            except Exception:
                duration_sec = 0
            
            logger.info(f"Diarize request: {duration_sec:.1f}s audio, file={file.filename}")
            
            # Build command
            cmd = [
                "python", DIARIZE_SCRIPT,
                "--whisper-model", "large-v3",
                "-a", temp_audio,
                "--device", "cuda",
            ]
            if language:
                cmd.extend(["--language", language])
            
            # Run diarization pipeline
            logger.info(f"Running whisper-diarization pipeline...")
            result = subprocess.run(
                cmd,
                capture_output=True, encoding="utf-8", errors="replace",
                
                timeout=600,  # 10 min timeout
                cwd="/tmp/whisper-diarization",
            )
            
            if result.returncode != 0:
                logger.error(f"Diarization failed: {result.stderr[-500:]}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Diarization pipeline failed: {result.stderr[-200:]}"
                )
            
            # Find output files
            base_name = os.path.splitext(temp_audio)[0]
            srt_file = base_name + ".srt"
            txt_file = base_name + ".txt"
            
            processing_time = time.time() - start_time
            
            # Read text output
            full_text = ""
            if os.path.exists(txt_file):
                with open(txt_file, "r", encoding="utf-8") as f:
                    full_text = f.read()
            
            # Parse SRT for structured segments
            segments = parse_srt_with_speakers(srt_file)
            
            # Count unique speakers
            speakers = set(s["speaker"] for s in segments)
            
            # Return based on format
            if format == "text":
                return JSONResponse(content={"transcript": full_text})
            
            # Read SRT content
            srt_content = ""
            if os.path.exists(srt_file):
                with open(srt_file, "r", encoding="utf-8") as f:
                    srt_content = f.read()
            
            response = {
                "text": full_text,
                "duration_sec": round(duration_sec, 3),
                "diarization": {
                    "num_speakers": len(speakers),
                    "speakers": sorted(list(speakers)),
                    "segments": segments,
                    "method": "whisper-diarization (MSDD)",
                },
                "srt": srt_content,
                "metrics": {
                    "total_processing_time_sec": round(processing_time, 3),
                    "audio_duration_sec": round(duration_sec, 3),
                    "real_time_factor": round(processing_time / duration_sec, 4) if duration_sec > 0 else 0,
                },
            }
            
            return JSONResponse(content=response)
        
        finally:
            # Cleanup temp files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info("Registered /diarize endpoint (whisper-diarization MSDD pipeline)")
