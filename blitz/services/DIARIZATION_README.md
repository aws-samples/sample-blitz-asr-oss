# Speaker Diarization for Blitz ASR

Adds "who spoke when" capability using pyannote/speaker-diarization-3.1.

## Architecture

```
Audio → [Diarization: pyannote 3.1] → Speaker segments (who, when)
     → [Transcription: existing blitz pipeline] → Word timestamps (what, when)
     → [Merge] → Speaker-labeled transcript (who said what, when)
```

## Prerequisites

1. **Install pyannote.audio** (add to requirements.txt or Dockerfile):
   ```
   pyannote.audio>=3.1.0
   ```

2. **Get a HuggingFace token** with access to pyannote models:
   - Go to: https://huggingface.co/pyannote/speaker-diarization-3.1
   - Accept the user agreement
   - Get your token: https://huggingface.co/settings/tokens

3. **Set the token** in `.env`:
   ```bash
   HF_TOKEN=hf_your_token_here
   BLITZ_DIARIZATION_ENABLED=true
   ```

## GPU Memory Impact

| Component | VRAM |
|-----------|------|
| Existing models (Tier 1+2+3) | ~9.5 GB |
| Pyannote diarization 3.1 | ~1.0 GB |
| **Total with diarization** | **~10.5 GB** |

Still fits comfortably on a 24GB L4 GPU. ✅

## API Usage

### With diarization enabled, add `diarize=true` parameter:

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@meeting.mp3" \
  -F "format=json" \
  -F "diarize=true"
```

### Optional parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `diarize` | bool | Enable speaker detection (default: false) |
| `num_speakers` | int | Exact number of speakers (if known) |
| `min_speakers` | int | Minimum speakers expected |
| `max_speakers` | int | Maximum speakers expected (default: 10) |

### Response (with diarization):

```json
{
  "text": "Hello how are you? I'm doing great thanks.",
  "language": {"code": "en", "confidence": 0.99, ...},
  "engine": "tier1",
  "duration_sec": 15.2,
  "diarization": {
    "num_speakers": 2,
    "processing_time_sec": 2.1,
    "method": "pyannote-3.1",
    "segments": [
      {
        "speaker": "SPEAKER_00",
        "start": 0.5,
        "end": 3.2,
        "text": "Hello how are you?"
      },
      {
        "speaker": "SPEAKER_01",
        "start": 3.8,
        "end": 6.1,
        "text": "I'm doing great thanks."
      }
    ],
    "transcript": "[00:00.5 - 00:03.2] SPEAKER_00: Hello how are you?\n[00:03.8 - 00:06.1] SPEAKER_01: I'm doing great thanks."
  }
}
```

## Integration into server.py

Add to the `/transcribe` endpoint, after transcription completes:

```python
from blitz.services.diarization import DiarizationService, format_diarized_json, format_diarized_transcript

# Initialize (at startup, in app_state.py)
diarization_service = DiarizationService(device="cuda")

# In /transcribe endpoint:
diarize = request.form.get("diarize", "false").lower() == "true"
if diarize and diarization_service.is_available:
    # Run diarization
    diar_result = diarization_service.diarize(
        audio, sample_rate=16000,
        num_speakers=num_speakers,  # from request params
    )

    # Merge with word timestamps
    if word_timestamps:
        merged = diarization_service.merge_with_transcription(diar_result, word_timestamps)
    else:
        merged = diar_result.segments

    # Add to response
    response["diarization"] = {
        "num_speakers": diar_result.num_speakers,
        "processing_time_sec": diar_result.processing_time_sec,
        "method": diar_result.method,
        "segments": format_diarized_json(merged),
        "transcript": format_diarized_transcript(merged),
    }
```

## Deployment on EC2

```bash
# 1. Install pyannote in the container
docker exec blitz pip install pyannote.audio

# 2. Set HuggingFace token
echo "HF_TOKEN=hf_your_token" >> .env

# 3. Copy diarization.py to EC2
# (already in blitz/services/ if source is volume-mounted)

# 4. Restart
docker restart blitz

# 5. Test
curl -X POST http://localhost:8000/transcribe \
  -F "file=@meeting.mp3" \
  -F "format=json" \
  -F "diarize=true" \
  -F "num_speakers=2"
```

## Performance Expectations

| Audio Length | Diarization Time | Notes |
|---|---|---|
| 10s | ~0.5s | Quick |
| 1 min | ~1-2s | Typical meeting segment |
| 10 min | ~5-10s | Conference call |
| 1 hour | ~30-60s | Full meeting |

Diarization runs **in parallel** with transcription on the same GPU (different memory segments). Total processing time ≈ max(transcription_time, diarization_time) + merge_time.

## Limitations

- **No speaker identification**: Labels are "SPEAKER_00", "SPEAKER_01" etc. — not actual names
- **No cross-file consistency**: Speaker labels reset per request (SPEAKER_00 in file A ≠ SPEAKER_00 in file B)
- **Minimum audio**: Needs at least 1-2 seconds of speech per speaker for reliable detection
- **Overlapping speech**: May be less accurate when multiple speakers talk simultaneously
