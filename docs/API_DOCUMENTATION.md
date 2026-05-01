# API Documentation - Blitz Transcription Service

## Overview

Blitz is a high-performance automatic speech recognition (ASR) service that intelligently routes transcription requests between three models:
- **Parakeet-TDT-CTC-110M**: Fastest English-only model (RTFx 5345, 110M params)
- **Parakeet-TDT-0.6B-v3**: High-quality model for 25 European languages (RTFx 3332)
- **Whisper Large-v3-Turbo**: Multilingual transformer model for 100+ languages

The service features:
- ✅ **Voice Activity Detection (VAD)**: Silero VAD for silence detection before transcription
- ✅ **Silent Audio Optimization**: Skip transcription entirely for silent files (80% faster, zero GPU cost)
- ✅ Automatic language detection with fallback mechanism (Whisper + VoxLingua107)
- ✅ Micro-batching for 30-50% throughput improvement
- ✅ Native timestamp support (NeMo 2.4+)
- ✅ Multiple output formats (text, JSON, SRT, VTT)
- ✅ Optimized for long-form audio (up to 60 minutes)

---

## Base URL

```
http://localhost:8000
```

For production deployments, replace with your domain or IP address.

---

## Endpoints

### 1. POST /transcribe

Transcribe audio files with automatic language detection and intelligent model routing.

#### Request

**Content-Type:** `multipart/form-data`

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | ✅ Yes | - | Audio file (WAV, MP3, FLAC, M4A, OGG, etc.) |
| `format` | String | ❌ No | `"text"` | Response format: `"text"`, `"json"`, `"srt"`, `"vtt"` |
| `language_hint` | String | ❌ No | `null` | ISO 639-1 language code (e.g., `"en"`, `"zh"`) to skip detection |

**Supported Audio Formats:**
- WAV, FLAC, OGG (via soundfile)
- MP3, M4A, AAC (via FFmpeg conversion)
- Any format supported by FFmpeg

**Supported Languages:**
- **Parakeet (25 languages):** bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk
- **Whisper (99+ languages):** en, zh, de, es, ru, ko, fr, ja, pt, tr, pl, ca, nl, ar, sv, it, id, hi, fi, vi, he, uk, el, ms, cs, ro, da, hu, ta, no, th, ur, hr, bg, lt, la, mi, ml, cy, sk, te, fa, lv, bn, sr, az, sl, kn, et, mk, br, eu, is, hy, ne, mn, bs, kk, sq, sw, gl, mr, pa, si, km, sn, yo, so, af, oc, ka, be, tg, sd, gu, am, yi, lo, uz, fo, ht, ps, tk, nn, mt, sa, lb, my, bo, tl, mg, as, tt, ln, ha, ba, jw, su

#### Response Formats

##### 1. Text Format (`format=text`)

**Content-Type:** `application/json`

Returns plain transcription text:

```json
{
  "text": "Hello world, this is a test transcription."
}
```

##### 2. JSON Format (`format=json`)

**Content-Type:** `application/json`

Returns full transcription with metadata, timestamps, and metrics:

```json
{
  "text": "Hello world, this is a test transcription.",
  "language": {
    "code": "en",
    "confidence": 0.95,
    "top_languages": {
      "en": 0.95,
      "de": 0.03,
      "fr": 0.02
    },
    "detection_method": "whisper_encoder",
    "detection_model": "openai/whisper-large-v3-turbo"
  },
  "vad": {
    "is_silent": false,
    "speech_ratio": 0.85,
    "speech_duration_sec": 8.9,
    "vad_time_ms": 45.2,
    "method": "silero_vad",
    "num_speech_segments": 12
  },
  "engine": "parakeet",
  "duration_sec": 10.5,
  "timestamps": {
    "segment": [
      {
        "text": "Hello world,",
        "start_time": 0.0,
        "end_time": 1.2
      },
      {
        "text": "this is a test transcription.",
        "start_time": 1.2,
        "end_time": 3.5
      }
    ]
  },
  "metrics": {
    "total_processing_time_sec": 0.524,
    "audio_duration_sec": 10.5,
    "real_time_factor": 0.05,
    "language_detection_time_ms": 45.23,
    "transcription_time_sec": 0.478,
    "estimated_gpu_cost_usd": 0.000073,
    "gpu_instance": "g6.xlarge",
    "gpu_cost_per_hour_usd": 0.505,
    "chunk_approach": "direct",
    "chunk_size_sec": null,
    "chunk_count": 1
  }
}
```

##### 3. SRT Format (`format=srt`)

**Content-Type:** `text/plain`

**Content-Disposition:** `attachment; filename="audio.wav.srt"`

Returns SubRip (SRT) subtitle file:

```srt
1
00:00:00,000 --> 00:00:01,200
Hello world,

2
00:00:01,200 --> 00:00:03,500
this is a test transcription.
```

##### 4. VTT Format (`format=vtt`)

**Content-Type:** `text/vtt`

**Content-Disposition:** `attachment; filename="audio.wav.vtt"`

Returns WebVTT subtitle file:

```vtt
WEBVTT

00:00:00.000 --> 00:00:01.200
Hello world,

00:00:01.200 --> 00:00:03.500
this is a test transcription.
```

##### 5. Silent Audio Response

**Content-Type:** `application/json`

When VAD detects silent audio (<10% speech ratio), transcription is skipped entirely for optimal performance:

```json
{
  "text": "",
  "language": {
    "code": "en",
    "confidence": 0.0,
    "top_languages": {},
    "detection_method": "skipped_silent_audio",
    "detection_model": "none"
  },
  "vad": {
    "is_silent": true,
    "speech_ratio": 0.0,
    "speech_duration_sec": 0.0,
    "vad_time_ms": 133.5,
    "method": "silero_vad",
    "num_speech_segments": 0
  },
  "engine": "none",
  "duration_sec": 10.0,
  "timestamps": null,
  "metrics": {
    "total_processing_time_sec": 0.15,
    "audio_duration_sec": 10.0,
    "real_time_factor": 0.015,
    "language_detection_time_ms": 0.0,
    "transcription_time_sec": 0.0,
    "estimated_gpu_cost_usd": 0.0,
    "gpu_instance": "none",
    "gpu_cost_per_hour_usd": 0.0,
    "chunk_approach": "skipped_silent",
    "chunk_size_sec": null,
    "chunk_count": 0
  }
}
```

**Benefits:**
- ~80% faster processing (150ms vs 700ms)
- Zero GPU cost
- No hallucinated transcription from noise

#### Example Requests

**cURL - Basic transcription:**
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "format=json"
```

**cURL - With language hint:**
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "format=json" \
  -F "language_hint=en"
```

**cURL - Download SRT subtitles:**
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "format=srt" \
  -o output.srt
```

**Python - Basic transcription:**
```python
import requests

with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": f},
        data={"format": "json"}
    )

result = response.json()
print(f"Transcription: {result['text']}")
print(f"Language: {result['language']['code']}")
print(f"Confidence: {result['language']['confidence']}")
```

**Python - With language hint:**
```python
import requests

with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": f},
        data={
            "format": "json",
            "language_hint": "en"
        }
    )

result = response.json()
print(f"Transcription: {result['text']}")
print(f"RTF: {result['metrics']['real_time_factor']}")
```

#### Response Schema Details

**LanguageInfo:**
| Field | Type | Description |
|-------|------|-------------|
| `code` | string | ISO 639-1 language code (e.g., "en", "zh") |
| `confidence` | float | Confidence score (0.0 to 1.0) |
| `top_languages` | object | Top 3 detected languages with scores |
| `detection_method` | string | Method used: "whisper_encoder", "voxlingua", "hint", or "skipped_silent_audio" |
| `detection_model` | string | Model name used for detection |

**VADInfo (Voice Activity Detection):**
| Field | Type | Description |
|-------|------|-------------|
| `is_silent` | boolean | `true` if audio is mostly silent (<10% speech ratio) |
| `speech_ratio` | float | Ratio of audio containing speech (0.0 to 1.0) |
| `speech_duration_sec` | float | Duration of speech detected in seconds |
| `vad_time_ms` | float | Time spent on VAD analysis in milliseconds |
| `method` | string | VAD method used (currently "silero_vad") |
| `num_speech_segments` | int | Number of speech segments detected |

**Note:** When `is_silent=true`, the transcription is skipped entirely (saves ~80% processing time and GPU cost). The response will have `text=""`, `engine="none"`, and zero cost metrics.

**TimestampSegment:**
| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Transcribed text segment |
| `start_time` | float | Start time in seconds |
| `end_time` | float | End time in seconds |

**ProcessingMetrics:**
| Field | Type | Description |
|-------|------|-------------|
| `total_processing_time_sec` | float | Total time from request to response |
| `audio_duration_sec` | float | Duration of audio file |
| `real_time_factor` | float | Processing time / audio duration (lower is better) |
| `language_detection_time_ms` | float | Time spent on language detection |
| `transcription_time_sec` | float | Time spent on transcription |
| `estimated_gpu_cost_usd` | float | Estimated GPU cost for this request |
| `gpu_instance` | string | GPU instance type (e.g., "g6.xlarge") |
| `gpu_cost_per_hour_usd` | float | GPU hourly cost |
| `chunk_approach` | string | Processing strategy: "direct", "temp", or "chunked" |
| `chunk_size_sec` | float/null | Chunk size if chunking was used |
| `chunk_count` | int | Number of chunks processed |

#### Error Responses

**400 Bad Request - Invalid format:**
```json
{
  "detail": "SRT format requires timestamps, but timestamps are not available for this transcription"
}
```

**400 Bad Request - Silent audio with subtitle format:**
```json
{
  "detail": "SRT format not available for silent audio (no speech detected)"
}
```

**413 Payload Too Large:**
```json
{
  "detail": "File too large. Maximum size: 100MB"
}
```

**504 Gateway Timeout:**
```json
{
  "detail": "Transcription timeout after 5 minutes"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Transcription failed: CUDA out of memory"
}
```

---

### 2. GET /health

Health check endpoint with model and configuration information.

#### Request

No parameters required.

#### Response

**Content-Type:** `application/json`

```json
{
  "status": "healthy",
  "models": {
    "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
    "whisper": "openai/whisper-large-v3-turbo"
  },
  "config": {
    "short_audio_threshold_sec": 600,
    "local_attention_threshold_sec": 1440,
    "parakeet_languages": 25,
    "whisper_languages": 99
  }
}
```

#### Example Request

**cURL:**
```bash
curl http://localhost:8000/health
```

**Python:**
```python
import requests

response = requests.get("http://localhost:8000/health")
health = response.json()

if health["status"] == "healthy":
    print("✅ Service is healthy")
    print(f"Parakeet: {health['models']['parakeet']}")
    print(f"Whisper: {health['models']['whisper']}")
else:
    print("❌ Service is unhealthy")
```

---

### 3. GET /models

Get detailed information about loaded models and their capabilities.

#### Request

No parameters required.

#### Response

**Content-Type:** `application/json`

```json
{
  "parakeet": {
    "model": "nvidia/parakeet-tdt-0.6b-v3",
    "languages": [
      "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr",
      "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro",
      "ru", "sk", "sl", "sv", "uk"
    ],
    "optimizations": {
      "local_attention_threshold_sec": 1440,
      "short_audio_threshold_sec": 600
    }
  },
  "whisper": {
    "model": "openai/whisper-large-v3-turbo",
    "languages": [
      "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo",
      "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es",
      "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "he",
      "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jw",
      "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo", "lt",
      "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my",
      "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro",
      "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr",
      "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr",
      "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh"
    ],
    "features": [
      "language_detection",
      "multilingual_transcription"
    ]
  }
}
```

#### Example Request

**cURL:**
```bash
curl http://localhost:8000/models
```

**Python:**
```python
import requests

response = requests.get("http://localhost:8000/models")
models = response.json()

print(f"Parakeet supports {len(models['parakeet']['languages'])} languages")
print(f"Whisper supports {len(models['whisper']['languages'])} languages")
print(f"Parakeet languages: {', '.join(models['parakeet']['languages'])}")
```

---

## Processing Pipeline

### 1. Audio Loading
- Supports WAV, FLAC, OGG (native), MP3, M4A (via FFmpeg)
- Automatic stereo to mono conversion
- Resampling to 16kHz (if needed)

### 2. Voice Activity Detection (VAD)
- **Method:** Silero VAD (CPU-based for efficiency)
- **Purpose:** Detect silent audio before expensive transcription
- **Threshold:** <10% speech ratio = silent
- **Analysis:** First 60 seconds (for long files)
- **Early Return:** Skip transcription entirely for silent audio
  - ~80% faster processing
  - Zero GPU cost
  - Empty text response with `engine="none"`

### 3. Language Detection
- **Method 1 (Primary):** Whisper encoder detection (GPU)
  - Fast: ~150-200ms for typical audio
  - Accurate: 95%+ confidence for clear audio
  - Uses already-loaded Whisper model (no extra memory)
  - Best accuracy for English and common languages

- **Method 2 (Fallback):** VoxLingua107 (CPU)
  - Activated when Whisper confidence < 70%
  - Supports 107 languages
  - CPU-based, doesn't block GPU

- **Method 3 (Hint):** Skip detection if `language_hint` provided
  - Fastest: 0ms detection time
  - Use when language is known

**Note:** Whisper is primary because testing showed VoxLingua has false positives
(English detected as Welsh) and is ~6x slower (~900ms vs ~150ms).

### 4. Model Routing

```
┌─────────────────────┐
│ Language Detected   │
└──────────┬──────────┘
           │
           ├─── In Parakeet languages? ─── Yes ──► Parakeet (CTC, fastest)
           │
           └─── In Whisper languages? ─── Yes ──► Whisper (Transformer)
                                         │
                                         No ──► Whisper with English fallback
```

**Parakeet (25 languages):**
- Faster inference (CTC model)
- Lower memory usage
- Optimized for European languages
- Direct array processing for short audio

**Whisper (99+ languages):**
- Broader language support
- Transformer-based
- Better for complex audio
- 30-second chunking strategy

### 5. Micro-Batching

Requests are gathered into batches for efficient GPU utilization:

- **Batch timeout:** 15ms (configurable)
- **Max batch size:** 4 requests (configurable)
- **Throughput improvement:** 30-50% for concurrent requests

### 6. Processing Strategies

#### Short Audio (< 600s)
- **Strategy:** Direct array processing
- **Memory:** Low (no temp files)
- **Speed:** Fastest

#### Medium Audio (600s - 1440s)
- **Strategy:** Temp file processing
- **Memory:** Medium (single temp file)
- **Speed:** Fast

#### Long Audio (> 1440s)
- **Strategy:** Chunking + local attention
- **Memory:** Low (processes in chunks)
- **Speed:** Moderate (but prevents OOM)

### 7. Timestamp Extraction

**Native support (NeMo 2.4+):**
- Word-level timestamps from Parakeet
- Segment-level timestamps from Whisper
- Automatic SRT/VTT generation

---

## Performance Characteristics

### Latency

**Real-Time Factor (RTF)** - Processing time / Audio duration:
- **Target:** RTF < 0.1 (10x faster than real-time)
- **Typical:** RTF 0.03-0.08 depending on audio length

**Breakdown (30-second audio, g6.xlarge):**
- Audio loading: 50-100ms
- Language detection: 40-50ms
- Transcription: 800-1500ms
- **Total:** ~1000-1700ms (RTF ~0.05)

**Latency sources:**
- Network upload time (depends on bandwidth)
- Batch gathering (0-15ms)
- Model inference (dominant)

### Throughput

**Sequential requests:** ~1-2 requests/second
**Concurrent requests (batch=4):** ~3-4 requests/second
**Improvement:** 2-3x with micro-batching

### Memory Usage

**GPU Memory (g6.xlarge - 24GB):**
- Parakeet model: ~2.5GB
- Whisper model: ~6GB
- VoxLingua model: ~0.5GB
- Working memory: ~2-3GB per request
- **Total:** ~15-20GB peak

**Recommendations:**
- g6.xlarge (24GB): MAX_BATCH_SIZE=2-4
- g5.xlarge (24GB): MAX_BATCH_SIZE=4
- p4d.24xlarge (80GB): MAX_BATCH_SIZE=8

### Cost Estimation

**g6.xlarge pricing (us-east-1):**
- On-Demand: $0.505/hour
- Spot: ~$0.15-0.30/hour

**Cost per request (30-second audio):**
- Processing time: ~1.5 seconds
- Cost: ~$0.0002 (on-demand)
- **At scale:** $0.10-0.20 per hour of audio transcribed

---

## Best Practices

### 1. Use Language Hints When Possible

Skip language detection for known languages:
```python
# Faster: Skip detection
requests.post(url, files={"file": f}, data={"language_hint": "en"})

# Slower: Auto-detection
requests.post(url, files={"file": f})
```

### 2. Batch Requests

Send concurrent requests to benefit from micro-batching:
```python
import concurrent.futures
import requests

files = ["audio1.wav", "audio2.wav", "audio3.wav", "audio4.wav"]

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for file_path in files:
        with open(file_path, "rb") as f:
            future = executor.submit(
                requests.post,
                "http://localhost:8000/transcribe",
                files={"file": f},
                data={"format": "json"}
            )
            futures.append(future)

    results = [f.result().json() for f in futures]
```

### 3. Choose Appropriate Format

- **Text:** Simplest, smallest response
- **JSON:** Full metadata, timestamps, metrics
- **SRT/VTT:** Direct subtitle download

### 4. Handle Errors Gracefully

```python
import requests
from requests.exceptions import RequestException

try:
    response = requests.post(url, files={"file": f}, timeout=300)
    response.raise_for_status()
    result = response.json()
except requests.exceptions.Timeout:
    print("Request timed out after 5 minutes")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e.response.status_code} - {e.response.json()['detail']}")
except RequestException as e:
    print(f"Request failed: {e}")
```

### 5. Monitor Performance

Track these metrics:
```python
result = response.json()
rtf = result["metrics"]["real_time_factor"]
confidence = result["language"]["confidence"]

if rtf > 0.2:
    print(f"⚠️  High RTF: {rtf} (expected < 0.1)")

if confidence < 0.7:
    print(f"⚠️  Low confidence: {confidence}")
```

---

## Rate Limiting

Currently, there are no enforced rate limits. However, for optimal performance:
- Limit concurrent requests to 2-4x your `MAX_BATCH_SIZE`
- For very long audio (>30 minutes), allow 5+ minutes for processing
- Monitor GPU memory usage and adjust batch size accordingly

---

## Support & Troubleshooting

### Common Issues

**1. "CUDA out of memory"**
- Reduce `MAX_BATCH_SIZE` environment variable
- Lower `LOCAL_ATTENTION_THRESHOLD_SEC` for long audio
- Process shorter audio segments

**2. "Request timeout"**
- Increase timeout for very long audio files
- Default timeout: 5 minutes (300 seconds)

**3. "Invalid audio format"**
- Ensure FFmpeg is installed for MP3/M4A support
- Convert audio to WAV if issues persist

**4. Low confidence scores**
- Check audio quality (background noise, clarity)
- Verify language is in supported list
- Try providing `language_hint` if you know the language

---

## Changelog

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for version history and migration guides.

---

## OpenAPI Documentation

Interactive API documentation is available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

These provide:
- Interactive API testing
- Complete schema documentation
- Example requests and responses
- Type definitions for all models
