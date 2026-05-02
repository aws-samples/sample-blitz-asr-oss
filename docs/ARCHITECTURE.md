# Blitz Audio Transcription Flow

## System Architecture with Performance Metrics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CLIENT REQUEST                                    │
│                                                                              │
│  • HTTP POST /transcribe                                                    │
│  • Multipart form: audio file + format + language_hint (optional)           │
│  • Supported formats: WAV, MP3, M4A, FLAC                                   │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI ENDPOINT                                     │
│                         (blitz/server.py)                                    │
│                                                                              │
│  Performance: ~1-2ms routing overhead                                       │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AUDIO LOADING & VALIDATION                              │
│                                                                              │
│  • Load audio file into memory                                              │
│  • Convert to numpy array (16kHz mono)                                      │
│  • Calculate audio duration                                                 │
│                                                                              │
│  Performance: ~50-100ms for typical file                                    │
│  Test case: 24-second WAV file (0.73 MB)                                    │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LANGUAGE DETECTION                                      │
│                                                                              │
│  Step 1: Whisper primary detection (GPU-based)                              │
│          • Whisper large-v3-turbo                                            │
│          • 100+ languages supported                                          │
│          • Performance: ~200-300ms                                           │
│          • High accuracy detection                                           │
│                                                                              │
│  Step 2: VoxLingua107 fallback (if Whisper confidence < 0.70)               │
│          • CPU-based, 107 languages                                          │
│          • Performance: ~150ms                                               │
│          • Used for low-confidence edge cases                                │
│                                                                              │
│  Typical detection time: 200-300ms (Whisper only, 90%+ of cases)            │
│  Fallback detection time: 350-450ms (Whisper + VoxLingua, rare)             │
│                                                                              │
│  Accuracy: Whisper primary provides superior detection accuracy              │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   VOICE ACTIVITY DETECTION (VAD)                             │
│                      (conditional, for uncertain audio)                      │
│                                                                              │
│  Trigger: Only runs if language_confidence < min_language_confidence         │
│           (High confidence = clearly speech → skip VAD)                      │
│           (Low confidence = might be silence/noise → check VAD)              │
│                                                                              │
│  Process:                                                                    │
│  • Analyzes first 60 seconds of audio                                       │
│  • Detects speech segments using Silero VAD                                 │
│  • Calculates speech ratio and duration                                     │
│                                                                              │
│  Outcomes:                                                                   │
│  • Speech detected → Continue to transcription                              │
│  • Silence detected → Early return with "engine": "none"                    │
│                       (saves GPU resources, ~0.7s response)                 │
│                                                                              │
│  Performance: ~50-100ms when triggered                                      │
│  Skip rate: ~90%+ (most audio has high language confidence)                 │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHUNKING LOGIC                                       │
│                      (for long audio only)                                   │
│                                                                              │
│  Thresholds:                                                                │
│  • Short audio (<600s): No chunking, direct processing                      │
│  • Medium audio (600-1440s): Chunked, standard attention                    │
│  • Long audio (>1440s): Chunked, local attention mode                       │
│                                                                              │
│  Chunk parameters:                                                          │
│  • chunk_length_sec: 480 (8 minutes)                                        │
│  • chunk_overlap_sec: 1 (prevents word cutoffs)                             │
│                                                                              │
│  Performance: ~5-10ms chunking overhead per chunk                           │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MICRO-BATCHING QUEUE                                    │
│                     (async request batching)                                 │
│                                                                              │
│  Configuration:                                                              │
│  • batch_timeout_ms: 15 (wait up to 15ms to gather requests)                │
│  • max_batch_size: 4 (process up to 4 requests together)                    │
│                                                                              │
│  How it works:                                                              │
│  1. Request arrives → added to queue                                        │
│  2. Wait up to 15ms for more requests                                       │
│  3. When timeout expires OR batch full → process batch                      │
│  4. GPU processes all requests in batch simultaneously                      │
│                                                                              │
│  Performance Impact:                                                        │
│  • Adds 0-15ms latency per request                                          │
│  • Improves throughput by ~9% (1.71 → 1.87 req/s)                           │
│  • Best at concurrency 10-20                                                │
│                                                                              │
│  Measured Results (24s audio):                                              │
│  ├─ Concurrency 5:  1.71 req/s, RTF 0.069, 0.69 min/s capacity             │
│  ├─ Concurrency 10: 1.87 req/s, RTF 0.115, 0.75 min/s capacity ✅          │
│  ├─ Concurrency 20: 1.87 req/s, RTF 0.217, 0.75 min/s capacity ✅          │
│  └─ Concurrency 40: 1.86 req/s, RTF 0.431, 0.74 min/s capacity             │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MODEL SELECTION                                        │
│                                                                              │
│  3-Tier Routing Logic:                                                      │
│  1. Language in TIER1_LANGUAGES → Tier 1: CTC-110M (fastest)               │
│  2. Language in TIER2_LANGUAGES → Tier 2: Parakeet 0.6B                    │
│  3. Other language → Tier 3: Whisper-Large-v3-Turbo                        │
│                                                                              │
│  Tier 1: Parakeet-TDT-CTC-110M (NVIDIA) - English (default):                        │
│  • Supported: English only                                                  │
│  • Performance: RTFx 5345 (60% faster than 0.6B)                            │
│  • WER: 7.49%                                                               │
│  • Memory: ~0.5GB (FP16)                                                    │
│                                                                              │
│  Tier 2: Parakeet-TDT-0.6B-v3 (NVIDIA):                                     │
│  • Supported: 24 European languages                                         │
│  • Languages: bg,hr,cs,da,nl,en,et,fi,fr,de,el,hu,it,lv,lt,mt,              │
│               pl,pt,ro,sk,sl,es,sv,ru,uk                                    │
│  • Performance: RTFx 3332                                                   │
│  • WER: 6.32%                                                               │
│  • Memory: ~2.5GB (BF16)                                                    │
│                                                                              │
│  Tier 3: Whisper-Large-v3-Turbo (OpenAI):                                   │
│  • Supported: 100+ languages                                                │
│  • Used for: Non-European languages (Chinese, Japanese, Arabic, etc.)       │
│  • Memory: ~6GB                                                             │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GPU INFERENCE                                        │
│                      (NVIDIA L4 / A10G / A100)                               │
│                                                                              │
│  Precision (configurable via TIER2_PRECISION):                              │
│  • bfloat16 (default): 94.46% accuracy, 125x real-time, balanced            │
│  • fp32:               94.34% accuracy, 100x real-time, max accuracy        │
│  • fp16 (ONNX):        89.74% accuracy, 167x real-time, max speed           │
│                                                                              │
│  Batch Processing:                                                          │
│  • Input: Batch of 1-4 audio chunks                                         │
│  • GPU memory: ~4-8GB per batch (depends on audio length)                   │
│  • Attention mode: Standard or local (for >1440s audio)                     │
│                                                                              │
│  Performance (24-second audio):                                             │
│  • Single request: ~1.5-1.8s processing time                                │
│  • Batched (2-4 requests): ~1.5-2.8s per request                            │
│  • RTF: 0.069-0.217 (depending on concurrency)                              │
│                                                                              │
│  Memory Management:                                                         │
│  • Automatic GPU memory cleanup after each batch                            │
│  • torch.cuda.empty_cache() + gc.collect()                                  │
│  • Prevents OOM on long-running servers                                     │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      POST-PROCESSING                                         │
│                                                                              │
│  For chunked audio:                                                         │
│  • Combine chunk transcriptions                                             │
│  • Remove overlap regions                                                   │
│  • Merge timestamps (if requested)                                          │
│                                                                              │
│  Format conversion:                                                         │
│  • text: Plain text output                                                  │
│  • json: Structured JSON with metadata                                      │
│  • vtt: WebVTT subtitles with timestamps                                    │
│  • srt: SRT subtitles with timestamps                                       │
│                                                                              │
│  Performance: ~10-20ms                                                      │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       METRICS CALCULATION                                    │
│                                                                              │
│  Computed metrics:                                                          │
│  • total_processing_time_sec: End-to-end time                               │
│  • audio_duration_sec: Length of audio file                                 │
│  • real_time_factor (RTF): processing_time / audio_duration                 │
│  • language_detection_time_ms: Time for language detection                  │
│  • transcription_time_sec: GPU inference time                               │
│                                                                              │
│  Cost analysis (g6.xlarge, us-east-1):                                      │
│  • On-Demand: $0.8048/hour                                                  │
│  • Spot: $0.35/hour (57% savings)                                           │
│  • 1-yr Savings: $0.4890/hour (39% savings)                                 │
│  • 3-yr Savings: $0.3215/hour (60% savings)                                 │
│                                                                              │
│  Cost per 1,000 minutes of audio (@ 1.74 min/s):                            │
│  • On-Demand: $0.1285                                                       │
│  • Spot: $0.0559                                                            │
│  • Weighted avg w/ overhead: $0.25 (Spot), $0.35 (1-yr Savings)             │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HTTP RESPONSE                                       │
│                                                                              │
│  {                                                                           │
│    "transcription": "...",                                                  │
│    "language": "en",                                                        │
│    "engine": "tier1",  // or "tier2", "tier3"                               │
│    "metrics": {                                                             │
│      "total_processing_time_sec": 1.65,                                     │
│      "audio_duration_sec": 24.0,                                            │
│      "real_time_factor": 0.069,                                             │
│      "language_detection_time_ms": 245.3,                                   │
│      "transcription_time_sec": 1.40,                                        │
│      "cost_per_1000_min_audio_usd": {                                       │
│        "on_demand": 0.13,                                                   │
│        "spot": 0.06,                                                        │
│        ...                                                                  │
│      }                                                                      │
│    }                                                                        │
│  }                                                                           │
│                                                                              │
│  Total end-to-end latency (24s audio):                                      │
│  • Low concurrency (5): ~1.5-1.8s                                           │
│  • Optimal concurrency (10-20): ~2.5-5.5s                                   │
│  • High concurrency (40): ~9.3-10.4s                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Performance Summary

### Real-Time Factor (RTF)
```
RTF = processing_time / audio_duration
RTFx = audio_duration / processing_time (inverse, higher is better)

Higher RTFx is better (RTFx > 1.0 = faster than real-time)

Per-Tier Performance (mixed workload benchmark):
┌─────────────────────────────────────────────────────────────┐
│  Tier    Model              Avg RTFx   Peak RTFx   Avg Time │
├─────────────────────────────────────────────────────────────┤
│  Tier 1  CTC-110M (en)         28x       167x      10.91s   │
│  Tier 2  Parakeet 0.6B         22x       111x       6.09s   │
│  Tier 3  Whisper-Large         26x        50x      16.14s   │
└─────────────────────────────────────────────────────────────┘

Notes:
• Peak RTFx achieved on long audio (30+ minutes)
• Short audio (<30s) shows lower RTFx due to fixed overhead
• Tier 1 excels on long English audio (167x on 59-min file)
```

### Throughput & Capacity
```
Measured Capacity (mixed workload, sequential processing):
├─ Audio capacity: 1.74 minutes/second
├─ Success rate: 100%
└─ Test: 35 files, 457.7 min audio, 263s processing

System can transcribe:
├─ Per second: 1.74 minutes of audio
├─ Per minute: 104 minutes of audio
├─ Per hour: 6,264 minutes of audio (104 hours)
└─ Per day: 150,336 minutes of audio (2,506 hours)

Per-Engine Breakdown:
├─ Tier 1 (English):    176.4 min processed, 6 files
├─ Tier 2 (European):   239.8 min processed, 24 files
└─ Tier 3 (Other):       40.2 min processed, 3 files
```

### Micro-Batching Impact
```
Benefit Analysis:
├─ Latency cost: 0-15ms per request (batch timeout)
├─ Throughput gain: +8.9% (1.71 → 1.87 req/s)
├─ GPU utilization: Better (process 2-4 requests per batch)
└─ Optimal concurrency: 10-20 concurrent requests

Interpretation: MODEST benefit
• Micro-batching provides measurable improvement
• Server may be bottlenecked on CPU preprocessing
• Requests benefit from batching but not dramatically
```

### Cost Efficiency
```
Cost per 1,000 Minutes by Engine (Spot pricing):
┌────────────────────────────────────────────────┐
│  Engine   Languages        Spot    1-Yr Savings │
├────────────────────────────────────────────────┤
│  Tier 1   English         $0.20        $0.29   │
│  Tier 2   European        $0.26        $0.37   │
│  Tier 3   Other           $0.22        $0.31   │
├────────────────────────────────────────────────┤
│  Weighted Average         $0.25        $0.35   │
└────────────────────────────────────────────────┘

Daily processing capacity (24/7 operation, Spot instance):
├─ Audio processed: 6,264 min/hour × 24 = 150,336 minutes/day
├─ Daily cost: $0.35/hour × 24 = $8.40
└─ Cost per minute: $8.40 / 150,336 = $0.0000559/min

Long audio is most cost-effective:
• 59-min English file: $0.03/1000 min (Tier 1, 167x RTFx)
• 46-min Spanish file: $0.05/1000 min (Tier 2, 111x RTFx)
• Short files (<30s): Higher per-minute cost due to fixed overhead
```

## Architecture Decisions

### Why Micro-Batching?
- **Problem**: GPU is underutilized with single requests
- **Solution**: Wait up to 15ms to gather 2-4 concurrent requests
- **Result**: 8.9% throughput improvement with minimal latency cost
- **Trade-off**: Adds 0-15ms latency but improves GPU efficiency

### Why Chunking?
- **Problem**: Long audio causes GPU OOM (out of memory)
- **Solution**: Split audio into 480-second (8-minute) chunks
- **Result**: Can process hours of audio on 24GB GPU
- **Trade-off**: Slight overhead for chunk processing

### Why Three Tiers?
- **Tier 1 (CTC-110M)**: Fastest for English (RTFx 5345, 110M params)
- **Tier 2 (Parakeet 0.6B)**: Best quality for European languages (RTFx 3332, 600M params)
- **Tier 3 (Whisper)**: Broadest language support (100+ languages)
- **Decision**: Route to optimal model based on detected language
- **Memory**: All three fit comfortably on 24GB GPU (~9.5GB total)

### Why Language Detection?
- **Problem**: User doesn't always know audio language
- **Solution**: Auto-detect using Whisper (GPU, primary) + VoxLingua107 (CPU, fallback)
- **Result**: 99%+ accuracy, ~200-300ms typical latency
- **Why Whisper Primary**: Testing showed VoxLingua has false positives (English→Welsh) and is 6x slower
- **Trade-off**: Adds upfront latency but enables correct model selection

## Related Files
- `blitz/server.py`: Main server implementation
- `blitz/config.py`: Configuration and thresholds
- `tests/test_microbatch_benefit.py`: Performance testing tool
- `docs/LOAD_TESTING.md`: Load testing documentation
