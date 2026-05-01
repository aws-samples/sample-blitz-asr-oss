# Blitz Async Transcription Flow

This document describes the asynchronous architecture of the Blitz transcription server, including language detection, model routing, and micro-batching.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           BLITZ ASYNC TRANSCRIPTION FLOW                                │
└─────────────────────────────────────────────────────────────────────────────────────────┘

  CLIENT REQUEST
       │
       ▼
┌──────────────┐     ┌─────────────────────────────────────────────────────────────────┐
│ POST         │     │                    FASTAPI SERVER                               │
│ /transcribe  │────▶│  ┌────────────────────────────────────────────────────────────┐ │
│              │     │  │ 1. AUDIO LOADING                                           │ │
│ - file       │     │  │    ┌─────────────┐    ┌─────────────┐                      │ │
│ - format     │     │  │    │ soundfile   │───▶│ FFmpeg      │ (fallback)           │ │
│ - lang_hint  │     │  │    │ WAV/FLAC/OGG│    │ MP3/M4A/etc │                      │ │
└──────────────┘     │  │    └─────────────┘    └─────────────┘                      │ │
                     │  │              │                                              │ │
                     │  │              ▼                                              │ │
                     │  │    ┌─────────────────┐                                      │ │
                     │  │    │ Stereo → Mono   │                                      │ │
                     │  │    │ → float32       │                                      │ │
                     │  │    └────────┬────────┘                                      │ │
                     │  └─────────────│────────────────────────────────────────────────┘ │
                     │                │                                                  │
                     │                ▼                                                  │
                     │  ┌────────────────────────────────────────────────────────────┐   │
                     │  │ 2. LANGUAGE DETECTION                                      │   │
                     │  │                                                            │   │
                     │  │   lang_hint provided?                                      │   │
                     │  │         │                                                  │   │
                     │  │    YES  │  NO                                              │   │
                     │  │    ┌────┴────┐                                             │   │
                     │  │    │         ▼                                             │   │
                     │  │    │  ┌─────────────────┐                                  │   │
                     │  │    │  │ Whisper Encoder │ (PRIMARY)                        │   │
                     │  │    │  │ GPU ~200-300ms  │                                  │   │
                     │  │    │  └────────┬────────┘                                  │   │
                     │  │    │           │                                           │   │
                     │  │    │           ▼                                           │   │
                     │  │    │    confidence >= 0.70?                                │   │
                     │  │    │      │           │                                    │   │
                     │  │    │     YES          NO                                   │   │
                     │  │    │      │           ▼                                    │   │
                     │  │    │      │    ┌─────────────────┐                         │   │
                     │  │    │      │    │ VoxLingua107    │ (FALLBACK)              │   │
                     │  │    │      │    │ CPU Secondary   │                         │   │
                     │  │    │      │    └────────┬────────┘                         │   │
                     │  │    │      │             │                                  │   │
                     │  │    ▼      ▼             ▼                                  │   │
                     │  │  ┌─────────────────────────┐                               │   │
                     │  │  │ DETECTED LANGUAGE CODE  │                               │   │
                     │  │  │ + confidence score      │                               │   │
                     │  │  └───────────┬─────────────┘                               │   │
                     │  └──────────────│──────────────────────────────────────────────┘  │
                     │                 │                                                  │
                     │                 ▼                                                  │
                     │  ┌────────────────────────────────────────────────────────────┐   │
                     │  │ 3. VOICE ACTIVITY DETECTION (CONDITIONAL)                  │   │
                     │  │                                                            │   │
                     │  │    confidence < 0.70?                                      │   │
                     │  │         │          │                                       │   │
                     │  │        YES         NO                                      │   │
                     │  │         │          │                                       │   │
                     │  │         ▼          │                                       │   │
                     │  │  ┌─────────────┐   │                                       │   │
                     │  │  │ Silero VAD  │   │                                       │   │
                     │  │  │ (CPU)       │   │                                       │   │
                     │  │  └──────┬──────┘   │                                       │   │
                     │  │         │          │                                       │   │
                     │  │    is_silent?      │                                       │   │
                     │  │     │       │      │                                       │   │
                     │  │    YES      NO     │                                       │   │
                     │  │     │       │      │                                       │   │
                     │  │     ▼       └──────┼───────────────────┐                   │   │
                     │  │ ┌────────┐         │                   │                   │   │
                     │  │ │ RETURN │         │    SKIP VAD       │                   │   │
                     │  │ │ empty  │         │    (high conf)    │                   │   │
                     │  │ │ text   │         │                   │                   │   │
                     │  │ └────────┘         │                   │                   │   │
                     │  └───────────────────│───────────────────│────────────────────┘   │
                     │                      │                   │                        │
                     │                      ▼                   ▼                        │
                     │  ┌────────────────────────────────────────────────────────────┐   │
                     │  │ 4. MODEL ROUTING                                           │   │
                     │  │                                                            │   │
                     │  │         ┌──────────────────────────────────┐               │   │
                     │  │         │     LANGUAGE ROUTER              │               │   │
                     │  │         │                                  │               │   │
                     │  │         │  lang in PARAKEET_LANGS (25)?    │               │   │
                     │  │         │        │              │          │               │   │
                     │  │         │       YES             NO         │               │   │
                     │  │         │        │              │          │               │   │
                     │  │         │        ▼              ▼          │               │   │
                     │  │         │  engine="parakeet"  engine="whisper"             │   │
                     │  │         │                                  │               │   │
                     │  │         │  PARAKEET: bg,hr,cs,da,nl,en,   │               │   │
                     │  │         │  et,fi,fr,de,el,hu,it,lv,lt,    │               │   │
                     │  │         │  mt,pl,pt,ro,sk,sl,es,sv,ru,uk  │               │   │
                     │  │         │                                  │               │   │
                     │  │         │  WHISPER: 100+ languages         │               │   │
                     │  │         └──────────────────────────────────┘               │   │
                     │  └────────────────────────────────────────────────────────────┘   │
                     │                                                                   │
                     └───────────────────────────────────────────────────────────────────┘
                                              │
                                              │ job = {job_id, engine, audio, sr, ...}
                                              │ result_future = asyncio.Future()
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                          ASYNC MICRO-BATCHING QUEUE                                     │
│                                                                                         │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                         asyncio.Queue (shared)                                    │  │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                                 │  │
│  │  │ J1  │ │ J2  │ │ J3  │ │ J4  │ │ J5  │ │ J6  │  ...                            │  │
│  │  │ P   │ │ W   │ │ P   │ │ P   │ │ W   │ │ P   │                                 │  │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                                 │  │
│  │    │       │       │       │       │       │                                      │  │
│  └────┼───────┼───────┼───────┼───────┼───────┼──────────────────────────────────────┘  │
│       │       │       │       │       │       │                                         │
│       │       │       │       │       │       │   P = Parakeet job                      │
│       │       │       │       │       │       │   W = Whisper job                       │
│       ▼       ▼       ▼       ▼       ▼       ▼                                         │
│  ┌────────────────────────────────┐  ┌────────────────────────────────┐                 │
│  │     PARAKEET BATCH WORKER      │  │      WHISPER BATCH WORKER      │                 │
│  │                                │  │                                │                 │
│  │  ┌──────────────────────────┐  │  │  ┌──────────────────────────┐  │                 │
│  │  │ 1. Wait for first job    │  │  │  │ 1. Wait for first job    │  │                 │
│  │  │    where engine="parakeet│  │  │  │    where engine="whisper"│  │                 │
│  │  └────────────┬─────────────┘  │  │  └────────────┬─────────────┘  │                 │
│  │               │                │  │               │                │                 │
│  │               ▼                │  │               ▼                │                 │
│  │  ┌──────────────────────────┐  │  │  ┌──────────────────────────┐  │                 │
│  │  │ 2. Gather batch within   │  │  │  │ 2. Gather batch within   │  │                 │
│  │  │    15ms timeout window   │  │  │  │    15ms timeout window   │  │                 │
│  │  │    (max 4 jobs)          │  │  │  │    (max 4 jobs)          │  │                 │
│  │  └────────────┬─────────────┘  │  │  └────────────┬─────────────┘  │                 │
│  │               │                │  │               │                │                 │
│  │               ▼                │  │               ▼                │                 │
│  │  ┌──────────────────────────┐  │  │  ┌──────────────────────────┐  │                 │
│  │  │ 3. process_batch_func()  │  │  │  │ 3. process_batch_func()  │  │                 │
│  │  │    NeMo 0.6B or ONNX     │  │  │  │    large-v3-turbo        │  │                 │
│  │  └────────────┬─────────────┘  │  │  └────────────┬─────────────┘  │                 │
│  │               │                │  │               │                │                 │
│  │               ▼                │  │               ▼                │                 │
│  │  ┌──────────────────────────┐  │  │  ┌──────────────────────────┐  │                 │
│  │  │ 4. Set result on Future  │  │  │  │ 4. Set result on Future  │  │                 │
│  │  │    future.set_result()   │  │  │  │    future.set_result()   │  │                 │
│  │  └──────────────────────────┘  │  │  └──────────────────────────┘  │                 │
│  │                                │  │                                │                 │
│  └────────────────────────────────┘  └────────────────────────────────┘                 │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │ await result_future (timeout=300s)
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              RESPONSE FORMATTING                                        │
│                                                                                         │
│    result = {text, timestamps, chunk_approach, chunk_count}                             │
│                              │                                                          │
│                              ▼                                                          │
│    ┌─────────────────────────────────────────────────────────────────┐                  │
│    │                    FORMAT ROUTER                                │                  │
│    │                                                                 │                  │
│    │      format="text"/"json"    format="srt"    format="vtt"       │                  │
│    │              │                    │               │             │                  │
│    │              ▼                    ▼               ▼             │                  │
│    │        ┌─────────┐          ┌─────────┐    ┌─────────┐          │                  │
│    │        │  JSON   │          │   SRT   │    │   VTT   │          │                  │
│    │        │ Response│          │ Subtitle│    │ Subtitle│          │                  │
│    │        └─────────┘          └─────────┘    └─────────┘          │                  │
│    └─────────────────────────────────────────────────────────────────┘                  │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                                         CLIENT RESPONSE


┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              CONCURRENT REQUEST FLOW                                    │
│                                                                                         │
│   Request 1 ──┐                                                                         │
│   Request 2 ──┼──▶ Queue ──▶ ┌─────────────┐     ┌─────────────┐                        │
│   Request 3 ──┤              │  Batch 1    │────▶│ GPU Batch   │──┐                     │
│   Request 4 ──┘              │  (gathered  │     │ Inference   │  │                     │
│                              │  in 15ms)   │     └─────────────┘  │                     │
│                              └─────────────┘                      │                     │
│                                                                   ▼                     │
│   Request 5 ──┐                              ┌──────────────────────────────┐           │
│   Request 6 ──┼──▶ Queue ──▶ ┌─────────────┐ │  Results returned via        │           │
│   Request 7 ──┘              │  Batch 2    │ │  asyncio.Future per request  │           │
│                              │  (next      │ └──────────────────────────────┘           │
│                              │   window)   │                                            │
│                              └─────────────┘                                            │
│                                                                                         │
│   Throughput: ~1.87 req/s at optimal concurrency (10-20 concurrent)                     │
│   Latency overhead: 0-15ms per request (batch gathering window)                         │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Key Async Components

| Component | Purpose |
|-----------|---------|
| `asyncio.Queue` | Shared queue for all transcription jobs |
| `asyncio.Future` | Per-request result placeholder |
| `BatchWorker` | Engine-specific worker (filters by `engine` field) |
| `batch_timeout_ms=15` | Window to gather batch (adds 0-15ms latency) |
| `max_batch_size=4` | Max jobs processed together |

## Flow Summary

1. **Request arrives** - Audio loaded, language detected
2. **Routing decision** - Parakeet (25 langs) or Whisper (100+ langs)
3. **Job submitted** - Added to shared `asyncio.Queue` with `Future`
4. **Batch worker** - Filters jobs by engine, gathers batch in 15ms window
5. **GPU inference** - Batch processed together for efficiency
6. **Results distributed** - Each `Future.set_result()` unblocks waiting request
7. **Response formatted** - JSON/SRT/VTT returned to client

## Source Files

| File | Description |
|------|-------------|
| `blitz/server.py` | FastAPI endpoint, language detection, routing |
| `blitz/services/batch.py` | Generic `BatchWorker` class |
| `blitz/app_state.py` | Application state, queue initialization |
| `blitz/services/language_detection.py` | Whisper + VoxLingua107 detection |
| `blitz/services/vad.py` | Silero VAD for silence detection |
