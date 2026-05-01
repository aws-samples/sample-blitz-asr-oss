# Load Testing Guide

This directory contains load testing tools for the Blitz transcription service.

## 📊 Test Scripts

### 1. `tests/test_load_loop.py` - Loop-Based Load Testing (Recommended)

**Purpose**: Extended stability testing with comprehensive reporting

**What It Tests**:
- Multi-hour stability and performance
- Per-language cost analysis
- Language detection confidence and model usage
- Low-cost transcription identification

**Usage**:
```bash
# Run 10 loops over all test files
python tests/test_load_loop.py --loops 10

# Run for 1 hour
python tests/test_load_loop.py --minutes 60

# Test specific directory
python tests/test_load_loop.py --minutes 30 --test-dir tests/fixtures/podcasts/
```

**Report Includes**:
- Per-engine statistics (count, RTF, time, audio)
- Cost per 1,000 minutes by engine and language
- Language detection statistics (avg confidence, model per language)
- Low-cost transcriptions (<= $0.06/1000 min)
- Latency statistics (P95, P99)
- Loop-over-loop stability metrics

### 2. `tests/test_load.py` - Simple Sequential Load Test

**Purpose**: Quick sequential load testing for basic validation

**Usage**:
```bash
# 10 sequential requests (default)
python tests/test_load.py

# Custom count
python tests/test_load.py -n 25
```

### 3. `tests/test_batch_load.py` - HTTP Request Pattern Comparison

**Purpose**: Compare different HTTP request patterns (concurrent vs sequential)

**What It Tests**:
- Client-side HTTP concurrency patterns
- How concurrent vs sequential requests affect overall latency
- Impact of HTTP connection overhead

**Important**: Both modes benefit from server-side micro-batching, so this test shows whether HTTP concurrency adds value when the server already has async batching enabled.

**Usage**:
```bash
# Basic test with default batch sizes (2, 4, 8)
python3 tests/test_batch_load.py --url http://localhost:8003/transcribe

# Custom batch sizes
python3 tests/test_batch_load.py --url http://localhost:8003/transcribe --batch-sizes 2,4,6,8,10,12

# Different output format
python3 tests/test_batch_load.py --url http://localhost:8003/transcribe --format json
```

**Expected Results**:
- When server has micro-batching enabled (default: `batch_timeout_ms=15ms`), concurrent HTTP requests may NOT show improvement
- Both modes benefit from server-side batching
- May see negative savings if HTTP concurrency adds overhead

### 4. `tests/test_microbatch_benefit.py` - Server Micro-Batching Effectiveness

**Purpose**: Measure actual server-side micro-batching benefits

**What It Tests**:
- Server's ability to batch concurrent requests together
- GPU utilization improvement from batching
- Optimal concurrency level for maximum throughput

**How It Works**:
1. Floods server with N concurrent requests simultaneously
2. Requests arrive within 15ms window → server batches them
3. GPU processes multiple requests in single batch → better efficiency
4. Compares throughput at different concurrency levels

**Usage**:
```bash
# Basic test with default concurrency levels (5, 10, 20, 40)
python3 tests/test_microbatch_benefit.py --url http://localhost:8003/transcribe

# Custom concurrency levels
python3 tests/test_microbatch_benefit.py --url http://localhost:8003/transcribe --concurrency 10,20,30,40,50

# Test with specific audio file
python3 tests/test_microbatch_benefit.py --url http://localhost:8003/transcribe --audio path/to/file.wav

# More iterations for better averaging
python3 tests/test_microbatch_benefit.py --url http://localhost:8003/transcribe --iterations 5
```

**Expected Results**:
- Higher concurrency → More requests batched → Better throughput
- Diminishing returns above `max_batch_size=4` (configurable)
- Shows optimal concurrency level for your workload
- Cost analysis per 1,000 requests

## 🎯 Which Test Should I Use?

### Use `tests/test_load_loop.py` when (Recommended):
- ✅ Running extended stability tests (hours)
- ✅ Analyzing per-language performance and cost
- ✅ Checking language detection confidence
- ✅ Identifying low-cost transcription candidates
- ✅ Comprehensive reporting with P95/P99 latencies

### Use `tests/test_load.py` when:
- ✅ Quick validation of server functionality
- ✅ Simple sequential testing
- ✅ Basic latency measurements

### Use `tests/test_batch_load.py` when:
- ✅ Testing different HTTP client patterns
- ✅ Comparing batch vs sequential request strategies
- ✅ Evaluating cost per 1,000 minutes of audio
- ✅ Stress testing with many audio files

### Use `tests/test_microbatch_benefit.py` when:
- ✅ Tuning server-side batching parameters
- ✅ Finding optimal concurrency for your workload
- ✅ Measuring GPU batching efficiency
- ✅ Cost analysis per 1,000 requests (not minutes)
- ✅ Testing with single audio file at high concurrency

## ⚙️ Server Configuration

Both tests work with the server's micro-batching configuration:

```bash
# Default settings (recommended for g6.xlarge)
export BATCH_TIMEOUT_MS=15    # Wait up to 15ms to gather batch
export MAX_BATCH_SIZE=4       # Process up to 4 requests together

# To test WITHOUT micro-batching (for comparison)
export BATCH_TIMEOUT_MS=0     # Disable batching
export MAX_BATCH_SIZE=1       # Single request processing
```

**Recommended Testing Workflow**:
1. Run `tests/test_microbatch_benefit.py` to find optimal concurrency
2. Use that concurrency level in your production client
3. Run `tests/test_batch_load.py` periodically to verify cost efficiency

## 📁 Test Audio Files

Test scripts use audio files from `tests/fixtures/` directory:

```
tests/fixtures/
├── beeping_only_test_audio.wav           # English test files
├── beeping_only_test_audio_short.wav
├── english-test-hour-audio_processed_asr_ready.wav
├── german.mp3                        # European language samples
├── french.mp3
├── spanish.mp3
├── italian.mp3
├── dutch.mp3
├── portuguese.mp3
├── polish.mp3
├── czech.mp3
├── greek.mp3
├── russian.mp3
├── ukrainian.mp3
├── romainian.mp3
├── finish.mp3
├── chinese_OSR_cn_000_0072_8k.wav   # Non-European languages
├── chinese_OSR_cn_000_0075_8k.wav
├── podcasts/                         # Long-form podcast episodes (10-45 min)
│   ├── spanish-283-benvinguts.mp3
│   ├── spanish-286-pasando.mp3
│   └── italian-easy-italian-53-milan.mp3
└── mono-wav-tests/                   # Pre-processed mono WAV (16kHz)
    ├── beeping_only_test_audio_mono.wav
    ├── german_mono.wav
    └── ...                           # All fixtures converted to mono WAV
```

### Mono WAV Test Files

The `mono-wav-tests/` folder contains pre-processed versions of all audio files:
- Converted to mono (single channel)
- Resampled to 16kHz (ASR-optimized)
- WAV format (no decoding overhead)

Use these to measure the cost impact of client-side audio preprocessing.

### Adding Test Files

```bash
# Add your test audio files
cp /path/to/samples/*.wav tests/fixtures/

# Or download sample
wget https://example.com/sample.wav -O test_audio/sample.wav
```

**Supported formats**: WAV, MP3, M4A, FLAC

## 📈 Understanding Results

### Real-Time Factor (RTF)
- `RTF = processing_time / audio_duration`
- **Lower is better** (RTF < 1.0 means faster than real-time)
- RTF of 0.021 means 21ms to process 1 second of audio

### Throughput
- Requests processed per second
- **Higher is better**
- Shows how many concurrent requests server can handle

### Cost per 1,000 Minutes
- Standard transcription pricing metric (`tests/test_batch_load.py`)
- Calculates cost based on actual GPU time
- Compares across AWS pricing models (On-Demand, Spot, Savings Plans)

### Cost per 1,000 Requests
- Alternative pricing metric (`tests/test_microbatch_benefit.py`)
- Useful for request-based billing
- Shows cost efficiency of batching

## 🔧 Troubleshooting

### Connection Refused
```bash
# Check server is running
curl http://localhost:8003/health

# If on different port, use --url
python3 tests/test_batch_load.py --url http://localhost:YOUR_PORT/transcribe
```

### Port Mapping Issues
```bash
# Server advertises 0.0.0.0:8000 but may be accessible on different port
# Try common ports: 8000, 8003, 8080

# Check Docker port mapping
docker ps  # Look at PORTS column
```

### All Requests Failing
```bash
# 1. Verify server connectivity
curl -X POST http://localhost:8003/transcribe \
  -F "audio_file=@test_audio/sample.wav" \
  -F "format=text"

# 2. Check audio files exist
ls -lh test_audio/

# 3. Check server logs for errors
docker logs YOUR_CONTAINER_NAME
```

### Out of Memory
```bash
# Reduce batch sizes
python3 tests/test_batch_load.py --batch-sizes 2,4,6

# Or reduce concurrency
python3 tests/test_microbatch_benefit.py --concurrency 5,10,15

# Check server chunking configuration
export CHUNK_LENGTH_SEC=480  # 8 minutes (default)
```

## 📚 Example Output

### test_batch_load.py
```
================================================================================
📊 BATCH SIZE 4 RESULTS
================================================================================

📊 REAL-TIME FACTOR (RTF):
   Batch:       0.021
   Sequential:  0.012
   Batch WORSE by:    0.009 (75.0% higher RTF) ⚠️

💰 COST PER 1,000 MINUTES (On-Demand):
   Batch:       $0.28
   Sequential:  $0.16
   Batch costs MORE: $0.12 (75.0% more expensive) ⚠️
```

### test_microbatch_benefit.py
```
================================================================================
📊 SUMMARY: Micro-Batching Efficiency by Concurrency Level
================================================================================

Concurrency   Throughput    Avg RTF  Total Time  Success Rate
------------ ------------ ---------- ------------ -------------
5               12.45/s      0.018      8.02s        100.0%
10              23.12/s      0.019      8.65s        100.0% ✅
20              38.76/s      0.021     10.31s        100.0% ✅
40              42.18/s      0.022     18.99s        100.0% ✅

💡 OPTIMAL CONCURRENCY: 40 concurrent requests
   Best throughput:  42.18 req/s
   Best RTF:         0.022
   Improvement:      238.8% vs concurrency=5
```

## 🚀 Production Recommendations

Based on test results:

1. **Set optimal concurrency**: Use `tests/test_microbatch_benefit.py` to find sweet spot
2. **Monitor RTF**: Keep below 0.1 for good user experience
3. **Use Spot instances**: 35% cost savings vs On-Demand
4. **Enable micro-batching**: Default settings work well for most workloads
5. **Chunk long audio**: Keep `CHUNK_LENGTH_SEC=480` to prevent OOM

## 📝 Related Files

- `blitz/config.py` - Server configuration and tuning parameters
- `blitz/server.py` - Main server implementation
- `requirements.txt` - Python dependencies
- `README.md` - Main project documentation
