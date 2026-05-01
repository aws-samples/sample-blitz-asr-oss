# Testing Micro-Batching

## Quick Start

### 1. Build the Docker Image

```bash
cd /Users/bmaguir/git/GitHub/blitz

# Build with BuildKit for better caching
DOCKER_BUILDKIT=1 docker build -t parakeet-whisper-microbatch:latest .
```

### 2. Run the Container

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8003:8000 \
  --name parakeet-whisper-test \
  parakeet-whisper-microbatch:latest
```

**Expected startup output:**
```
================================================================================
Starting Parakeet + Whisper ASR Router with Micro-Batching
================================================================================

🔥 Warming up models...
✅ Parakeet model ready
✅ Whisper model ready

🚀 Starting batch workers (timeout=15.0ms, max_batch=4)
🔄 Parakeet batch worker started (batch_size=4, timeout=15.0ms)
🔄 Whisper batch worker started (batch_size=4, timeout=15.0ms)
✅ Server ready for requests

================================================================================
```

### 3. Verify Server is Running

```bash
# Health check
curl http://localhost:8003/health | jq '.'
```

**Expected response:**
```json
{
  "status": "healthy",
  "models": {
    "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
    "whisper": "openai/whisper-tiny"
  },
  "config": {
    "short_audio_threshold_sec": 600,
    "local_attention_threshold_sec": 1440,
    "parakeet_languages": 25,
    "whisper_languages": "all"
  }
}
```

## Test Scenarios

### Test 1: Single Request (Baseline - No Batching)

```bash
# Single request - establishes baseline performance
curl -X POST "http://localhost:8003/transcribe" \
  -F "file=@test_audio.wav" | jq '.'
```

**What to observe:**
- Normal latency (no batching delay)
- Response includes metrics with `transcribe_time_seconds`

### Test 2: Manual Concurrent Requests (Simple)

```bash
# Send 4 requests simultaneously in background
for i in {1..4}; do
  curl -X POST "http://localhost:8003/transcribe" \
    -F "file=@test_audio.wav" \
    > /tmp/response_$i.json &
done

# Wait for all to complete
wait

# Check results
for i in {1..4}; do
  echo "Request $i:"
  jq '.metrics.transcribe_time_seconds' /tmp/response_$i.json
done
```

**What to observe:**
- All 4 requests should complete in ~same total time as 1 request
- This proves they were batched together

### Test 3: Automated Benchmark (Recommended)

```bash
# Install dependencies
pip install aiohttp

# Run benchmark with default levels (1, 2, 4, 8 concurrent)
python test_microbatch.py --audio test_audio.wav
```

**Expected output:**
```
================================================================================
🚀 Micro-Batching Performance Benchmark
================================================================================
URL:        http://localhost:8003
Audio file: test_audio.wav
Levels:     [1, 2, 4, 8]
================================================================================

✅ Server is healthy
   Models: {'parakeet': 'nvidia/parakeet-tdt-0.6b-v3', 'whisper': 'openai/whisper-tiny'}

================================================================================
🧪 Testing with 1 concurrent requests
================================================================================

⏱️  Starting 1 concurrent requests...
✅ Request 0: 1.234s - parakeet - "this is the transcribed text..."

================================================================================
📊 Results Summary
================================================================================
Total time:        1.234s
Successful:        1/1
Failed:            0/1

Latency (avg):     1.234s
Latency (p50):     1.234s
Latency (p95):     1.234s
Latency (p99):     1.234s

Throughput:        0.81 requests/sec
================================================================================

⏸️  Pausing 2 seconds before next test...

================================================================================
🧪 Testing with 4 concurrent requests
================================================================================

⏱️  Starting 4 concurrent requests...
✅ Request 0: 1.250s - parakeet - "this is the transcribed text..."
✅ Request 1: 1.248s - parakeet - "this is the transcribed text..."
✅ Request 2: 1.252s - parakeet - "this is the transcribed text..."
✅ Request 3: 1.249s - parakeet - "this is the transcribed text..."

================================================================================
📊 Results Summary
================================================================================
Total time:        1.252s
Successful:        4/4
Failed:            0/4

Latency (avg):     1.250s
Latency (p50):     1.249s
Latency (p95):     1.252s
Latency (p99):     1.252s

Throughput:        3.19 requests/sec  ← 3.9x faster!
================================================================================

================================================================================
📈 Performance Comparison
================================================================================

Concurrent   Total Time   Throughput      Avg Latency
------------ ------------ --------------- ------------
1                  1.23s         0.81/s      1.234s
2                  1.24s         1.61/s      1.236s
4                  1.25s         3.19/s      1.250s
8                  2.51s         3.19/s      1.265s

================================================================================

🎯 Speedup Analysis (vs 1 concurrent):

  2 concurrent:
    - Throughput: 1.99x faster
    - Latency: +2.0ms

  4 concurrent:
    - Throughput: 3.93x faster
    - Latency: +16.0ms

  8 concurrent:
    - Throughput: 3.93x faster
    - Latency: +31.0ms
```

**What this shows:**
- **Baseline (1 request)**: 0.81 requests/sec
- **Batched (4 requests)**: 3.19 requests/sec = **3.9x faster!**
- **Latency increase**: Only +16ms for 4x throughput

### Test 4: Custom Concurrency Level

```bash
# Test specific concurrency
python test_microbatch.py --audio test_audio.wav --concurrent 8

# Test custom levels
python test_microbatch.py --audio test_audio.wav --levels 1 4 8 16
```

### Test 5: Stress Test (High Concurrency)

```bash
# Test with 16 concurrent requests (4 batches of 4)
python test_microbatch.py --audio test_audio.wav --concurrent 16
```

## Verifying Micro-Batching is Working

### Method 1: Check Throughput Scaling

**Without batching:**
- 1 request = 1.0s → 1.0 req/s
- 4 requests = 4.0s → 1.0 req/s (no improvement)

**With batching:**
- 1 request = 1.0s → 1.0 req/s
- 4 requests = 1.2s → 3.3 req/s (**3.3x improvement**)

### Method 2: Check Latency Addition

**With batching:**
- Single request latency: ~1.0s
- Concurrent request latency: ~1.0-1.015s (+0-15ms batching delay)

If latency increases by 0-15ms but throughput increases 2-4x, batching is working!

### Method 3: Monitor Server Logs

Add logging to batch workers to see batches forming:

```python
# In server.py batch worker (optional debugging)
print(f"📦 Processing batch of {len(batch)} requests")
```

Restart container and watch logs during concurrent test:
```
📦 Processing batch of 4 requests
```

## Troubleshooting

### Issue: No Throughput Improvement

**Problem:** 4 concurrent requests take 4x longer than 1 request

**Diagnosis:**
```bash
# Check if batch workers started
docker logs parakeet-whisper-test | grep "batch worker"

# Should see:
# 🔄 Parakeet batch worker started
# 🔄 Whisper batch worker started
```

**Solutions:**
1. Verify batch workers started successfully
2. Check requests arrive within 15ms window
3. Ensure using Python async test script (not sequential curl)

### Issue: All Requests Fail

**Problem:** Requests timeout or return errors

**Diagnosis:**
```bash
# Check server logs
docker logs parakeet-whisper-test --tail 50

# Check if server is running
curl http://localhost:8003/health
```

**Solutions:**
1. Check GPU is available: `docker run --gpus all`
2. Check memory: increase Docker memory limit
3. Check for Python errors in logs

### Issue: Latency Too High

**Problem:** Each request adds >50ms latency

**Diagnosis:**
```bash
# Check batch timeout setting
docker logs parakeet-whisper-test | grep "timeout="
# Should show: timeout=15.0ms
```

**Solution:**
```bash
# Reduce batch timeout
docker run --gpus all -p 8003:8000 \
  -e BATCH_TIMEOUT_MS=5.0 \
  parakeet-whisper-microbatch:latest
```

## Performance Expectations

### Expected Results on A10G GPU

| Concurrent | Total Time | Throughput | Latency | Speedup |
|-----------|-----------|-----------|---------|---------|
| 1 request | ~1.0s | 1.0 req/s | 1.0s | 1.0x |
| 2 requests | ~1.01s | 2.0 req/s | 1.005s | 2.0x |
| 4 requests | ~1.02s | 3.9 req/s | 1.010s | 3.9x |
| 8 requests | ~2.04s | 3.9 req/s | 1.020s | 3.9x |

**Key Observations:**
- Throughput scales nearly linearly up to batch size (4)
- Latency increases minimally (0-15ms)
- Beyond batch size (8+), multiple batches are needed

### GPU Utilization

**Monitor with:**
```bash
# In separate terminal
watch -n 1 nvidia-smi
```

**Expected:**
- Idle: 0-5% GPU utilization
- Single request: 60-70% GPU utilization
- Batched (4 concurrent): 85-95% GPU utilization ✅

## Creating Test Audio

If you don't have test audio:

```bash
# Create a 5-second test audio file
ffmpeg -f lavfi -i "sine=frequency=440:duration=5" -ar 16000 test_audio.wav

# Or record from microphone (5 seconds)
ffmpeg -f avfoundation -i ":0" -t 5 test_audio.wav  # macOS
ffmpeg -f alsa -i default -t 5 test_audio.wav       # Linux
```

## Next Steps

After confirming micro-batching works:

1. **Tune parameters** for your workload:
   - Increase `MAX_BATCH_SIZE` to 8 for higher throughput
   - Decrease `BATCH_TIMEOUT_MS` to 5ms for lower latency

2. **Implement additional optimizations**:
   - FP16 precision (40-50% memory reduction)
   - reset_fast_path() (15-25% speed improvement)
   - VAD chunking for very long audio

3. **Deploy to production** with monitoring:
   - Track throughput and latency metrics
   - Monitor GPU utilization
   - Set up alerts for errors

## Reference

- Implementation: `server.py` (batch workers, async queue)
- Configuration: `Dockerfile` (environment variables)
- Documentation: `MICRO_BATCHING.md`
- Optimizations: `PARAKEET_OPTIMIZATIONS.md`
