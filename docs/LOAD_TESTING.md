# Load Testing Guide

## Quick Start

```bash
# Install dependencies (if needed)
pip install aiohttp

# Put test audio files in tests/fixtures/ directory
mkdir -p tests/fixtures/
cp your_audio_files.mp3 tests/fixtures/

# Basic load test (10 sequential requests)
# Uses all files from tests/fixtures/ directory
python tests/test_load.py

# Loop-based load test (recommended for extended testing)
python tests/test_load_loop.py --loops 10

# Time-based load test (1 hour)
python tests/test_load_loop.py --minutes 60

# Custom server URL
python tests/test_load.py --url http://localhost:8003/transcribe
```

## Test Fixtures

Test audio files are organized in `tests/fixtures/`:

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
├── podcasts/                         # Long-form podcast episodes
│   ├── spanish-283-benvinguts.mp3
│   ├── spanish-284-metemo.mp3
│   ├── ...
│   └── italian-easy-italian-53-milan.mp3
└── mono-wav-tests/                   # Pre-processed mono WAV (16kHz)
    ├── beeping_only_test_audio_mono.wav
    ├── german_mono.wav
    ├── spanish-286-pasando_mono.wav
    └── ...
```

### Mono WAV Test Files

The `mono-wav-tests/` folder contains pre-processed versions of all audio files:
- Converted to mono (single channel)
- Resampled to 16kHz (ASR-optimized)
- WAV format (no decoding overhead)

Use these to measure the cost impact of client-side audio preprocessing vs. sending raw MP3s.

## Usage

```bash
python tests/test_load.py [options]
```

The script automatically finds all audio files in the `tests/fixtures/` directory and rotates through them.

**Important**: Requests are sent **sequentially** (one at a time) to avoid memory issues. Each request waits for the previous one to complete before sending the next.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--test-dir` | `tests/fixtures` | Directory containing audio files |
| `--url` | `http://localhost:8003/transcribe` | Server endpoint |
| `-n, --num-requests` | `10` | Total number of requests to send |
| `-f, --format` | `json` | Response format (text, json, srt, vtt) |

### Supported Audio Formats

The script automatically finds these audio file types:
- `.wav` - WAV audio
- `.mp3` - MP3 audio
- `.flac` - FLAC audio
- `.m4a` - M4A/AAC audio
- `.ogg` - Ogg Vorbis
- `.opus` - Opus audio

## Test Scenarios

### 1. Quick Test
```bash
# 5 sequential requests
python tests/test_load.py -n 5
```

### 2. Standard Test (Default)
```bash
# 10 sequential requests
python tests/test_load.py
```

### 3. Extended Test
```bash
# 25 sequential requests
python tests/test_load.py -n 25
```

### 4. Thorough Test
```bash
# 50 sequential requests
python tests/test_load.py -n 50
```

### 5. Format Testing
```bash
# Test text format
python tests/test_load.py --format text

# Test JSON format
python tests/test_load.py --format json

# Test SRT format (should fail - timestamps disabled)
python tests/test_load.py --format srt
```

### 6. Different Test Directory
```bash
# Use a different directory with audio files
python tests/test_load.py --test-dir audio_samples/
```

## Interpreting Results

### Success Criteria

✅ **Passed**: ≥95% success rate
⚠️  **Warning**: 80-94% success rate
❌ **Failed**: <80% success rate

### Key Metrics

1. **Success Rate**: Percentage of requests that succeeded
2. **Throughput**: Requests processed per second
3. **Latency**: Time to process each request
   - Min/Max: Range of response times
   - Mean: Average response time
   - Median: Middle value (better for skewed distributions)
   - Std Dev: Consistency of response times

### What to Look For

**Good Signs:**
- ✅ 100% success rate
- ✅ Consistent latencies (low std dev)
- ✅ Stable memory usage
- ✅ No CUDA OOM errors
- ✅ No timeout errors

**Warning Signs:**
- ⚠️  Success rate 80-95%
- ⚠️  High latency variance
- ⚠️  Occasional timeout errors
- ⚠️  Gradual memory increase

**Critical Issues:**
- ❌ Success rate <80%
- ❌ CUDA Out of Memory errors
- ❌ Server crashes
- ❌ Consistent timeout errors
- ❌ 500 Internal Server Errors

## Example Output

```
======================================================================
🔥 SEQUENTIAL LOAD TEST STARTING
======================================================================
URL:          http://localhost:8003/transcribe
Test dir:     tests/fixtures/
Audio files:  3 files found
   [1] ap-german-sect-1-part-b-sel-1-effective-2019.mp3
   [2] test_audio_short.wav
   [3] test_audio_long.mp3
Mode:         Sequential (one at a time)
Total:        10 requests
Format:       json
======================================================================

📤 Request 1/10
   ✅ Success
   📄 File: ap-german-sect-1-part-b-sel-1-effective-2019.mp3
   ⏱️  Time: 2.31s
   📝 Text length: 159 chars

📤 Request 2/10
   ✅ Success
   📄 File: test_audio_short.wav
   ⏱️  Time: 1.45s
   📝 Text length: 87 chars

📤 Request 3/10
   ✅ Success
   📄 File: test_audio_long.mp3
   ⏱️  Time: 3.12s
   📝 Text length: 234 chars

...

======================================================================
📊 LOAD TEST RESULTS
======================================================================

📈 Summary:
   Total requests:     10
   ✅ Successful:      10 (100.0%)
   ❌ Failed:          0 (0.0%)
   ⏱️  Total time:       21.45s
   🚀 Overall throughput: 0.47 req/s

⏱️  Latency Statistics (successful requests):
   Min:     1.45s
   Max:     3.12s
   Mean:    2.15s
   Median:  2.08s
   Std Dev: 0.52s

📝 Transcription Statistics:
   Min length:     87 chars
   Max length:     234 chars
   Mean length:    156 chars

📁 Files Tested:
   ap-german-sect-1-part-b-sel-1-effective-2019.mp3: 4 requests
   test_audio_long.mp3: 3 requests
   test_audio_short.wav: 3 requests

📄 Sample Response (Request 0):
   Language: de
   Engine: parakeet
   Duration: 5.2s
   Text: Kaspersky hat eine Malware entdeckt...
   Metrics:
      Processing time: 2.31s
      GPU cost: $0.000019

======================================================================
✅ TEST PASSED - Server is stable!
======================================================================
```

## Monitoring During Tests

### GPU Memory
```bash
# In another terminal, monitor GPU usage
watch -n 1 nvidia-smi
```

Look for:
- Memory usage should stay under 22GB
- No "out of memory" messages
- GPU utilization should be high during processing

### Server Logs
```bash
# Watch server logs
docker logs -f blitz-container
```

Look for:
- Batch worker processing messages
- No error stack traces
- Successful transcriptions

### Docker Stats
```bash
# Monitor container resources
docker stats blitz-container
```

Look for:
- CPU usage patterns
- Memory usage
- Network I/O

## Troubleshooting

### High Failure Rate

**Possible causes:**
1. Server not running or wrong URL
2. CUDA OOM errors
3. Network timeout (increase timeout in script)
4. Audio file issues

**Solutions:**
```bash
# Check server is running
curl http://localhost:8003/transcribe

# Check server logs
docker logs blitz-container | tail -50

# Try with fewer requests
python tests/test_load.py -n 3
```

### Slow Performance

**Possible causes:**
1. GPU not being utilized
2. Large audio files
3. File I/O bottleneck

**Solutions:**
```bash
# Check GPU usage
nvidia-smi

# Use smaller audio files for testing
# Check server logs for processing times
docker logs blitz-container
```

### Memory Issues

**Possible causes:**
1. Audio files too long
2. Memory leak
3. Models not unloading properly

**Solutions:**
```bash
# Use shorter audio clips
# Check for memory leaks in server logs
# Monitor GPU memory during test
watch -n 1 nvidia-smi
```

## Advanced Usage

### Continuous Testing

```bash
# Run tests in a loop for stability testing
for i in {1..10}; do
    echo "Round $i"
    python tests/test_load.py -n 10
    sleep 5
done
```

### Save Results

```bash
# Save results to file
python tests/test_load.py -n 10 | tee load_test_results.txt
```

### Different Test Directories

```bash
# Test with different audio collections
python tests/test_load.py --test-dir short_clips/ -n 20
python tests/test_load.py --test-dir long_form/ -n 10
```

## Loop-Based Load Testing

The `test_load_loop.py` script provides comprehensive load testing with detailed reporting.

### Usage

```bash
# Run 10 loops (default)
python tests/test_load_loop.py --loops 10

# Run for 60 minutes
python tests/test_load_loop.py --minutes 60

# Run for 4 hours
python tests/test_load_loop.py --minutes 240

# Use specific test directory
python tests/test_load_loop.py --minutes 30 --test-dir tests/fixtures/podcasts/

# Custom server URL
python tests/test_load_loop.py --loops 10 --url http://localhost:8000/transcribe
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--loops` | `10` | Number of loops to run |
| `--minutes` | - | Run for specified minutes (takes precedence over --loops) |
| `--test-dir` | `test` | Directory containing audio files |
| `--url` | `http://localhost:8003/transcribe` | Server endpoint |
| `-f, --format` | `json` | Response format (text, json, srt, vtt) |

### Report Sections

The final report includes:

1. **Per-Engine Statistics**: Count, avg RTF, avg time, total audio per engine
2. **Cost per 1,000 Minutes by Engine**: Spot and 1-year savings pricing
3. **Cost per 1,000 Minutes by Language**: Breakdown by detected language
4. **Language Detection Statistics**: Average confidence and detection model per language
5. **Low-Cost Transcriptions**: Files at or below $0.06/1000 min with duration, time, and engine
6. **Latency Statistics**: Min, max, mean, median, std dev, P95, P99
7. **Loop-over-Loop Stability**: Throughput and capacity variance across loops

### Example Output

```
Language Detection Statistics:
  Language   Avg Confidence Detection Model
  -------------------------------------------------------
  es                  92.3% voxlingua107
  en                  97.1% voxlingua107
  de                  94.5% voxlingua107
  -------------------------------------------------------

Low-Cost Transcriptions (<= $0.06/1000 min):
  Filename                                   Duration      Time Engine            Cost
  ------------------------------------------------------------------------------------------
  spanish-286-pasando.mp3                     45.7 min   27.93s parakeet       $  0.0312
  kelsey-hightower-kubernetes.mp3             43.2 min   25.11s parakeet_ctc   $  0.0298
  italian-easy-italian-53-milan.mp3           12.1 min    8.45s parakeet       $  0.0356
  ------------------------------------------------------------------------------------------
  Total: 3 files qualify for low-cost transcription
```

## Performance Targets

Based on A10G 24GB GPU with FP32 (sequential mode):

| Metric | Target | Notes |
|--------|--------|-------|
| Success rate | ≥95% | Should be 100% for stability |
| Latency (5s audio) | 2-3s | ~0.4-0.6× realtime |
| Latency (30s audio) | 10-15s | ~0.3-0.5× realtime |
| GPU memory | <22GB | Should stay stable across requests |
| Memory stability | No growth | Watch for memory leaks |

## Next Steps

After successful load testing:

1. **Monitor production**: Set up monitoring and alerting
2. **Memory analysis**: Verify no memory leaks over extended runs
3. **Scale horizontally**: Add more instances if needed for higher throughput
4. **Optimize if needed**: Consider BFloat16 or INT8 quantization for better performance
