# Documentation

This directory contains detailed technical documentation for the Blitz ASR server.

## Quick Reference

- **[LOAD_TESTING.md](LOAD_TESTING.md)** - Complete guide for load testing the server
- **[PARAKEET_OPTIMIZATIONS.md](PARAKEET_OPTIMIZATIONS.md)** - Performance optimization strategies
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Summary of implemented optimizations

## Performance & Optimization

### BFloat16 Precision
- **[BFLOAT16_PRECISION.md](BFLOAT16_PRECISION.md)** - Detailed explanation of BFloat16 implementation
  - 50% memory reduction (19GB → 9.5GB)
  - 10-30% speed improvement
  - Quality analysis and benchmarks

### Micro-Batching
- **[MICRO_BATCHING.md](MICRO_BATCHING.md)** - Micro-batching implementation guide
  - 30-50% throughput improvement
  - Request queuing and batch processing
  - Configuration and tuning

Note: Micro-batching is currently **disabled** due to memory constraints with FP32 models. The server runs in sequential mode (one request at a time).

## Testing

### Load Testing
- **[LOAD_TESTING.md](LOAD_TESTING.md)** - Complete load testing guide
  - Sequential testing (one request at a time)
  - Test scenarios and examples
  - Result interpretation
  - Troubleshooting guide

### Micro-Batch Testing (Legacy)
- **[TESTING_MICROBATCH.md](TESTING_MICROBATCH.md)** - Micro-batch testing guide (legacy)
- **tests/test_microbatch.py** - Micro-batch test script

Note: Micro-batching is enabled with BFloat16 precision for optimal performance.

## Development History

### Implementation Notes
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Complete implementation summary
  - What was implemented
  - Memory usage evolution
  - Testing plan and success criteria

### Docker Configuration
- **[DOCKERFILE_CHANGES.md](DOCKERFILE_CHANGES.md)** - Dockerfile modification history
  - Environment variables
  - Model pre-downloading
  - Configuration changes

### Improvements Tracking
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** - Tracked improvements and optimizations
  - Performance benchmarks
  - Memory optimization history
  - Quality analysis

## Current Configuration

**Production Setup:**
- **Precision:** FP32 (for stability)
- **Mode:** Sequential (one request at a time)
- **Memory:** ~19GB GPU usage
- **Timestamps:** Disabled (for stability)
- **Supported Formats:** text, json (SRT/VTT disabled)

**Why Sequential Mode:**
- FP32 models use 19GB → only ~3GB free on A10G 24GB
- Not enough memory for concurrent batching
- Sequential ensures stability and prevents OOM errors

**Future Optimizations:**
If memory becomes available (via BFloat16, INT8, or larger GPU):
- Enable micro-batching for 30-50% throughput improvement
- Enable timestamps for SRT/VTT support
- Consider Flash Attention for long-form audio

## Related Files

### Root Directory
- **README.md** - Main project README
- **Dockerfile** - Production Docker image
- **blitz/server.py** - FastAPI server implementation

### Docker Directory
- **docker/Dockerfile.batch** - AWS Batch processing image
- **docker/Dockerfile.onnx** - ONNX FP16 backend image

### Test Directory
- **tests/fixtures/** - Audio files for testing
- **tests/test_load.py** - Load testing script
- **tests/test_integration.py** - Integration tests
