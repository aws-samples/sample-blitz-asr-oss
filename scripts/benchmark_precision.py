#!/usr/bin/env python3
"""
Benchmark Parakeet model with different precision modes: fp32, fp16, bf16.

Tests inference latency and memory usage across precision modes.
"""

import gc
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from blitz.models.parakeet_factory import create_parakeet_model


def load_audio(audio_path: str, target_sr: int = 16000) -> tuple:
    """Load audio file and return (samples, duration_sec)."""
    audio, sr = sf.read(audio_path, dtype="float32")

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if sr != target_sr:
        import torchaudio

        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio_tensor).squeeze().numpy()
        sr = target_sr

    duration_sec = len(audio) / sr
    return audio, duration_sec


def benchmark_precision(precision: str, audio_path: str, iterations: int = 3) -> dict:
    """Benchmark a specific precision mode."""
    print(f"\n{'='*60}")
    print(f"Benchmarking precision: {precision.upper()}")
    print("=" * 60)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load audio
    audio, duration_sec = load_audio(audio_path)
    print(f"Audio duration: {duration_sec:.2f}s")

    # Load model with specified precision
    load_start = time.perf_counter()
    try:
        model = create_parakeet_model(backend="nemo", precision=precision)
        load_time_ms = (time.perf_counter() - load_start) * 1000
        print(f"Model loaded in {load_time_ms:.0f}ms")
    except Exception as e:
        print(f"Failed to load model with {precision}: {e}")
        return None

    # Warmup
    print("Warmup run...")
    _ = model.transcribe(audio, timestamps=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark runs
    times = []
    transcripts = []
    for i in range(iterations):
        gc.collect()
        start = time.perf_counter()
        transcript, _ = model.transcribe(audio, timestamps=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
        transcripts.append(transcript)
        rtf = elapsed_ms / 1000 / duration_sec
        print(f"  Iteration {i+1}: {elapsed_ms:.1f}ms (RTF: {rtf:.3f}x)")

    # Calculate stats
    avg_ms = np.mean(times)
    std_ms = np.std(times)
    min_ms = np.min(times)
    max_ms = np.max(times)
    avg_rtf = avg_ms / 1000 / duration_sec

    result = {
        "precision": precision,
        "load_time_ms": load_time_ms,
        "avg_inference_ms": avg_ms,
        "std_inference_ms": std_ms,
        "min_inference_ms": min_ms,
        "max_inference_ms": max_ms,
        "avg_rtf": avg_rtf,
        "audio_duration_sec": duration_sec,
        "transcript_sample": transcripts[0][:100] if transcripts else "",
    }

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def print_summary(results: list):
    """Print comparison summary."""
    print("\n" + "=" * 70)
    print("PRECISION BENCHMARK SUMMARY")
    print("=" * 70)

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("No valid results to display")
        return

    print(f"\n{'Precision':<12} {'Load (ms)':<12} {'Avg (ms)':<12} {'RTF':<10} {'Speedup':<12}")
    print("-" * 60)

    # Use fp32 as baseline
    baseline = next((r for r in valid_results if r["precision"] == "fp32"), valid_results[0])

    for r in valid_results:
        speedup = baseline["avg_inference_ms"] / r["avg_inference_ms"] if r["avg_inference_ms"] > 0 else 0
        speedup_str = f"{speedup:.2f}x" if r["precision"] != "fp32" else "baseline"
        print(
            f"{r['precision']:<12} "
            f"{r['load_time_ms']:<12.0f} "
            f"{r['avg_inference_ms']:<12.1f} "
            f"{r['avg_rtf']:<10.3f} "
            f"{speedup_str:<12}"
        )

    print("\n" + "=" * 70)


def main():
    """Run precision benchmarks."""
    # Test audio file
    test_audio = Path(__file__).parent.parent / "tests" / "fixtures" / "test_audio.wav"
    if not test_audio.exists():
        print(f"Test audio not found: {test_audio}")
        sys.exit(1)

    print(f"Test audio: {test_audio}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'}")

    # Precisions to test
    precisions = ["fp32", "fp16", "bfloat16"]
    results = []

    for precision in precisions:
        try:
            result = benchmark_precision(precision, str(test_audio), iterations=3)
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking {precision}: {e}")
            results.append(None)

    print_summary(results)


if __name__ == "__main__":
    main()
