#!/usr/bin/env python3
"""
Benchmark Parakeet CTC-110M vs 0.6B on English audio.

Compares the fast English-only CTC model against the multilingual 0.6B:
- CTC-110M: 110M params, RTFx ~5345, WER 7.49% (English only)
- 0.6B-v3:  600M params, RTFx ~3332, WER 6.32% (25 European languages)

Expected speedup: ~60% faster for CTC-110M on English.
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

from blitz.models.parakeet_factory import (
    create_parakeet_model,
    create_parakeet_ctc_model,  # Creates Tier 1 model
)


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


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def benchmark_model(
    model_name: str,
    model,
    audio: np.ndarray,
    duration_sec: float,
    iterations: int = 3,
) -> dict:
    """Benchmark a model on the given audio."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print("=" * 60)
    print(f"Audio duration: {duration_sec:.2f}s")

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
        rtfx = 1 / rtf if rtf > 0 else 0
        print(f"  Iteration {i+1}: {elapsed_ms:.1f}ms (RTF: {rtf:.4f}, RTFx: {rtfx:.0f})")

    # Calculate stats
    avg_ms = np.mean(times)
    std_ms = np.std(times)
    min_ms = np.min(times)
    max_ms = np.max(times)
    avg_rtf = avg_ms / 1000 / duration_sec
    avg_rtfx = 1 / avg_rtf if avg_rtf > 0 else 0

    return {
        "model_name": model_name,
        "avg_inference_ms": avg_ms,
        "std_inference_ms": std_ms,
        "min_inference_ms": min_ms,
        "max_inference_ms": max_ms,
        "avg_rtf": avg_rtf,
        "avg_rtfx": avg_rtfx,
        "audio_duration_sec": duration_sec,
        "transcript": transcripts[0],
    }


def print_comparison(results: list, audio_file: str):
    """Print comparison between models."""
    print("\n" + "=" * 70)
    print(f"BENCHMARK COMPARISON: {Path(audio_file).name}")
    print("=" * 70)

    if len(results) < 2:
        print("Need at least 2 results for comparison")
        return

    print(f"\n{'Model':<25} {'Avg (ms)':<12} {'RTFx':<10} {'Speedup':<12}")
    print("-" * 65)

    # Use 0.6B as baseline
    baseline = next((r for r in results if "0.6b" in r["model_name"].lower()), results[0])

    for r in results:
        speedup = baseline["avg_inference_ms"] / r["avg_inference_ms"]
        speedup_str = f"{speedup:.2f}x" if "ctc" in r["model_name"].lower() else "baseline"
        print(
            f"{r['model_name']:<25} "
            f"{r['avg_inference_ms']:<12.1f} "
            f"{r['avg_rtfx']:<10.0f} "
            f"{speedup_str:<12}"
        )

    print("\n" + "-" * 65)
    ctc_result = next((r for r in results if "ctc" in r["model_name"].lower()), None)
    if ctc_result and baseline:
        speedup_pct = ((baseline["avg_inference_ms"] - ctc_result["avg_inference_ms"])
                      / baseline["avg_inference_ms"] * 100)
        print(f"CTC-110M is {speedup_pct:.1f}% faster than 0.6B")


def save_transcript(transcript: str, output_path: Path, model_name: str, stats: dict):
    """Save transcript to markdown file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = f"""# {model_name} Transcript

## Stats
- Model: {model_name}
- Audio Duration: {stats['audio_duration_sec']:.2f}s
- Inference Time: {stats['avg_inference_ms']:.1f}ms
- RTFx: {stats['avg_rtfx']:.0f}
- RTF: {stats['avg_rtf']:.4f}

## Transcript

{transcript}
"""

    output_path.write_text(content)
    print(f"Saved transcript to {output_path}")


def main():
    """Run CTC vs 0.6B benchmark."""
    # Test files
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    short_audio = fixtures_dir / "english.mp3"
    long_audio = fixtures_dir / "test-hour-audio_processed_asr_ready.wav"

    # Output directory for transcripts
    output_dir = fixtures_dir / "benchmark-transcripts"

    print("=" * 70)
    print("PARAKEET CTC-110M vs 0.6B BENCHMARK")
    print("=" * 70)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Short audio: {short_audio}")
    print(f"Long audio: {long_audio}")

    # Load models
    print("\nLoading models...")

    print("\n[1/2] Loading Parakeet 0.6B (bfloat16)...")
    load_start = time.perf_counter()
    model_06b = create_parakeet_model(backend="nemo", precision="bfloat16")
    load_06b_ms = (time.perf_counter() - load_start) * 1000
    mem_06b = get_gpu_memory_mb()
    print(f"  Loaded in {load_06b_ms:.0f}ms, GPU memory: {mem_06b:.0f}MB")

    print("\n[2/2] Loading Parakeet CTC-110M (fp16)...")
    load_start = time.perf_counter()
    model_ctc = create_parakeet_ctc_model(precision="fp16")
    load_ctc_ms = (time.perf_counter() - load_start) * 1000
    mem_ctc = get_gpu_memory_mb() - mem_06b
    print(f"  Loaded in {load_ctc_ms:.0f}ms, additional GPU memory: {mem_ctc:.0f}MB")

    total_mem = get_gpu_memory_mb()
    print(f"\nTotal GPU memory with both models: {total_mem:.0f}MB")

    # Run benchmarks
    all_results = []

    # Short audio benchmark
    if short_audio.exists():
        print("\n" + "#" * 70)
        print("SHORT AUDIO BENCHMARK")
        print("#" * 70)

        audio, duration = load_audio(str(short_audio))

        result_06b = benchmark_model("Parakeet-0.6B-v3", model_06b, audio, duration)
        result_ctc = benchmark_model("Parakeet-CTC-110M", model_ctc, audio, duration)

        print_comparison([result_06b, result_ctc], str(short_audio))
        all_results.extend([result_06b, result_ctc])

    # Long audio benchmark (Jony Ive)
    if long_audio.exists():
        print("\n" + "#" * 70)
        print("LONG AUDIO BENCHMARK (Jony Ive Interview)")
        print("#" * 70)

        audio, duration = load_audio(str(long_audio))

        result_06b = benchmark_model("Parakeet-0.6B-v3", model_06b, audio, duration)
        result_ctc = benchmark_model("Parakeet-CTC-110M", model_ctc, audio, duration)

        print_comparison([result_06b, result_ctc], str(long_audio))
        all_results.extend([result_06b, result_ctc])

        # Save CTC transcript
        save_transcript(
            result_ctc["transcript"],
            output_dir / "ctc-110m.md",
            "Parakeet-CTC-110M",
            result_ctc
        )
    else:
        print(f"\nLong audio not found: {long_audio}")
        print("Skipping long audio benchmark")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nModels loaded:")
    print(f"  - Parakeet 0.6B (bfloat16): {load_06b_ms:.0f}ms load time")
    print(f"  - Parakeet CTC-110M (fp16): {load_ctc_ms:.0f}ms load time")
    print(f"\nGPU Memory:")
    print(f"  - 0.6B alone: ~{mem_06b:.0f}MB")
    print(f"  - CTC-110M additional: ~{mem_ctc:.0f}MB")
    print(f"  - Both models: ~{total_mem:.0f}MB")
    print(f"\nNote: g6.xlarge (L4 24GB) has plenty of headroom for both models")

    # Cleanup
    del model_06b
    del model_ctc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
