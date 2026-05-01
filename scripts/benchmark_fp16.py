#!/usr/bin/env python3
"""
Benchmark FP16 ONNX vs BFloat16 NeMo Parakeet models.

Compares:
- Inference latency (cold start, warm, per-minute-of-audio)
- Memory usage (VRAM, RAM)
- Throughput (audio minutes per second)
- Word Error Rate (if ground truth provided)

Usage:
    python scripts/benchmark_fp16.py --audio-dir ./test_audio/
    python scripts/benchmark_fp16.py --audio-file sample.wav --iterations 10
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    backend: str
    precision: str
    audio_file: str
    audio_duration_sec: float
    cold_start_ms: float
    inference_ms: float
    total_ms: float
    rtf: float  # Real-time factor (inference_time / audio_duration)
    transcript_length: int
    vram_mb: Optional[float] = None
    ram_mb: Optional[float] = None


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark summary."""
    backend: str
    precision: str
    num_files: int
    total_audio_sec: float
    avg_inference_ms: float
    avg_rtf: float
    p50_inference_ms: float
    p95_inference_ms: float
    p99_inference_ms: float
    throughput_audio_min_per_sec: float
    peak_vram_mb: Optional[float] = None


def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except:
        pass
    return None


def get_ram_usage():
    """Get current RAM usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        pass
    return None


def load_audio(audio_path: str, target_sr: int = 16000) -> tuple:
    """Load audio file and return (samples, duration_sec)."""
    import soundfile as sf

    audio, sr = sf.read(audio_path, dtype='float32')

    # Convert stereo to mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if sr != target_sr:
        import torchaudio
        import torch
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio_tensor).squeeze().numpy()
        sr = target_sr

    duration_sec = len(audio) / sr
    return audio, duration_sec


def benchmark_model(
    model,
    audio_path: str,
    backend: str,
    precision: str,
    warmup: bool = True
) -> BenchmarkResult:
    """Benchmark a single model on a single audio file."""
    import torch

    # Load audio
    audio, duration_sec = load_audio(audio_path)

    # Warmup run (not timed)
    if warmup:
        logger.debug("  Warmup run...")
        _ = model.transcribe(audio, timestamps=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Clear caches
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Measure VRAM before
    vram_before = get_gpu_memory()
    ram_before = get_ram_usage()

    # Timed inference
    start = time.perf_counter()
    transcript, _ = model.transcribe(audio, timestamps=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    # Measure VRAM after
    vram_after = get_gpu_memory()
    ram_after = get_ram_usage()

    inference_ms = (end - start) * 1000
    rtf = inference_ms / 1000 / duration_sec if duration_sec > 0 else 0

    return BenchmarkResult(
        backend=backend,
        precision=precision,
        audio_file=os.path.basename(audio_path),
        audio_duration_sec=duration_sec,
        cold_start_ms=0.0,  # Filled in separately
        inference_ms=inference_ms,
        total_ms=inference_ms,
        rtf=rtf,
        transcript_length=len(transcript) if transcript else 0,
        vram_mb=vram_after - vram_before if vram_before and vram_after else None,
        ram_mb=ram_after - ram_before if ram_before and ram_after else None,
    )


def create_summary(results: List[BenchmarkResult]) -> BenchmarkSummary:
    """Create summary statistics from benchmark results."""
    if not results:
        raise ValueError("No results to summarize")

    inference_times = [r.inference_ms for r in results]
    rtfs = [r.rtf for r in results]
    total_audio = sum(r.audio_duration_sec for r in results)
    total_inference_sec = sum(r.inference_ms for r in results) / 1000

    return BenchmarkSummary(
        backend=results[0].backend,
        precision=results[0].precision,
        num_files=len(results),
        total_audio_sec=total_audio,
        avg_inference_ms=np.mean(inference_times),
        avg_rtf=np.mean(rtfs),
        p50_inference_ms=np.percentile(inference_times, 50),
        p95_inference_ms=np.percentile(inference_times, 95),
        p99_inference_ms=np.percentile(inference_times, 99),
        throughput_audio_min_per_sec=(total_audio / 60) / total_inference_sec if total_inference_sec > 0 else 0,
        peak_vram_mb=max((r.vram_mb for r in results if r.vram_mb), default=None),
    )


def print_comparison(nemo_summary: BenchmarkSummary, onnx_summary: BenchmarkSummary):
    """Print side-by-side comparison of results."""
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON: NeMo (BFloat16) vs ONNX (FP16)")
    print("=" * 70)

    def fmt(val, unit=""):
        if val is None:
            return "N/A"
        if isinstance(val, float):
            return f"{val:.2f}{unit}"
        return f"{val}{unit}"

    def speedup(nemo_val, onnx_val):
        if nemo_val and onnx_val and onnx_val > 0:
            ratio = nemo_val / onnx_val
            if ratio > 1:
                return f"({ratio:.2f}x faster)"
            else:
                return f"({1/ratio:.2f}x slower)"
        return ""

    rows = [
        ("Backend", nemo_summary.backend, onnx_summary.backend, ""),
        ("Precision", nemo_summary.precision, onnx_summary.precision, ""),
        ("Files tested", str(nemo_summary.num_files), str(onnx_summary.num_files), ""),
        ("Total audio", fmt(nemo_summary.total_audio_sec / 60, " min"), fmt(onnx_summary.total_audio_sec / 60, " min"), ""),
        ("", "", "", ""),
        ("Avg inference", fmt(nemo_summary.avg_inference_ms, " ms"), fmt(onnx_summary.avg_inference_ms, " ms"),
         speedup(nemo_summary.avg_inference_ms, onnx_summary.avg_inference_ms)),
        ("P50 inference", fmt(nemo_summary.p50_inference_ms, " ms"), fmt(onnx_summary.p50_inference_ms, " ms"), ""),
        ("P95 inference", fmt(nemo_summary.p95_inference_ms, " ms"), fmt(onnx_summary.p95_inference_ms, " ms"), ""),
        ("P99 inference", fmt(nemo_summary.p99_inference_ms, " ms"), fmt(onnx_summary.p99_inference_ms, " ms"), ""),
        ("", "", "", ""),
        ("Avg RTF", fmt(nemo_summary.avg_rtf, "x"), fmt(onnx_summary.avg_rtf, "x"),
         speedup(nemo_summary.avg_rtf, onnx_summary.avg_rtf)),
        ("Throughput", fmt(nemo_summary.throughput_audio_min_per_sec, " min/s"), fmt(onnx_summary.throughput_audio_min_per_sec, " min/s"),
         speedup(onnx_summary.throughput_audio_min_per_sec, nemo_summary.throughput_audio_min_per_sec)),
        ("Peak VRAM", fmt(nemo_summary.peak_vram_mb, " MB"), fmt(onnx_summary.peak_vram_mb, " MB"), ""),
    ]

    print(f"\n{'Metric':<20} {'NeMo (BF16)':<20} {'ONNX (FP16)':<20} {'Comparison':<20}")
    print("-" * 80)
    for row in rows:
        if row[0] == "":
            print()
        else:
            print(f"{row[0]:<20} {row[1]:<20} {row[2]:<20} {row[3]:<20}")

    print("\n" + "=" * 70)


def run_benchmark(
    audio_files: List[str],
    iterations: int = 1,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """Run full benchmark comparison."""
    from blitz.models.parakeet_factory import create_parakeet_model

    results = {
        "nemo": [],
        "onnx": []
    }

    # Benchmark NeMo (BFloat16)
    logger.info("\n" + "=" * 50)
    logger.info("Loading NeMo model (BFloat16)...")
    logger.info("=" * 50)

    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    nemo_start = time.perf_counter()
    try:
        nemo_model = create_parakeet_model(
            backend="nemo",
            precision="bfloat16"
        )
        nemo_cold_start = (time.perf_counter() - nemo_start) * 1000
        logger.info(f"NeMo model loaded in {nemo_cold_start:.0f}ms")

        for audio_file in audio_files:
            logger.info(f"\nBenchmarking NeMo on: {os.path.basename(audio_file)}")
            for i in range(iterations):
                result = benchmark_model(
                    nemo_model,
                    audio_file,
                    backend="nemo",
                    precision="bfloat16",
                    warmup=(i == 0)
                )
                if i == 0:
                    result.cold_start_ms = nemo_cold_start
                results["nemo"].append(result)
                logger.info(f"  Iteration {i+1}: {result.inference_ms:.1f}ms (RTF: {result.rtf:.3f}x)")

        # Cleanup NeMo model
        del nemo_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Failed to benchmark NeMo model: {e}")
        raise

    # Benchmark ONNX (FP16)
    logger.info("\n" + "=" * 50)
    logger.info("Loading ONNX model (FP16)...")
    logger.info("=" * 50)

    onnx_start = time.perf_counter()
    try:
        onnx_model = create_parakeet_model(
            backend="onnx",
            precision="fp16"
        )
        onnx_cold_start = (time.perf_counter() - onnx_start) * 1000
        logger.info(f"ONNX model loaded in {onnx_cold_start:.0f}ms")

        for audio_file in audio_files:
            logger.info(f"\nBenchmarking ONNX on: {os.path.basename(audio_file)}")
            for i in range(iterations):
                result = benchmark_model(
                    onnx_model,
                    audio_file,
                    backend="onnx",
                    precision="fp16",
                    warmup=(i == 0)
                )
                if i == 0:
                    result.cold_start_ms = onnx_cold_start
                results["onnx"].append(result)
                logger.info(f"  Iteration {i+1}: {result.inference_ms:.1f}ms (RTF: {result.rtf:.3f}x)")

        del onnx_model

    except ImportError as e:
        logger.warning(f"ONNX backend not available: {e}")
        logger.warning("Install with: pip install -r requirements-onnx.txt")
    except FileNotFoundError as e:
        logger.warning(f"ONNX models not found: {e}")
        logger.warning("Download with: python scripts/download_fp16_models.py")
    except Exception as e:
        logger.error(f"Failed to benchmark ONNX model: {e}")
        raise

    # Generate summaries
    summaries = {}
    if results["nemo"]:
        summaries["nemo"] = create_summary(results["nemo"])
    if results["onnx"]:
        summaries["onnx"] = create_summary(results["onnx"])

    # Print comparison
    if "nemo" in summaries and "onnx" in summaries:
        print_comparison(summaries["nemo"], summaries["onnx"])

    # Save results
    if output_file:
        output_data = {
            "results": {
                "nemo": [asdict(r) for r in results["nemo"]],
                "onnx": [asdict(r) for r in results["onnx"]]
            },
            "summaries": {k: asdict(v) for k, v in summaries.items()}
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nResults saved to: {output_file}")

    return {"results": results, "summaries": summaries}


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FP16 ONNX vs BFloat16 NeMo Parakeet models"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        help="Directory containing audio files to benchmark"
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        help="Single audio file to benchmark"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per file (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file for results (default: benchmark_results.json)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Maximum number of files to benchmark (default: 10)"
    )

    args = parser.parse_args()

    # Collect audio files
    audio_files = []
    if args.audio_file:
        audio_files = [args.audio_file]
    elif args.audio_dir:
        audio_dir = Path(args.audio_dir)
        extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
        for ext in extensions:
            audio_files.extend(audio_dir.glob(f"*{ext}"))
        audio_files = [str(f) for f in sorted(audio_files)[:args.max_files]]
    else:
        logger.error("Must specify --audio-file or --audio-dir")
        sys.exit(1)

    if not audio_files:
        logger.error("No audio files found")
        sys.exit(1)

    logger.info(f"Benchmarking {len(audio_files)} audio file(s) with {args.iterations} iteration(s) each")

    run_benchmark(
        audio_files=audio_files,
        iterations=args.iterations,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
