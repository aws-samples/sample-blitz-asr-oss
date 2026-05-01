#!/usr/bin/env python3
"""
Standalone benchmark for FP16 ONNX Parakeet model.

Does NOT require NeMo or the full blitz stack - just onnx-asr and basic deps.

Usage:
    python scripts/benchmark_onnx_standalone.py --audio-dir ./tests/fixtures/
    python scripts/benchmark_onnx_standalone.py --audio-file sample.wav --iterations 10
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
    transcript_preview: str
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
        try:
            import torchaudio
            import torch
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio_tensor).squeeze().numpy()
        except ImportError:
            # Fallback: simple linear interpolation
            from scipy import signal
            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, num_samples)
        sr = target_sr

    duration_sec = len(audio) / sr
    return audio, duration_sec


def download_onnx_models(model_path: str) -> bool:
    """Download FP16 ONNX models if not present."""
    from huggingface_hub import hf_hub_download

    required_files = [
        ("grikdotnet/parakeet-tdt-0.6b-fp16", "encoder-model.fp16.onnx"),
        ("grikdotnet/parakeet-tdt-0.6b-fp16", "decoder_joint-model.fp16.onnx"),
        ("istupakov/parakeet-tdt-0.6b-v3-onnx", "config.json"),
        ("istupakov/parakeet-tdt-0.6b-v3-onnx", "vocab.txt"),
        ("istupakov/parakeet-tdt-0.6b-v3-onnx", "nemo128.onnx"),
    ]

    os.makedirs(model_path, exist_ok=True)

    for repo_id, filename in required_files:
        local_file = os.path.join(model_path, filename)
        if not os.path.exists(local_file):
            logger.info(f"Downloading {filename} from {repo_id}...")
            hf_hub_download(repo_id, filename, local_dir=model_path)

    return True


def create_onnx_model(model_path: str):
    """Create ONNX ASR model."""
    try:
        from onnx_asr import OnnxASR
    except ImportError:
        raise ImportError(
            "onnx-asr not installed. Install with: pip install onnx-asr onnxruntime-gpu"
        )

    # Download models if needed
    download_onnx_models(model_path)

    # Create model
    model = OnnxASR(model_path)
    return model


def benchmark_onnx_model(
    model,
    audio_path: str,
    warmup: bool = True
) -> BenchmarkResult:
    """Benchmark ONNX model on a single audio file."""
    # Load audio
    audio, duration_sec = load_audio(audio_path)

    # Warmup run (not timed)
    if warmup:
        logger.debug("  Warmup run...")
        _ = model.recognize(audio)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except:
            pass

    # Clear caches
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass

    # Measure VRAM before
    vram_before = get_gpu_memory()
    ram_before = get_ram_usage()

    # Timed inference
    start = time.perf_counter()
    transcript = model.recognize(audio)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except:
        pass
    end = time.perf_counter()

    # Measure VRAM after
    vram_after = get_gpu_memory()
    ram_after = get_ram_usage()

    inference_ms = (end - start) * 1000
    rtf = inference_ms / 1000 / duration_sec if duration_sec > 0 else 0

    return BenchmarkResult(
        backend="onnx",
        precision="fp16",
        audio_file=os.path.basename(audio_path),
        audio_duration_sec=duration_sec,
        cold_start_ms=0.0,  # Filled in separately
        inference_ms=inference_ms,
        total_ms=inference_ms,
        rtf=rtf,
        transcript_length=len(transcript) if transcript else 0,
        transcript_preview=transcript[:100] + "..." if transcript and len(transcript) > 100 else transcript,
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


def print_results(summary: BenchmarkSummary):
    """Print benchmark results."""
    print("\n" + "=" * 70)
    print("ONNX FP16 BENCHMARK RESULTS")
    print("=" * 70)

    def fmt(val, unit=""):
        if val is None:
            return "N/A"
        if isinstance(val, float):
            return f"{val:.2f}{unit}"
        return f"{val}{unit}"

    rows = [
        ("Backend", summary.backend),
        ("Precision", summary.precision),
        ("Files tested", str(summary.num_files)),
        ("Total audio", fmt(summary.total_audio_sec / 60, " min")),
        ("", ""),
        ("Avg inference", fmt(summary.avg_inference_ms, " ms")),
        ("P50 inference", fmt(summary.p50_inference_ms, " ms")),
        ("P95 inference", fmt(summary.p95_inference_ms, " ms")),
        ("P99 inference", fmt(summary.p99_inference_ms, " ms")),
        ("", ""),
        ("Avg RTF", fmt(summary.avg_rtf, "x")),
        ("Throughput", fmt(summary.throughput_audio_min_per_sec, " min/s")),
        ("Peak VRAM", fmt(summary.peak_vram_mb, " MB")),
    ]

    print(f"\n{'Metric':<25} {'Value':<25}")
    print("-" * 50)
    for row in rows:
        if row[0] == "":
            print()
        else:
            print(f"{row[0]:<25} {row[1]:<25}")

    print("\n" + "=" * 70)


def run_benchmark(
    audio_files: List[str],
    model_path: str = "./models/parakeet",
    iterations: int = 1,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """Run ONNX FP16 benchmark."""
    results = []

    # Load ONNX model
    logger.info("\n" + "=" * 50)
    logger.info("Loading ONNX model (FP16)...")
    logger.info("=" * 50)

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

    onnx_start = time.perf_counter()
    try:
        onnx_model = create_onnx_model(model_path)
        onnx_cold_start = (time.perf_counter() - onnx_start) * 1000
        logger.info(f"ONNX model loaded in {onnx_cold_start:.0f}ms")

        for audio_file in audio_files:
            logger.info(f"\nBenchmarking ONNX on: {os.path.basename(audio_file)}")
            for i in range(iterations):
                result = benchmark_onnx_model(
                    onnx_model,
                    audio_file,
                    warmup=(i == 0)
                )
                if i == 0:
                    result.cold_start_ms = onnx_cold_start
                results.append(result)
                logger.info(f"  Iteration {i+1}: {result.inference_ms:.1f}ms (RTF: {result.rtf:.3f}x)")
                logger.info(f"    Transcript: {result.transcript_preview}")

        del onnx_model

    except ImportError as e:
        logger.error(f"ONNX backend not available: {e}")
        logger.error("Install with: pip install onnx-asr onnxruntime-gpu")
        raise
    except FileNotFoundError as e:
        logger.error(f"ONNX models not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to benchmark ONNX model: {e}")
        raise

    # Generate summary
    if results:
        summary = create_summary(results)
        print_results(summary)

        # Save results
        if output_file:
            output_data = {
                "results": [asdict(r) for r in results],
                "summary": asdict(summary)
            }
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"\nResults saved to: {output_file}")

        return {"results": results, "summary": summary}

    return {"results": [], "summary": None}


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FP16 ONNX Parakeet model (standalone, no NeMo required)"
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
        "--model-path",
        type=str,
        default="./models/parakeet",
        help="Path to ONNX model files (default: ./models/parakeet)"
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
        default="benchmark_onnx_results.json",
        help="Output JSON file for results (default: benchmark_onnx_results.json)"
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
        model_path=args.model_path,
        iterations=args.iterations,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
