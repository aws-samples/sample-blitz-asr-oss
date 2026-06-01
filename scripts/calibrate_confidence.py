#!/usr/bin/env python3
"""
Confidence Calibration Script for Language Detection Models.

Runs both Whisper and VoxLingua on test audio samples to:
1. Compare confidence score distributions
2. Measure accuracy on known languages
3. Recommend optimal thresholds for each model
4. Analyze agreement rate between models

Usage:
    python scripts/calibrate_confidence.py --audio-dir tests/fixtures
    python scripts/calibrate_confidence.py --audio-file sample.wav --language en

Output:
    - Confidence distribution statistics
    - Accuracy metrics per model
    - Recommended thresholds
    - Agreement analysis
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from blitz.models.whisper import WhisperModel
from blitz.models.voxlingua import VoxLinguaModel


def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """Load audio file and return (audio_array, sample_rate)."""
    audio, sr = sf.read(file_path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype('float32'), sr


def run_calibration(
    audio_files: List[Tuple[str, Optional[str]]],  # (path, expected_language)
    whisper_model: WhisperModel,
    voxlingua_model: VoxLinguaModel,
    verbose: bool = True
) -> Dict:
    """
    Run both models on audio files and collect statistics.

    Args:
        audio_files: List of (file_path, expected_language) tuples
        whisper_model: Initialized Whisper model
        voxlingua_model: Initialized VoxLingua model
        verbose: Print progress

    Returns:
        Dictionary with calibration results
    """
    results = {
        "whisper": {
            "confidences": [],
            "languages": [],
            "times_ms": [],
            "correct": 0,
            "total": 0
        },
        "voxlingua": {
            "confidences": [],
            "languages": [],
            "times_ms": [],
            "correct": 0,
            "total": 0
        },
        "agreement": {
            "same_language": 0,
            "total": 0
        },
        "files": []
    }

    for i, (file_path, expected_lang) in enumerate(audio_files):
        if verbose:
            print(f"\n[{i+1}/{len(audio_files)}] Processing: {os.path.basename(file_path)}")

        try:
            audio, sr = load_audio(file_path)
            duration = len(audio) / sr

            # Run Whisper
            whisper_start = time.time()
            w_lang, w_conf, w_top = whisper_model.detect_language(audio, sr)
            whisper_time = (time.time() - whisper_start) * 1000

            # Run VoxLingua
            vox_start = time.time()
            v_lang, v_conf, v_top = voxlingua_model.detect_language(audio, sr)
            vox_time = (time.time() - vox_start) * 1000

            # Record results
            results["whisper"]["confidences"].append(w_conf)
            results["whisper"]["languages"].append(w_lang)
            results["whisper"]["times_ms"].append(whisper_time)

            results["voxlingua"]["confidences"].append(v_conf)
            results["voxlingua"]["languages"].append(v_lang)
            results["voxlingua"]["times_ms"].append(vox_time)

            # Check accuracy if expected language provided
            if expected_lang:
                results["whisper"]["total"] += 1
                results["voxlingua"]["total"] += 1
                if w_lang == expected_lang:
                    results["whisper"]["correct"] += 1
                if v_lang == expected_lang:
                    results["voxlingua"]["correct"] += 1

            # Check agreement
            results["agreement"]["total"] += 1
            if w_lang == v_lang:
                results["agreement"]["same_language"] += 1

            # Store file details
            file_result = {
                "file": os.path.basename(file_path),
                "duration_sec": round(duration, 2),
                "expected": expected_lang,
                "whisper": {"lang": w_lang, "conf": round(w_conf, 4), "time_ms": round(whisper_time, 1)},
                "voxlingua": {"lang": v_lang, "conf": round(v_conf, 4), "time_ms": round(vox_time, 1)},
                "agree": w_lang == v_lang
            }
            results["files"].append(file_result)

            if verbose:
                print(f"  Duration: {duration:.1f}s")
                print(f"  Whisper:   {w_lang} ({w_conf:.3f}) in {whisper_time:.0f}ms")
                print(f"  VoxLingua: {v_lang} ({v_conf:.3f}) in {vox_time:.0f}ms")
                print(f"  Agreement: {'yes' if w_lang == v_lang else 'no'}")

        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue

    return results


def analyze_results(results: Dict) -> Dict:
    """Analyze calibration results and generate recommendations."""
    analysis = {}

    for model in ["whisper", "voxlingua"]:
        confs = results[model]["confidences"]
        times = results[model]["times_ms"]

        if not confs:
            continue

        analysis[model] = {
            "confidence": {
                "mean": round(np.mean(confs), 4),
                "std": round(np.std(confs), 4),
                "min": round(np.min(confs), 4),
                "max": round(np.max(confs), 4),
                "p25": round(np.percentile(confs, 25), 4),
                "p50": round(np.percentile(confs, 50), 4),
                "p75": round(np.percentile(confs, 75), 4),
                "p90": round(np.percentile(confs, 90), 4),
            },
            "latency_ms": {
                "mean": round(np.mean(times), 1),
                "std": round(np.std(times), 1),
                "min": round(np.min(times), 1),
                "max": round(np.max(times), 1),
                "p50": round(np.percentile(times, 50), 1),
                "p95": round(np.percentile(times, 95), 1),
            }
        }

        # Accuracy if available
        if results[model]["total"] > 0:
            accuracy = results[model]["correct"] / results[model]["total"]
            analysis[model]["accuracy"] = round(accuracy, 4)

    # Agreement rate
    if results["agreement"]["total"] > 0:
        agreement_rate = results["agreement"]["same_language"] / results["agreement"]["total"]
        analysis["agreement_rate"] = round(agreement_rate, 4)

    # Generate threshold recommendations
    analysis["recommendations"] = generate_recommendations(analysis)

    return analysis


def generate_recommendations(analysis: Dict) -> Dict:
    """Generate threshold recommendations based on analysis."""
    recommendations = {}

    for model in ["whisper", "voxlingua"]:
        if model not in analysis:
            continue

        conf = analysis[model]["confidence"]

        # Recommend threshold at ~80th percentile of confidence
        # This means ~20% of cases will need fallback
        # Adjust based on accuracy if available
        base_threshold = conf["p75"]

        # If we have accuracy data and it's high, we can be more aggressive
        if "accuracy" in analysis[model] and analysis[model]["accuracy"] > 0.95:
            # High accuracy, use lower threshold (fewer fallbacks)
            recommended = max(0.5, conf["p50"])
        else:
            # Use conservative threshold
            recommended = base_threshold

        recommendations[model] = {
            "recommended_threshold": round(recommended, 2),
            "fallback_rate_at_recommended": f"~{int((1 - recommended/conf['max']) * 100)}%",
            "reasoning": f"Based on p75={conf['p75']:.2f}, mean={conf['mean']:.2f}"
        }

    return recommendations


def print_report(analysis: Dict):
    """Print formatted analysis report."""
    print("\n" + "=" * 70)
    print("LANGUAGE DETECTION CALIBRATION REPORT")
    print("=" * 70)

    for model in ["whisper", "voxlingua"]:
        if model not in analysis:
            continue

        print(f"\n{model.upper()}")
        print("-" * 40)

        conf = analysis[model]["confidence"]
        print(f"  Confidence Distribution:")
        print(f"    Mean:  {conf['mean']:.3f} (±{conf['std']:.3f})")
        print(f"    Range: {conf['min']:.3f} - {conf['max']:.3f}")
        print(f"    P25/P50/P75: {conf['p25']:.3f} / {conf['p50']:.3f} / {conf['p75']:.3f}")
        print(f"    P90:   {conf['p90']:.3f}")

        lat = analysis[model]["latency_ms"]
        print(f"\n  Latency (ms):")
        print(f"    Mean:  {lat['mean']:.0f}ms (±{lat['std']:.0f})")
        print(f"    P50:   {lat['p50']:.0f}ms")
        print(f"    P95:   {lat['p95']:.0f}ms")

        if "accuracy" in analysis[model]:
            print(f"\n  Accuracy: {analysis[model]['accuracy']*100:.1f}%")

    if "agreement_rate" in analysis:
        print(f"\n\nMODEL AGREEMENT RATE: {analysis['agreement_rate']*100:.1f}%")

    if "recommendations" in analysis:
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        for model, rec in analysis["recommendations"].items():
            print(f"\n{model.upper()}:")
            print(f"  Recommended threshold: {rec['recommended_threshold']}")
            print(f"  Expected fallback rate: {rec['fallback_rate_at_recommended']}")
            print(f"  Reasoning: {rec['reasoning']}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate language detection confidence thresholds"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        help="Directory containing audio files to test"
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        help="Single audio file to test"
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Expected language code for accuracy testing (e.g., 'en')"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file output"
    )

    args = parser.parse_args()

    if not args.audio_dir and not args.audio_file:
        print("Error: Must specify --audio-dir or --audio-file")
        sys.exit(1)

    # Collect audio files
    audio_files = []

    if args.audio_file:
        audio_files.append((args.audio_file, args.language))

    if args.audio_dir:
        audio_dir = Path(args.audio_dir)
        for ext in ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"]:
            for f in audio_dir.glob(ext):
                # Try to extract language from filename (e.g., "sample_en.wav")
                expected = None
                if args.language:
                    expected = args.language
                else:
                    # Try to parse from filename
                    parts = f.stem.split("_")
                    if len(parts) >= 2 and len(parts[-1]) == 2:
                        expected = parts[-1]
                audio_files.append((str(f), expected))

    if not audio_files:
        print("Error: No audio files found")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio file(s)")

    # Initialize models
    print("\nInitializing models...")
    print("-" * 40)

    whisper_model = WhisperModel(
        primary_model_name="openai/whisper-large-v3-turbo",
        language_detection_model="large-v3-turbo"
    )

    voxlingua_model = VoxLinguaModel()

    # Run calibration
    print("\nRunning calibration...")
    print("-" * 40)

    results = run_calibration(
        audio_files,
        whisper_model,
        voxlingua_model,
        verbose=not args.quiet
    )

    # Analyze results
    analysis = analyze_results(results)

    # Print report
    print_report(analysis)

    # Save to file if requested
    if args.output:
        output_data = {
            "analysis": analysis,
            "files": results["files"]
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
