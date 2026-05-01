#!/usr/bin/env python3
"""
Calculate Word Error Rate (WER) between ASR transcriptions and ground truth.

Compares model outputs against YouTube auto-generated transcript as baseline.
"""

import json
import re
import sys
from pathlib import Path


def normalize_text(text: str) -> str:
    """
    Normalize text for WER comparison.

    - Lowercase
    - Remove punctuation
    - Collapse whitespace
    - Remove common filler words that may differ between transcripts
    """
    text = text.lower()

    # Remove common annotations like [Applause], [Music], etc.
    text = re.sub(r'\[.*?\]', '', text)

    # Remove punctuation except apostrophes in contractions
    text = re.sub(r"[^\w\s']", ' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def calculate_wer(reference: str, hypothesis: str) -> dict:
    """
    Calculate Word Error Rate and related metrics.

    WER = (S + D + I) / N

    Where:
    - S = substitutions
    - D = deletions
    - I = insertions
    - N = number of words in reference

    Returns dict with WER, word counts, and error breakdown.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Dynamic programming for edit distance
    n = len(ref_words)
    m = len(hyp_words)

    # Create DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Initialize base cases
    for i in range(n + 1):
        dp[i][0] = i  # deletions
    for j in range(m + 1):
        dp[0][j] = j  # insertions

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + 1     # substitution
                )

    # Backtrack to count S, D, I
    i, j = n, m
    substitutions = 0
    deletions = 0
    insertions = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    wer = (substitutions + deletions + insertions) / max(n, 1)
    accuracy = max(0, 1 - wer) * 100

    return {
        "wer": wer,
        "wer_percent": wer * 100,
        "accuracy_percent": accuracy,
        "reference_words": n,
        "hypothesis_words": m,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "total_errors": substitutions + deletions + insertions
    }


def load_youtube_transcript(path: Path) -> str:
    """Load YouTube transcript from markdown file."""
    text = path.read_text()
    # Remove line numbers and join
    lines = []
    for line in text.strip().split('\n'):
        # Skip empty lines and annotation-only lines
        line = line.strip()
        if line and not line.startswith('[') and not line.endswith(']'):
            lines.append(line)
    return ' '.join(lines)


def load_model_transcript(path: Path) -> str:
    """Load model transcript from JSON response file."""
    content = path.read_text()
    try:
        data = json.loads(content)
        return data.get('text', '')
    except json.JSONDecodeError:
        # Plain text file
        return content


def main():
    base_dir = Path(__file__).parent.parent / 'benchmark-transcripts'

    # Load transcripts
    youtube_path = base_dir / 'jony-ive-youtube-gold-standard.md'
    fp32_path = base_dir / 'fp32.md'
    bf16_path = base_dir / 'bf16.md'
    fp16_path = base_dir / 'fp16.md'

    print("=" * 70)
    print("PARAKEET ASR ACCURACY BENCHMARK")
    print("Reference: YouTube Auto-Generated Transcript")
    print("=" * 70)
    print()

    # Load YouTube ground truth
    youtube_raw = load_youtube_transcript(youtube_path)
    youtube_normalized = normalize_text(youtube_raw)
    print(f"YouTube transcript: {len(youtube_raw):,} chars, {len(youtube_normalized.split()):,} words")
    print()

    # FP32 (NeMo)
    if fp32_path.exists() and fp32_path.stat().st_size > 0:
        fp32_raw = load_model_transcript(fp32_path)
        fp32_normalized = normalize_text(fp32_raw)
        print(f"FP32 (NeMo) transcript: {len(fp32_raw):,} chars, {len(fp32_normalized.split()):,} words")

        fp32_metrics = calculate_wer(youtube_normalized, fp32_normalized)
        print(f"\n--- FP32 (NeMo) vs YouTube ---")
        print(f"Word Error Rate:  {fp32_metrics['wer_percent']:.2f}%")
        print(f"Accuracy:         {fp32_metrics['accuracy_percent']:.2f}%")
        print(f"Substitutions:    {fp32_metrics['substitutions']:,}")
        print(f"Deletions:        {fp32_metrics['deletions']:,}")
        print(f"Insertions:       {fp32_metrics['insertions']:,}")
        print(f"Total Errors:     {fp32_metrics['total_errors']:,} / {fp32_metrics['reference_words']:,} words")
    else:
        print("FP32 transcript: NOT AVAILABLE")
        fp32_metrics = None

    print()

    # BF16 (NeMo)
    if bf16_path.exists() and bf16_path.stat().st_size > 0:
        bf16_raw = load_model_transcript(bf16_path)
        bf16_normalized = normalize_text(bf16_raw)
        print(f"BF16 (NeMo) transcript: {len(bf16_raw):,} chars, {len(bf16_normalized.split()):,} words")

        bf16_metrics = calculate_wer(youtube_normalized, bf16_normalized)
        print(f"\n--- BF16 (NeMo) vs YouTube ---")
        print(f"Word Error Rate:  {bf16_metrics['wer_percent']:.2f}%")
        print(f"Accuracy:         {bf16_metrics['accuracy_percent']:.2f}%")
        print(f"Substitutions:    {bf16_metrics['substitutions']:,}")
        print(f"Deletions:        {bf16_metrics['deletions']:,}")
        print(f"Insertions:       {bf16_metrics['insertions']:,}")
        print(f"Total Errors:     {bf16_metrics['total_errors']:,} / {bf16_metrics['reference_words']:,} words")
    else:
        print("BF16 transcript: NOT AVAILABLE")
        bf16_metrics = None

    print()

    # FP16 (ONNX)
    if fp16_path.exists() and fp16_path.stat().st_size > 0:
        fp16_raw = load_model_transcript(fp16_path)
        fp16_normalized = normalize_text(fp16_raw)
        print(f"FP16 (ONNX) transcript: {len(fp16_raw):,} chars, {len(fp16_normalized.split()):,} words")

        fp16_metrics = calculate_wer(youtube_normalized, fp16_normalized)
        print(f"\n--- FP16 (ONNX) vs YouTube ---")
        print(f"Word Error Rate:  {fp16_metrics['wer_percent']:.2f}%")
        print(f"Accuracy:         {fp16_metrics['accuracy_percent']:.2f}%")
        print(f"Substitutions:    {fp16_metrics['substitutions']:,}")
        print(f"Deletions:        {fp16_metrics['deletions']:,}")
        print(f"Insertions:       {fp16_metrics['insertions']:,}")
        print(f"Total Errors:     {fp16_metrics['total_errors']:,} / {fp16_metrics['reference_words']:,} words")
    else:
        print("FP16 transcript: NOT AVAILABLE (file empty)")
        fp16_metrics = None

    print()
    print("=" * 70)
    print("SUMMARY - ACCURACY COMPARISON")
    print("=" * 70)

    # Collect all available metrics
    results = []
    if fp32_metrics:
        results.append(("FP32 (NeMo)", fp32_metrics))
    if bf16_metrics:
        results.append(("BF16 (NeMo)", bf16_metrics))
    if fp16_metrics:
        results.append(("FP16 (ONNX)", fp16_metrics))

    if len(results) >= 2:
        # Sort by WER (lower is better)
        results.sort(key=lambda x: x[1]['wer'])

        print(f"\n{'Backend':<15} {'WER':<10} {'Accuracy':<12} {'Errors':<10}")
        print("-" * 50)
        for name, metrics in results:
            print(f"{name:<15} {metrics['wer_percent']:>6.2f}%   {metrics['accuracy_percent']:>6.2f}%     {metrics['total_errors']:>6,}")

        winner_name, winner_metrics = results[0]
        runner_up_name, runner_up_metrics = results[1]
        improvement = runner_up_metrics['wer_percent'] - winner_metrics['wer_percent']

        print()
        print(f"🏆 WINNER: {winner_name}")
        print(f"   WER: {winner_metrics['wer_percent']:.2f}% ({improvement:.2f}% better than {runner_up_name})")
        print(f"   Accuracy: {winner_metrics['accuracy_percent']:.2f}%")
    elif len(results) == 1:
        name, metrics = results[0]
        print(f"Only {name} available: {metrics['accuracy_percent']:.2f}% accuracy")
    else:
        print("No model transcripts available for comparison")


if __name__ == '__main__':
    main()
