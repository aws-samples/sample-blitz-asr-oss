#!/usr/bin/env python3
"""
Test script for micro-batching performance verification.

Usage:
    python test_microbatch.py --help
    python test_microbatch.py --url http://localhost:8003 --concurrent 4
"""

import argparse
import asyncio
import aiohttp
import time
import json
from pathlib import Path
from typing import List, Tuple
import statistics


async def transcribe_request(
    session: aiohttp.ClientSession,
    url: str,
    audio_file: Path,
    request_id: int
) -> Tuple[int, float, dict]:
    """Send a single transcription request and measure latency."""
    start_time = time.time()

    try:
        with open(audio_file, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file',
                          f,
                          filename=audio_file.name,
                          content_type='audio/wav')

            async with session.post(f"{url}/transcribe", data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    latency = time.time() - start_time
                    return request_id, latency, result
                else:
                    error = await response.text()
                    latency = time.time() - start_time
                    return request_id, latency, {"error": error, "status": response.status}

    except Exception as e:
        latency = time.time() - start_time
        return request_id, latency, {"error": str(e)}


async def run_concurrent_test(
    url: str,
    audio_file: Path,
    num_concurrent: int
) -> dict:
    """Run concurrent transcription requests and measure performance."""

    print(f"\n{'='*80}")
    print(f"🧪 Testing with {num_concurrent} concurrent requests")
    print(f"{'='*80}\n")

    # Create client session
    timeout = aiohttp.ClientTimeout(total=600)  # 10 minute timeout
    async with aiohttp.ClientSession(timeout=timeout) as session:

        # Create concurrent tasks
        tasks = [
            transcribe_request(session, url, audio_file, i)
            for i in range(num_concurrent)
        ]

        # Start timer
        start_time = time.time()

        # Execute all requests concurrently
        print(f"⏱️  Starting {num_concurrent} concurrent requests...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate total time
        total_time = time.time() - start_time

        # Process results
        latencies = []
        successful = 0
        failed = 0

        for request_id, latency, result in results:
            if isinstance(result, Exception):
                print(f"❌ Request {request_id}: Exception - {result}")
                failed += 1
            elif "error" in result:
                print(f"❌ Request {request_id}: {result.get('error', 'Unknown error')}")
                failed += 1
            else:
                print(f"✅ Request {request_id}: {latency:.3f}s - {result.get('engine', 'unknown')} - \"{result.get('text', '')[:50]}...\"")
                latencies.append(latency)
                successful += 1

        # Calculate metrics
        if latencies:
            avg_latency = statistics.mean(latencies)
            p50_latency = statistics.median(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0]
            p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) > 2 else latencies[0]
            throughput = successful / total_time
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = throughput = 0

        # Print summary
        print(f"\n{'='*80}")
        print(f"📊 Results Summary")
        print(f"{'='*80}")
        print(f"Total time:        {total_time:.3f}s")
        print(f"Successful:        {successful}/{num_concurrent}")
        print(f"Failed:            {failed}/{num_concurrent}")
        print(f"")
        print(f"Latency (avg):     {avg_latency:.3f}s")
        print(f"Latency (p50):     {p50_latency:.3f}s")
        print(f"Latency (p95):     {p95_latency:.3f}s")
        print(f"Latency (p99):     {p99_latency:.3f}s")
        print(f"")
        print(f"Throughput:        {throughput:.2f} requests/sec")
        print(f"{'='*80}\n")

        return {
            "num_concurrent": num_concurrent,
            "total_time": total_time,
            "successful": successful,
            "failed": failed,
            "avg_latency": avg_latency,
            "p50_latency": p50_latency,
            "p95_latency": p95_latency,
            "p99_latency": p99_latency,
            "throughput": throughput,
        }


async def run_benchmark(
    url: str,
    audio_file: Path,
    concurrent_levels: List[int]
) -> None:
    """Run benchmark across multiple concurrency levels."""

    print(f"\n{'='*80}")
    print(f"🚀 Micro-Batching Performance Benchmark")
    print(f"{'='*80}")
    print(f"URL:        {url}")
    print(f"Audio file: {audio_file}")
    print(f"Levels:     {concurrent_levels}")
    print(f"{'='*80}\n")

    # Verify server is up
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"✅ Server is healthy")
                    print(f"   Models: {health.get('models', {})}")
                else:
                    print(f"❌ Server health check failed: {response.status}")
                    return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return

    # Run tests
    all_results = []
    for level in concurrent_levels:
        result = await run_concurrent_test(url, audio_file, level)
        all_results.append(result)

        # Brief pause between tests
        if level != concurrent_levels[-1]:
            print("⏸️  Pausing 2 seconds before next test...\n")
            await asyncio.sleep(2)

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"📈 Performance Comparison")
    print(f"{'='*80}\n")
    print(f"{'Concurrent':<12} {'Total Time':<12} {'Throughput':<15} {'Avg Latency':<12}")
    print(f"{'-'*12} {'-'*12} {'-'*15} {'-'*12}")

    for r in all_results:
        print(f"{r['num_concurrent']:<12} {r['total_time']:>10.2f}s {r['throughput']:>12.2f}/s {r['avg_latency']:>10.3f}s")

    print(f"\n{'='*80}\n")

    # Calculate speedup
    if len(all_results) > 1:
        baseline = all_results[0]
        print(f"🎯 Speedup Analysis (vs {baseline['num_concurrent']} concurrent):\n")

        for r in all_results[1:]:
            speedup = r['throughput'] / baseline['throughput']
            latency_increase = r['avg_latency'] - baseline['avg_latency']

            print(f"  {r['num_concurrent']} concurrent:")
            print(f"    - Throughput: {speedup:.2f}x faster")
            print(f"    - Latency: +{latency_increase*1000:.1f}ms")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Test micro-batching performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default settings (1, 2, 4, 8 concurrent)
  python test_microbatch.py --audio test_audio.wav

  # Test specific concurrency level
  python test_microbatch.py --audio test_audio.wav --concurrent 4

  # Test multiple levels
  python test_microbatch.py --audio test_audio.wav --levels 1 2 4 8 16

  # Custom server URL
  python test_microbatch.py --url http://192.168.1.100:8003 --audio test_audio.wav
        """
    )

    parser.add_argument(
        "--url",
        default="http://localhost:8003",
        help="Server URL (default: http://localhost:8003)"
    )

    parser.add_argument(
        "--audio",
        type=Path,
        help="Path to audio file for testing"
    )

    parser.add_argument(
        "--concurrent",
        type=int,
        help="Number of concurrent requests (single test)"
    )

    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Concurrency levels to test (default: 1 2 4 8)"
    )

    args = parser.parse_args()

    # Find audio file
    audio_file = args.audio

    if not audio_file:
        # Try to find test audio in common locations
        possible_files = [
            Path("test_audio.wav"),
            Path("test-audio/test.wav"),
            Path("audio.wav"),
        ]

        for f in possible_files:
            if f.exists():
                audio_file = f
                break

        if not audio_file:
            print("❌ No audio file specified and no test_audio.wav found")
            print("   Use --audio to specify an audio file")
            return

    if not audio_file.exists():
        print(f"❌ Audio file not found: {audio_file}")
        return

    # Determine concurrency levels
    if args.concurrent:
        levels = [args.concurrent]
    else:
        levels = args.levels

    # Run benchmark
    asyncio.run(run_benchmark(args.url, audio_file, levels))


if __name__ == "__main__":
    main()
