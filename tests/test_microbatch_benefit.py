#!/usr/bin/env python3
"""
Test Server-Side Micro-Batching Benefits

This test measures the ACTUAL benefit of server-side micro-batching by flooding
the server with concurrent requests that arrive within the batch timeout window.

WHAT THIS TEST MEASURES:
- Server-side GPU batching efficiency (batch_timeout_ms=15ms, max_batch_size=4)
- How well the server batches concurrent requests together
- Throughput improvement from processing multiple requests in a single GPU batch

HOW IT WORKS:
1. Send FLOOD of concurrent requests (e.g., 20 requests simultaneously)
2. Requests arrive within 15ms window → server batches them
3. Compare throughput at different concurrency levels
4. Higher concurrency should show better GPU utilization

USAGE:
    # Test with default concurrency levels (5, 10, 20, 40)
    python3 test_microbatch_benefit.py

    # Test with custom concurrency levels
    python3 test_microbatch_benefit.py --concurrency 10,20,30,40,50

    # Use custom server URL
    python3 test_microbatch_benefit.py --url http://localhost:8003/transcribe

    # Test specific audio file
    python3 test_microbatch_benefit.py --audio path/to/file.wav

EXPECTED RESULTS:
- Higher concurrency → More requests batched together → Better throughput
- Diminishing returns above max_batch_size (default: 4)
- Lower RTF per request with optimal concurrency

REQUIREMENTS:
    pip install aiohttp
"""

import argparse
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any
import statistics

try:
    import aiohttp
except ImportError:
    print("❌ Error: aiohttp is required. Install with: pip install aiohttp")
    exit(1)


class MicroBatchTester:
    """Test server-side micro-batching effectiveness."""

    def __init__(self, url: str, audio_file: Path, format: str = "text"):
        self.url = url
        self.audio_file = audio_file
        self.format = format

        # Validate audio file
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Read audio file once
        self.audio_data = audio_file.read_bytes()
        self.audio_size_mb = len(self.audio_data) / (1024 * 1024)

        print(f"📁 Audio file: {audio_file.name} ({self.audio_size_mb:.2f} MB)")

    async def check_connectivity(self) -> bool:
        """Check if server is accessible."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url.replace("/transcribe", "/health"), timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        print("✅ Server connectivity: OK")
                        return True
                    else:
                        print(f"⚠️  Server returned status {resp.status}")
                        return False
        except Exception as e:
            print(f"❌ Server connectivity check: {type(e).__name__}")
            print(f"   Error: {str(e)}")
            print(f"\n💡 Make sure server is running and accessible at: {self.url}")
            return False

    async def send_request(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Send single transcription request."""
        start_time = time.time()

        try:
            # Create multipart form data
            data = aiohttp.FormData()
            data.add_field('file',
                          self.audio_data,
                          filename=self.audio_file.name,
                          content_type='audio/wav')
            data.add_field('format', self.format)

            async with session.post(self.url, data=data) as resp:
                elapsed = time.time() - start_time

                if resp.status == 200:
                    result = await resp.json()
                    # Extract metrics from response
                    metrics = result.get("metrics", {})

                    return {
                        "status": "success",
                        "elapsed": elapsed,
                        "processing_time": metrics.get("total_processing_time_sec", 0),
                        "audio_duration": metrics.get("audio_duration_sec", 0),
                    }
                else:
                    error_text = await resp.text()
                    return {
                        "status": "error",
                        "elapsed": elapsed,
                        "error": f"HTTP {resp.status}: {error_text[:200]}"
                    }

        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "status": "exception",
                "elapsed": elapsed,
                "error": f"{type(e).__name__}: {str(e)}"
            }

    async def flood_test(self, concurrency: int, iterations: int = 3) -> Dict[str, Any]:
        """
        Flood server with concurrent requests to test micro-batching.

        Args:
            concurrency: Number of concurrent requests to send
            iterations: Number of times to repeat the test for averaging
        """
        print(f"\n{'='*80}")
        print(f"🌊 FLOOD TEST: {concurrency} concurrent requests")
        print(f"{'='*80}")

        all_results = []

        for iteration in range(iterations):
            print(f"\n📍 Iteration {iteration + 1}/{iterations}:")

            # Create session with connection pooling
            connector = aiohttp.TCPConnector(limit=concurrency * 2)
            timeout = aiohttp.ClientTimeout(total=300)  # 5 min timeout

            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                # Send all requests concurrently
                start_time = time.time()
                tasks = [self.send_request(session) for _ in range(concurrency)]
                results = await asyncio.gather(*tasks)
                total_elapsed = time.time() - start_time

                # Analyze results
                successes = [r for r in results if r["status"] == "success"]
                failures = [r for r in results if r["status"] != "success"]

                print(f"   Total time:     {total_elapsed:.2f}s")
                print(f"   Successes:      {len(successes)}/{concurrency}")
                print(f"   Failures:       {len(failures)}/{concurrency}")

                if successes:
                    avg_processing = statistics.mean(r["processing_time"] for r in successes)
                    avg_audio_duration = statistics.mean(r["audio_duration"] for r in successes)
                    avg_rtf = avg_processing / avg_audio_duration if avg_audio_duration > 0 else 0
                    throughput = len(successes) / total_elapsed
                    audio_minutes_per_sec = (avg_audio_duration / 60) * throughput

                    print(f"   Avg processing: {avg_processing:.2f}s")
                    print(f"   Avg RTF:        {avg_rtf:.3f}")
                    print(f"   Throughput:     {throughput:.2f} req/s")
                    print(f"   Audio capacity: {audio_minutes_per_sec:.2f} min/s")

                    all_results.append({
                        "iteration": iteration + 1,
                        "total_elapsed": total_elapsed,
                        "successes": len(successes),
                        "failures": len(failures),
                        "avg_processing_time": avg_processing,
                        "avg_audio_duration": avg_audio_duration,
                        "avg_rtf": avg_rtf,
                        "throughput": throughput,
                        "audio_minutes_per_sec": audio_minutes_per_sec,
                    })
                else:
                    print("   ❌ All requests failed")

        if not all_results:
            return None

        # Calculate averages across iterations
        return {
            "concurrency": concurrency,
            "iterations": iterations,
            "avg_total_elapsed": statistics.mean(r["total_elapsed"] for r in all_results),
            "avg_successes": statistics.mean(r["successes"] for r in all_results),
            "avg_failures": statistics.mean(r["failures"] for r in all_results),
            "avg_processing_time": statistics.mean(r["avg_processing_time"] for r in all_results),
            "avg_audio_duration": statistics.mean(r["avg_audio_duration"] for r in all_results),
            "avg_rtf": statistics.mean(r["avg_rtf"] for r in all_results),
            "avg_throughput": statistics.mean(r["throughput"] for r in all_results),
            "avg_audio_minutes_per_sec": statistics.mean(r["audio_minutes_per_sec"] for r in all_results),
            "all_iterations": all_results,
        }

    async def run_test_suite(self, concurrency_levels: List[int], iterations: int = 3) -> None:
        """Run complete test suite with different concurrency levels."""
        print("=" * 80)
        print("🔥 SERVER-SIDE MICRO-BATCHING BENEFIT TEST")
        print("=" * 80)
        print(f"URL:             {self.url}")
        print(f"Audio file:      {self.audio_file.name} ({self.audio_size_mb:.2f} MB)")
        print(f"Concurrency:     {concurrency_levels}")
        print(f"Iterations:      {iterations} per concurrency level")
        print(f"Format:          {self.format}")
        print("=" * 80)

        print("\n📝 WHAT THIS TEST MEASURES:")
        print("   Server-side GPU batching effectiveness with micro-batching enabled")
        print("   (batch_timeout_ms=15ms, max_batch_size=4)")
        print()
        print("   How it works:")
        print("   1. Flood server with N concurrent requests simultaneously")
        print("   2. Requests arrive within 15ms → server batches them together")
        print("   3. GPU processes multiple requests in single batch → better efficiency")
        print()
        print("   Expected results:")
        print("   • Higher concurrency → More requests batched → Better throughput")
        print("   • Diminishing returns above max_batch_size=4")
        print("   • Lower RTF with optimal concurrency")
        print("=" * 80)

        # Run tests for each concurrency level
        results = []
        for concurrency in concurrency_levels:
            result = await self.flood_test(concurrency, iterations)
            if result:
                results.append(result)

            # Brief pause between tests
            await asyncio.sleep(2)

        if not results:
            print("\n❌ No successful test results")
            return

        # Print summary comparison
        self.print_summary(results)

    def print_summary(self, results: List[Dict[str, Any]]) -> None:
        """Print summary comparison across concurrency levels."""
        print("\n" + "=" * 80)
        print("📊 SUMMARY: Micro-Batching Efficiency by Concurrency Level")
        print("=" * 80)

        print(f"\n{'Concurrency':<12} {'Throughput':>12} {'Avg RTF':>10} {'Audio Cap':>12} {'Total Time':>12} {'Success Rate':>13}")
        print(f"{'-'*12} {'-'*12} {'-'*10} {'-'*12} {'-'*12} {'-'*13}")

        baseline_throughput = results[0]["avg_throughput"]

        for r in results:
            concurrency = r["concurrency"]
            throughput = r["avg_throughput"]
            rtf = r["avg_rtf"]
            audio_capacity = r["avg_audio_minutes_per_sec"]
            elapsed = r["avg_total_elapsed"]
            success_rate = r["avg_successes"] / concurrency * 100

            throughput_improvement = ((throughput - baseline_throughput) / baseline_throughput * 100) if baseline_throughput > 0 else 0

            if throughput_improvement > 0:
                symbol = "✅"
            else:
                symbol = ""

            print(f"{concurrency:<12} {throughput:>10.2f}/s {rtf:>10.3f} {audio_capacity:>9.2f}min/s {elapsed:>10.2f}s {success_rate:>12.1f}% {symbol}")

        # Find optimal concurrency
        best_result = max(results, key=lambda r: r["avg_throughput"])
        print(f"\n💡 OPTIMAL CONCURRENCY: {best_result['concurrency']} concurrent requests")
        print(f"   Best throughput:  {best_result['avg_throughput']:.2f} req/s")
        print(f"   Best RTF:         {best_result['avg_rtf']:.3f}")
        print(f"   Audio capacity:   {best_result['avg_audio_minutes_per_sec']:.2f} min/s")

        # Calculate improvement vs baseline
        improvement = ((best_result["avg_throughput"] - baseline_throughput) / baseline_throughput * 100)
        print(f"   Improvement:      {improvement:.1f}% vs concurrency={results[0]['concurrency']}")

        print("\n📈 INTERPRETATION:")
        if improvement > 50:
            print("   ✅ EXCELLENT: Micro-batching is highly effective!")
            print("   Server is efficiently batching concurrent requests together.")
        elif improvement > 20:
            print("   ✅ GOOD: Micro-batching provides measurable benefit.")
            print("   Server is batching some concurrent requests together.")
        elif improvement > 0:
            print("   ⚠️  MODEST: Micro-batching provides small benefit.")
            print("   Consider checking if requests are arriving within 15ms window.")
        else:
            print("   ⚠️  MINIMAL: Little to no benefit from higher concurrency.")
            print("   This may indicate requests aren't arriving close enough to batch,")
            print("   or the server is bottlenecked on CPU preprocessing, not GPU.")

        print("\n💰 COST ANALYSIS:")
        print("   With micro-batching, you can serve more requests per GPU-hour:")

        # Calculate cost per 1000 requests at different pricing models
        gpu_costs = {
            "On-Demand": 0.8048,
            "Spot": 0.52,
            "1-yr Savings": 0.52,
            "3-yr Savings": 0.45,
        }

        print(f"\n   Cost per 1,000 requests at optimal concurrency ({best_result['concurrency']}):")
        throughput_per_hour = best_result['avg_throughput'] * 3600

        for model, hourly_cost in gpu_costs.items():
            cost_per_1k_requests = (hourly_cost / throughput_per_hour) * 1000
            print(f"   {model:<15} ${cost_per_1k_requests:>8.4f}")

        # Calculate cost per 1,000 minutes of audio
        audio_minutes_per_hour = best_result['avg_audio_minutes_per_sec'] * 3600

        print(f"\n   Cost per 1,000 minutes of audio at optimal concurrency ({best_result['concurrency']}):")
        for model, hourly_cost in gpu_costs.items():
            cost_per_1k_minutes = (hourly_cost / audio_minutes_per_hour) * 1000
            print(f"   {model:<15} ${cost_per_1k_minutes:>8.4f}")

        print("=" * 80)


def find_audio_file(test_dir: Path) -> Path:
    """Find first audio file in test directory."""
    for pattern in ["*.wav", "*.mp3", "*.m4a", "*.flac"]:
        files = list(test_dir.glob(pattern))
        if files:
            return files[0]
    raise FileNotFoundError(f"No audio files found in {test_dir}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test server-side micro-batching benefits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default concurrency levels (5, 10, 20, 40)
  python3 test_microbatch_benefit.py

  # Test with custom concurrency levels
  python3 test_microbatch_benefit.py --concurrency 10,20,30,40,50

  # Use custom server URL
  python3 test_microbatch_benefit.py --url http://localhost:8003/transcribe

  # Test specific audio file
  python3 test_microbatch_benefit.py --audio path/to/file.wav
        """
    )

    parser.add_argument(
        "--url",
        default="http://localhost:8000/transcribe",
        help="Server transcription endpoint URL (default: http://localhost:8000/transcribe)"
    )

    parser.add_argument(
        "--audio",
        type=Path,
        help="Path to audio file (default: auto-detect from test_audio/)"
    )

    parser.add_argument(
        "--concurrency",
        default="5,10,20,40",
        help="Comma-separated list of concurrency levels to test (default: 5,10,20,40)"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per concurrency level (default: 3)"
    )

    parser.add_argument(
        "--format",
        choices=["text", "json", "vtt", "srt"],
        default="text",
        help="Output format (default: text)"
    )

    args = parser.parse_args()

    # Parse concurrency levels
    try:
        concurrency_levels = [int(x.strip()) for x in args.concurrency.split(",")]
    except ValueError:
        print(f"❌ Error: Invalid concurrency levels: {args.concurrency}")
        print("   Use comma-separated integers, e.g., --concurrency 5,10,20,40")
        return

    # Find audio file
    if args.audio:
        audio_file = args.audio
    else:
        test_dir = Path(__file__).parent / "test_audio"
        if not test_dir.exists():
            print(f"❌ Error: Test directory not found: {test_dir}")
            print("\n💡 Create test_audio/ directory or specify --audio path")
            return

        try:
            audio_file = find_audio_file(test_dir)
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("\n💡 Add audio files to test_audio/ or specify --audio path")
            return

    # Create tester and run
    try:
        tester = MicroBatchTester(args.url, audio_file, args.format)

        # Check connectivity first
        if not await tester.check_connectivity():
            return

        # Run test suite
        await tester.run_test_suite(concurrency_levels, args.iterations)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
