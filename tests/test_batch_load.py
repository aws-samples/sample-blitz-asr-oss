#!/usr/bin/env python3
"""
Batch vs Sequential Load Testing for Blitz

Compares batch processing (concurrent requests) vs sequential processing
to demonstrate micro-batching benefits and cost savings.

Tests multiple batch sizes: 2, 4, 8, 10, 12 files concurrently.
"""

import argparse
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

import aiohttp


class BatchLoadTester:
    def __init__(
        self,
        url: str,
        test_dir: str = "test",
        format: str = "json"
    ):
        self.url = url
        self.test_dir = Path(test_dir)
        self.format = format

        # Find all audio files in test directory
        self.audio_files = self._find_audio_files()
        if not self.audio_files:
            raise ValueError(f"No audio files found in {self.test_dir}/")

        print(f"Found {len(self.audio_files)} audio files")

    def _find_audio_files(self) -> List[Path]:
        """Find all audio files in test directory."""
        if not self.test_dir.exists():
            return []

        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus'}
        files = []

        for ext in audio_extensions:
            files.extend(self.test_dir.glob(f"*{ext}"))

        return sorted(files)

    async def send_request(
        self,
        session: aiohttp.ClientSession,
        audio_file: Path,
        request_id: int
    ) -> Dict:
        """Send a single transcription request."""
        start_time = time.time()

        try:
            # Prepare form data
            data = aiohttp.FormData()
            data.add_field('file',
                          open(audio_file, 'rb'),
                          filename=audio_file.name,
                          content_type='multipart/form-data')
            data.add_field('format', self.format)

            # Send request
            headers = {'accept': 'application/json'}
            async with session.post(self.url, data=data, headers=headers) as response:
                elapsed = time.time() - start_time

                if response.status == 200:
                    result = await response.json()
                    return {
                        "request_id": request_id,
                        "status": "success",
                        "elapsed": elapsed,
                        "filename": audio_file.name,
                        "text_length": len(result.get("text", "")),
                        "response": result
                    }
                else:
                    error_text = await response.text()
                    return {
                        "request_id": request_id,
                        "status": "error",
                        "elapsed": elapsed,
                        "filename": audio_file.name,
                        "http_status": response.status,
                        "error": error_text
                    }

        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "request_id": request_id,
                "status": "exception",
                "elapsed": elapsed,
                "filename": audio_file.name,
                "error": str(e)
            }

    async def run_batch(
        self,
        session: aiohttp.ClientSession,
        files: List[Path],
        mode_name: str
    ) -> Tuple[List[Dict], float]:
        """
        Run batch of requests concurrently.

        Returns:
            Tuple of (results, total_elapsed_time)
        """
        print(f"\n🔥 Running {mode_name} ({len(files)} files concurrently)")

        batch_start = time.time()

        # Send all requests concurrently
        tasks = [
            self.send_request(session, file, i)
            for i, file in enumerate(files)
        ]

        results = await asyncio.gather(*tasks)

        batch_elapsed = time.time() - batch_start

        # Show individual results
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")
        exception_count = sum(1 for r in results if r["status"] == "exception")

        print(f"   ✅ Batch completed in {batch_elapsed:.2f}s")
        print(f"   Results: {success_count} success, {error_count} errors, {exception_count} exceptions")

        # Show first few errors for diagnosis
        errors = [r for r in results if r["status"] in ["error", "exception"]]
        if errors:
            print(f"\n   ⚠️  Sample errors (showing first 3):")
            for r in errors[:3]:
                error_msg = r.get("error", "Unknown error")
                print(f"      - {r['filename']}: {error_msg[:100]}")

        return results, batch_elapsed

    async def run_sequential(
        self,
        session: aiohttp.ClientSession,
        files: List[Path]
    ) -> Tuple[List[Dict], float]:
        """
        Run requests sequentially (one at a time).

        Returns:
            Tuple of (results, total_elapsed_time)
        """
        print(f"\n🐌 Running SEQUENTIAL ({len(files)} files one-by-one)")

        seq_start = time.time()
        results = []

        for i, file in enumerate(files):
            result = await self.send_request(session, file, i)
            results.append(result)

            # Show error details for failed requests
            if result['status'] == 'exception':
                error_msg = result.get('error', 'Unknown error')
                print(f"   [{i+1}/{len(files)}] {file.name}: {result['status']} ({result['elapsed']:.2f}s) - {error_msg[:80]}")
            elif result['status'] == 'error':
                error_msg = result.get('error', 'Unknown error')
                http_status = result.get('http_status', 'N/A')
                print(f"   [{i+1}/{len(files)}] {file.name}: HTTP {http_status} ({result['elapsed']:.2f}s) - {error_msg[:80]}")
            else:
                print(f"   [{i+1}/{len(files)}] {file.name}: {result['status']} ({result['elapsed']:.2f}s)")

        seq_elapsed = time.time() - seq_start

        print(f"   ✅ Sequential completed in {seq_elapsed:.2f}s")

        return results, seq_elapsed

    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregated metrics from results."""
        successful = [r for r in results if r["status"] == "success"]

        if not successful:
            return {
                "success_count": 0,
                "total_audio_duration": 0.0,
                "total_processing_time": 0.0,
                "avg_rtf": 0.0,
                "avg_cost_per_1000_min": None
            }

        # Collect metrics
        total_audio_duration = 0.0
        total_processing_time = 0.0
        costs_on_demand = []
        costs_spot = []
        costs_1yr = []
        costs_3yr = []

        for r in successful:
            response = r["response"]
            metrics = response.get("metrics", {})

            # Audio duration
            duration = response.get("duration_sec", 0.0)
            total_audio_duration += duration

            # Processing time
            proc_time = metrics.get("total_processing_time_sec", 0.0)
            total_processing_time += proc_time

            # Cost per 1000 minutes
            cost_per_1000 = metrics.get("cost_per_1000_min_audio_usd", {})
            if cost_per_1000:
                costs_on_demand.append(cost_per_1000.get("on_demand", 0.0))
                costs_spot.append(cost_per_1000.get("spot", 0.0))
                costs_1yr.append(cost_per_1000.get("savings_1yr", 0.0))
                costs_3yr.append(cost_per_1000.get("savings_3yr", 0.0))

        # Calculate averages
        avg_costs = None
        if costs_on_demand:
            avg_costs = {
                "on_demand": statistics.mean(costs_on_demand),
                "spot": statistics.mean(costs_spot),
                "savings_1yr": statistics.mean(costs_1yr),
                "savings_3yr": statistics.mean(costs_3yr)
            }

        return {
            "success_count": len(successful),
            "total_audio_duration": total_audio_duration,
            "total_processing_time": total_processing_time,
            "avg_rtf": total_processing_time / total_audio_duration if total_audio_duration > 0 else 0,
            "avg_cost_per_1000_min": avg_costs
        }

    def print_comparison(
        self,
        batch_size: int,
        batch_results: List[Dict],
        batch_time: float,
        seq_results: List[Dict],
        seq_time: float
    ) -> None:
        """Print detailed comparison between batch and sequential."""
        print("\n" + "=" * 80)
        print(f"📊 BATCH SIZE {batch_size} - COMPARISON REPORT")
        print("=" * 80)

        # Calculate metrics
        batch_metrics = self.calculate_metrics(batch_results)
        seq_metrics = self.calculate_metrics(seq_results)

        # Time comparison
        print(f"\n⏱️  TIME COMPARISON:")
        print(f"   Batch:       {batch_time:.2f}s")
        print(f"   Sequential:  {seq_time:.2f}s")
        speedup = seq_time / batch_time if batch_time > 0 else 0
        time_saved = seq_time - batch_time
        time_saved_pct = (time_saved / seq_time * 100) if seq_time > 0 else 0
        print(f"   Speedup:     {speedup:.2f}x faster")
        print(f"   Time saved:  {time_saved:.2f}s ({time_saved_pct:.1f}%)")

        # Throughput comparison
        batch_throughput = batch_size / batch_time if batch_time > 0 else 0
        seq_throughput = batch_size / seq_time if seq_time > 0 else 0
        print(f"\n🚀 THROUGHPUT:")
        print(f"   Batch:       {batch_throughput:.2f} req/s")
        print(f"   Sequential:  {seq_throughput:.2f} req/s")
        throughput_gain = ((batch_throughput - seq_throughput) / seq_throughput * 100) if seq_throughput > 0 else 0
        print(f"   Improvement: {throughput_gain:.1f}%")

        # Audio capacity comparison (minutes of audio per second)
        batch_audio_capacity = (batch_metrics['total_audio_duration'] / 60) / batch_time if batch_time > 0 else 0
        seq_audio_capacity = (seq_metrics['total_audio_duration'] / 60) / seq_time if seq_time > 0 else 0
        print(f"\n🎵 AUDIO CAPACITY:")
        print(f"   Batch:       {batch_audio_capacity:.2f} min/s")
        print(f"   Sequential:  {seq_audio_capacity:.2f} min/s")
        audio_capacity_gain = ((batch_audio_capacity - seq_audio_capacity) / seq_audio_capacity * 100) if seq_audio_capacity > 0 else 0
        print(f"   Improvement: {audio_capacity_gain:.1f}%")

        # RTF comparison
        print(f"\n📊 REAL-TIME FACTOR (RTF):")
        print(f"   Batch:       {batch_metrics['avg_rtf']:.3f}")
        print(f"   Sequential:  {seq_metrics['avg_rtf']:.3f}")
        rtf_improvement = seq_metrics['avg_rtf'] - batch_metrics['avg_rtf']
        rtf_pct = (rtf_improvement / seq_metrics['avg_rtf'] * 100) if seq_metrics['avg_rtf'] > 0 else 0

        if rtf_improvement > 0:
            print(f"   Batch better by:   {rtf_improvement:.3f} ({rtf_pct:.1f}% lower RTF) ✅")
        elif rtf_improvement < 0:
            print(f"   Batch WORSE by:    {abs(rtf_improvement):.3f} ({abs(rtf_pct):.1f}% higher RTF) ⚠️")
        else:
            print(f"   Same performance")

        # Cost comparison
        if batch_metrics['avg_cost_per_1000_min'] and seq_metrics['avg_cost_per_1000_min']:
            batch_costs = batch_metrics['avg_cost_per_1000_min']
            seq_costs = seq_metrics['avg_cost_per_1000_min']

            print(f"\n💰 COST PER 1,000 MINUTES (On-Demand):")
            print(f"   Batch:       ${batch_costs['on_demand']:.2f}")
            print(f"   Sequential:  ${seq_costs['on_demand']:.2f}")
            cost_savings = seq_costs['on_demand'] - batch_costs['on_demand']
            cost_savings_pct = (cost_savings / seq_costs['on_demand'] * 100) if seq_costs['on_demand'] > 0 else 0

            if cost_savings > 0:
                print(f"   Batch saves:     ${cost_savings:.2f} ({cost_savings_pct:.1f}% cheaper) ✅")
            elif cost_savings < 0:
                print(f"   Batch costs MORE: ${abs(cost_savings):.2f} ({abs(cost_savings_pct):.1f}% more expensive) ⚠️")
            else:
                print(f"   Same cost")

            print(f"\n💰 COST PER 1,000 MINUTES (All Pricing Models):")
            print(f"   {'Model':<15} {'Batch':>10} {'Sequential':>12} {'Result':>20}")
            print(f"   {'-'*15} {'-'*10} {'-'*12} {'-'*20}")

            for model in ['on_demand', 'spot', 'savings_1yr', 'savings_3yr']:
                batch_cost = batch_costs[model]
                seq_cost = seq_costs[model]
                savings = seq_cost - batch_cost
                savings_pct = (savings / seq_cost * 100) if seq_cost > 0 else 0

                model_label = {
                    'on_demand': 'On-Demand',
                    'spot': 'Spot',
                    'savings_1yr': '1-yr Savings',
                    'savings_3yr': '3-yr Savings'
                }[model]

                if savings > 0:
                    result = f"✅ Save ${savings:.2f} ({savings_pct:.1f}%)"
                elif savings < 0:
                    result = f"⚠️  +${abs(savings):.2f} ({abs(savings_pct):.1f}%)"
                else:
                    result = "Same"

                print(f"   {model_label:<15} ${batch_cost:>9.2f} ${seq_cost:>11.2f}  {result}")

        # Success rate
        print(f"\n✅ SUCCESS RATE:")
        batch_success_rate = (batch_metrics['success_count'] / batch_size * 100) if batch_size > 0 else 0
        seq_success_rate = (seq_metrics['success_count'] / batch_size * 100) if batch_size > 0 else 0
        print(f"   Batch:       {batch_metrics['success_count']}/{batch_size} ({batch_success_rate:.1f}%)")
        print(f"   Sequential:  {seq_metrics['success_count']}/{batch_size} ({seq_success_rate:.1f}%)")

    async def check_connectivity(self, session: aiohttp.ClientSession) -> bool:
        """Check if the server is reachable."""
        try:
            # Try to connect to the base URL
            base_url = self.url.rsplit('/', 1)[0] if '/' in self.url else self.url
            async with session.get(base_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                print(f"✅ Server connectivity check: OK (HTTP {response.status})")
                return True
        except asyncio.TimeoutError:
            print(f"❌ Server connectivity check: TIMEOUT (server not responding)")
            return False
        except aiohttp.ClientConnectorError as e:
            print(f"❌ Server connectivity check: CONNECTION REFUSED")
            print(f"   Error: {str(e)[:150]}")
            return False
        except Exception as e:
            print(f"⚠️  Server connectivity check: FAILED")
            print(f"   Error: {str(e)[:150]}")
            return False

    async def run_test_suite(self, batch_sizes: List[int]) -> None:
        """Run complete test suite with multiple batch sizes."""
        print("=" * 80)
        print("🔥 BATCH vs SEQUENTIAL LOAD TEST")
        print("=" * 80)
        print(f"URL:           {self.url}")
        print(f"Test dir:      {self.test_dir}/")
        print(f"Audio files:   {len(self.audio_files)} files found")
        print(f"Batch sizes:   {batch_sizes}")
        print(f"Format:        {self.format}")
        print("=" * 80)

        # Add explanatory context
        print("\n📝 WHAT THIS TEST MEASURES:")
        print("   This test compares HTTP request patterns:")
        print("   • BATCH MODE: Sends N concurrent HTTP requests simultaneously")
        print("   • SEQUENTIAL MODE: Sends requests one-at-a-time")
        print()
        print("   ⚠️  IMPORTANT: Both modes benefit from server-side micro-batching.")
        print("   The server uses async queuing (batch_timeout_ms=15ms, max_batch_size=4)")
        print("   to optimize processing regardless of how requests arrive.")
        print()
        print("   💡 If batch mode shows HIGHER costs/RTF, it means concurrent HTTP")
        print("   requests add overhead without additional processing benefit when")
        print("   the server already has micro-batching enabled.")
        print("=" * 80)

        # Show available files
        print(f"\n📁 Available files:")
        for i, f in enumerate(self.audio_files, 1):
            print(f"   [{i}] {f.name}")

        # Create session with timeout
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout

        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Check connectivity before running tests
            print(f"\n🔌 Checking server connectivity...")
            is_connected = await self.check_connectivity(session)
            if not is_connected:
                print(f"\n⚠️  WARNING: Server appears to be unreachable!")
                print(f"   Make sure the server is running on: {self.url}")
                print(f"   Continuing with tests anyway...\n")
            else:
                print()

            # Track all results for error summary
            all_results = []

            for batch_size in batch_sizes:
                # Skip if batch size exceeds available files
                if batch_size > len(self.audio_files):
                    print(f"\n⚠️  Skipping batch size {batch_size} (only {len(self.audio_files)} files available)")
                    continue

                # Select files for this batch
                test_files = self.audio_files[:batch_size]

                print("\n" + "=" * 80)
                print(f"🧪 TESTING BATCH SIZE: {batch_size}")
                print("=" * 80)
                print(f"Selected files:")
                for i, f in enumerate(test_files, 1):
                    print(f"   [{i}] {f.name}")

                # Run batch mode
                batch_results, batch_time = await self.run_batch(
                    session, test_files, f"BATCH-{batch_size}"
                )

                # Wait a bit between tests
                await asyncio.sleep(2)

                # Run sequential mode
                seq_results, seq_time = await self.run_sequential(
                    session, test_files
                )

                # Track results for error summary
                all_results.extend(batch_results)
                all_results.extend(seq_results)

                # Print comparison
                self.print_comparison(
                    batch_size, batch_results, batch_time,
                    seq_results, seq_time
                )

                # Wait between different batch sizes
                if batch_size != batch_sizes[-1]:
                    await asyncio.sleep(3)

            # Print error summary if there were any errors
            self.print_error_summary(all_results)

        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETE")
        print("=" * 80)

    def print_error_summary(self, all_results: List[Dict]) -> None:
        """Print summary of errors encountered during testing."""
        errors = [r for r in all_results if r["status"] in ["error", "exception"]]

        if not errors:
            return

        print("\n" + "=" * 80)
        print("⚠️  ERROR SUMMARY")
        print("=" * 80)

        # Group errors by type
        error_types = {}
        for r in errors:
            error_msg = r.get("error", "Unknown error")
            # Extract first line of error for grouping
            error_type = error_msg.split('\n')[0][:80]
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(r["filename"])

        print(f"\nTotal errors: {len(errors)}")
        print(f"Unique error types: {len(error_types)}\n")

        for error_type, files in error_types.items():
            print(f"❌ {error_type}")
            print(f"   Affected files ({len(files)}): {', '.join(files[:5])}")
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more")
            print()


async def main():
    parser = argparse.ArgumentParser(
        description="Batch vs Sequential Load Testing for Blitz"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/transcribe",
        help="Server URL (default: http://localhost:8000/transcribe)"
    )
    parser.add_argument(
        "--test-dir",
        default="tests/fixtures",
        help="Directory containing test audio files (default: tests/fixtures/)"
    )
    parser.add_argument(
        "--batch-sizes",
        default="2,4,8,10,12,16,18,20",
        help="Comma-separated list of batch sizes to test (default: 2,4,8,10,12,16,18,20)"
    )
    parser.add_argument(
        "--format",
        default="json",
        choices=["text", "json", "srt", "vtt"],
        help="Response format (default: json)"
    )

    args = parser.parse_args()

    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    # Create tester and run
    tester = BatchLoadTester(
        url=args.url,
        test_dir=args.test_dir,
        format=args.format
    )

    await tester.run_test_suite(batch_sizes)


if __name__ == "__main__":
    asyncio.run(main())
