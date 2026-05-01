#!/usr/bin/env python3
"""
Loop-Based Load Testing Script for Blitz Transcription Server

Runs loops where each loop processes all files in the test directory.
Supports both loop-count and time-based modes for multi-hour stability testing.

Example:
    # Run 100 loops
    python test_load_loop.py --loops 100

    # Run for 60 minutes (1 hour)
    python test_load_loop.py --minutes 60

    # Run for 4 hours
    python test_load_loop.py --minutes 240
"""

import argparse
import asyncio
import logging
import signal
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class RequestResult:
    """Result from a single transcription request."""

    request_id: int
    loop_id: int
    filename: str
    status: str
    elapsed_sec: float
    audio_duration_sec: float = 0.0
    engine: str = ""
    language: str = ""
    lang_confidence: float = 0.0
    lang_model: str = ""
    text_length: int = 0
    rtf: float = 0.0
    error: str = ""
    # Cost per 1000 minutes
    cost_spot: float = 0.0
    cost_savings_1yr: float = 0.0


@dataclass
class LoopStats:
    """Statistics for a single loop iteration."""

    loop_id: int
    start_time: float
    end_time: float = 0.0
    success_count: int = 0
    error_count: int = 0
    total_audio_sec: float = 0.0
    total_elapsed_sec: float = 0.0

    @property
    def duration_sec(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0.0

    @property
    def throughput(self) -> float:
        total = self.success_count + self.error_count
        return total / self.duration_sec if self.duration_sec > 0 else 0.0

    @property
    def audio_capacity_min_per_sec(self) -> float:
        return (self.total_audio_sec / 60) / self.duration_sec if self.duration_sec > 0 else 0.0


@dataclass
class CumulativeStats:
    """Cumulative statistics across all loops."""

    results: list[RequestResult] = field(default_factory=list)
    loop_stats: list[LoopStats] = field(default_factory=list)
    start_time: float = 0.0

    # Per-engine tracking
    engine_results: dict[str, list[RequestResult]] = field(default_factory=dict)

    def add_result(self, result: RequestResult) -> None:
        self.results.append(result)
        if result.status == "success" and result.engine:
            if result.engine not in self.engine_results:
                self.engine_results[result.engine] = []
            self.engine_results[result.engine].append(result)

    @property
    def total_requests(self) -> int:
        return len(self.results)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.status == "success")

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.status != "success")

    @property
    def success_rate(self) -> float:
        return (self.success_count / self.total_requests * 100) if self.total_requests else 0.0

    @property
    def total_audio_sec(self) -> float:
        return sum(r.audio_duration_sec for r in self.results if r.status == "success")

    @property
    def elapsed_sec(self) -> float:
        return time.time() - self.start_time if self.start_time else 0.0


class LoopLoadTester:
    """Load tester that runs multiple loops over test files."""

    def __init__(
        self,
        url: str,
        test_dir: str,
        num_loops: Optional[int] = None,
        duration_minutes: Optional[float] = None,
        output_format: str = "json",
    ):
        self.url = url
        self.test_dir = Path(test_dir)
        self.num_loops = num_loops
        self.duration_minutes = duration_minutes
        self.duration_seconds = duration_minutes * 60 if duration_minutes else None
        self.output_format = output_format
        self.stats = CumulativeStats()
        self._shutdown_requested = False

        # Determine mode
        self.time_based_mode = duration_minutes is not None

        # Find audio files
        self.audio_files = self._find_audio_files()
        if not self.audio_files:
            raise ValueError(f"No audio files found in {self.test_dir}/")

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Optional[object]) -> None:
        """Handle shutdown signals gracefully."""
        logger.warning("Shutdown requested, finishing current loop...")
        self._shutdown_requested = True

    def _find_audio_files(self) -> list[Path]:
        """Find all audio files in test directory."""
        if not self.test_dir.exists():
            return []

        audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}
        files = []
        for ext in audio_extensions:
            files.extend(self.test_dir.glob(f"*{ext}"))
        return sorted(files)

    async def _send_request(
        self,
        session: aiohttp.ClientSession,
        audio_file: Path,
        request_id: int,
        loop_id: int,
    ) -> RequestResult:
        """Send a single transcription request."""
        start_time = time.time()

        try:
            data = aiohttp.FormData()
            data.add_field(
                "file",
                open(audio_file, "rb"),
                filename=audio_file.name,
                content_type="multipart/form-data",
            )
            data.add_field("format", self.output_format)

            async with session.post(
                self.url,
                data=data,
                headers={"accept": "application/json"},
            ) as response:
                elapsed = time.time() - start_time

                if response.status == 200:
                    resp_data = await response.json()

                    # Extract language info
                    lang_data = resp_data.get("language", {})
                    if isinstance(lang_data, dict):
                        language = lang_data.get("code", "unknown")
                        lang_confidence = lang_data.get("confidence", 0.0)
                        lang_model = lang_data.get("model", "")
                    else:
                        language = lang_data
                        lang_confidence = 0.0
                        lang_model = ""

                    # Extract metrics
                    metrics = resp_data.get("metrics", {})

                    # Extract cost per 1000 minutes
                    cost_data = metrics.get("cost_per_1000_min_audio_usd", {})
                    cost_spot = cost_data.get("spot", 0.0)
                    cost_savings_1yr = cost_data.get("savings_1yr", 0.0)

                    return RequestResult(
                        request_id=request_id,
                        loop_id=loop_id,
                        filename=audio_file.name,
                        status="success",
                        elapsed_sec=elapsed,
                        audio_duration_sec=resp_data.get("duration_sec", 0.0),
                        engine=resp_data.get("engine", "unknown"),
                        language=language,
                        lang_confidence=lang_confidence,
                        lang_model=lang_model,
                        text_length=len(resp_data.get("text", "")),
                        rtf=metrics.get("real_time_factor", 0.0),
                        cost_spot=cost_spot,
                        cost_savings_1yr=cost_savings_1yr,
                    )
                else:
                    error_text = await response.text()
                    return RequestResult(
                        request_id=request_id,
                        loop_id=loop_id,
                        filename=audio_file.name,
                        status="error",
                        elapsed_sec=elapsed,
                        error=f"HTTP {response.status}: {error_text[:100]}",
                    )

        except Exception as e:
            elapsed = time.time() - start_time
            return RequestResult(
                request_id=request_id,
                loop_id=loop_id,
                filename=audio_file.name,
                status="exception",
                elapsed_sec=elapsed,
                error=str(e)[:100],
            )

    async def _run_loop(
        self,
        session: aiohttp.ClientSession,
        loop_id: int,
    ) -> LoopStats:
        """Run a single loop iteration over all files."""
        loop_stats = LoopStats(loop_id=loop_id, start_time=time.time())

        for file_idx, audio_file in enumerate(self.audio_files):
            if self._shutdown_requested:
                break

            request_id = loop_id * len(self.audio_files) + file_idx
            result = await self._send_request(session, audio_file, request_id, loop_id)

            self.stats.add_result(result)

            if result.status == "success":
                loop_stats.success_count += 1
                loop_stats.total_audio_sec += result.audio_duration_sec
                loop_stats.total_elapsed_sec += result.elapsed_sec

                # Compact per-request output
                speedup = 1.0 / result.rtf if result.rtf > 0 else 0
                logger.info(
                    f"L{loop_id:03d} [{file_idx+1:02d}/{len(self.audio_files)}] "
                    f"{result.filename[:30]:30s} | "
                    f"{result.engine:12s} | "
                    f"{result.language:2s} | "
                    f"{result.audio_duration_sec:6.1f}s audio | "
                    f"{result.elapsed_sec:5.2f}s | "
                    f"{speedup:5.0f}x"
                )
            else:
                loop_stats.error_count += 1
                logger.error(
                    f"L{loop_id:03d} [{file_idx+1:02d}/{len(self.audio_files)}] "
                    f"{result.filename[:30]:30s} | FAILED: {result.error}"
                )

        loop_stats.end_time = time.time()
        self.stats.loop_stats.append(loop_stats)
        return loop_stats

    def _format_duration(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        return str(timedelta(seconds=int(seconds)))

    def _print_loop_summary(self, loop_stats: LoopStats, completed_loops: int) -> None:
        """Print summary after each loop."""
        elapsed = self.stats.elapsed_sec

        # Calculate ETA based on mode
        if self.time_based_mode:
            remaining_sec = max(0, self.duration_seconds - elapsed)
            progress_pct = min(100, elapsed / self.duration_seconds * 100)
            loop_display = f"Loop {completed_loops}"
            eta_display = self._format_duration(remaining_sec)
        else:
            avg_loop_time = elapsed / completed_loops if completed_loops > 0 else 0
            remaining_loops = self.num_loops - completed_loops
            remaining_sec = avg_loop_time * remaining_loops
            progress_pct = completed_loops / self.num_loops * 100
            loop_display = f"Loop {completed_loops}/{self.num_loops}"
            eta_display = self._format_duration(remaining_sec)

        logger.info("─" * 80)
        logger.info(
            f"{loop_display} complete | "
            f"Success: {loop_stats.success_count}/{loop_stats.success_count + loop_stats.error_count} | "
            f"Audio: {loop_stats.total_audio_sec/60:.1f}min | "
            f"Duration: {loop_stats.duration_sec:.1f}s | "
            f"Capacity: {loop_stats.audio_capacity_min_per_sec:.2f} min/s"
        )
        logger.info(
            f"Cumulative: {self.stats.success_count} success, {self.stats.error_count} errors | "
            f"Elapsed: {self._format_duration(elapsed)} | "
            f"Remaining: {eta_display} | "
            f"Progress: {progress_pct:.1f}%"
        )
        logger.info("─" * 80)

    def _print_final_report(self) -> None:
        """Print comprehensive final report."""
        elapsed = self.stats.elapsed_sec
        completed_loops = len(self.stats.loop_stats)

        print("\n" + "=" * 80)
        print("LOOP LOAD TEST COMPLETE")
        print("=" * 80)

        print(f"\nTest Configuration:")
        print(f"  URL:           {self.url}")
        print(f"  Test dir:      {self.test_dir}/")
        print(f"  Files/loop:    {len(self.audio_files)}")

        if self.time_based_mode:
            print(f"  Mode:          Time-based ({self.duration_minutes:.0f} min target)")
            print(f"  Loops run:     {completed_loops}")
        else:
            print(f"  Mode:          Loop-based")
            print(f"  Loops:         {completed_loops}/{self.num_loops}")

        print(f"  Total time:    {self._format_duration(elapsed)}")

        print(f"\nOverall Results:")
        print(f"  Total requests:  {self.stats.total_requests}")
        print(f"  Successful:      {self.stats.success_count} ({self.stats.success_rate:.1f}%)")
        print(f"  Failed:          {self.stats.error_count}")
        print(f"  Throughput:      {self.stats.total_requests / elapsed:.3f} req/s")
        print(f"  Audio processed: {self.stats.total_audio_sec / 60:.1f} min")
        print(f"  Audio capacity:  {(self.stats.total_audio_sec / 60) / elapsed:.2f} min/s")

        # Per-engine breakdown
        if self.stats.engine_results:
            print(f"\nPer-Engine Statistics:")
            print(f"  {'Engine':<15} {'Count':>8} {'Avg RTF':>10} {'Avg Time':>10} {'Audio (min)':>12}")
            print("  " + "-" * 60)

            for engine, results in sorted(self.stats.engine_results.items()):
                count = len(results)
                avg_rtf = statistics.mean(r.rtf for r in results) if results else 0
                avg_time = statistics.mean(r.elapsed_sec for r in results) if results else 0
                total_audio = sum(r.audio_duration_sec for r in results) / 60

                speedup = 1.0 / avg_rtf if avg_rtf > 0 else 0
                print(f"  {engine:<15} {count:>8} {speedup:>9.0f}x {avg_time:>9.2f}s {total_audio:>11.1f}")

        # Cost per 1000 minutes by engine
        if self.stats.engine_results:
            print(f"\nCost per 1,000 Minutes by Engine:")
            print(f"  {'Engine':<15} {'Languages':<12} {'Spot':>10} {'1-Yr Savings':>14}")
            print("  " + "-" * 55)

            # Engine to language mapping
            engine_langs = {
                "parakeet_ctc": "en",
                "parakeet": "24 EU langs",
                "whisper": "100+ langs",
            }

            for engine, results in sorted(self.stats.engine_results.items()):
                # Get costs from results that have cost data
                spot_costs = [r.cost_spot for r in results if r.cost_spot > 0]
                savings_costs = [r.cost_savings_1yr for r in results if r.cost_savings_1yr > 0]

                if spot_costs or savings_costs:
                    avg_spot = statistics.mean(spot_costs) if spot_costs else 0
                    avg_savings = statistics.mean(savings_costs) if savings_costs else 0
                    langs = engine_langs.get(engine, "unknown")

                    print(f"  {engine:<15} {langs:<12} ${avg_spot:>8.2f} ${avg_savings:>13.2f}")

            # Overall weighted average
            all_spot = [r.cost_spot for r in self.stats.results if r.status == "success" and r.cost_spot > 0]
            all_savings = [r.cost_savings_1yr for r in self.stats.results if r.status == "success" and r.cost_savings_1yr > 0]

            if all_spot or all_savings:
                print("  " + "-" * 55)
                avg_spot = statistics.mean(all_spot) if all_spot else 0
                avg_savings = statistics.mean(all_savings) if all_savings else 0
                print(f"  {'WEIGHTED AVG':<15} {'all':<12} ${avg_spot:>8.2f} ${avg_savings:>13.2f}")

        # Cost per 1000 minutes by language
        successful = [r for r in self.stats.results if r.status == "success"]
        if successful:
            # Group by language
            lang_results: dict[str, list[RequestResult]] = {}
            for r in successful:
                lang = r.language or "unknown"
                if lang not in lang_results:
                    lang_results[lang] = []
                lang_results[lang].append(r)

            if lang_results:
                print(f"\nCost per 1,000 Minutes by Language:")
                print(f"  {'Language':<10} {'Engine':<15} {'Count':>6} {'Audio (min)':>12} {'Spot':>10} {'1-Yr Savings':>14}")
                print("  " + "-" * 75)

                # Sort by audio duration descending
                sorted_langs = sorted(
                    lang_results.items(),
                    key=lambda x: sum(r.audio_duration_sec for r in x[1]),
                    reverse=True
                )

                for lang, results in sorted_langs:
                    count = len(results)
                    total_audio_min = sum(r.audio_duration_sec for r in results) / 60

                    # Get primary engine for this language
                    engine_counts: dict[str, int] = {}
                    for r in results:
                        engine_counts[r.engine] = engine_counts.get(r.engine, 0) + 1
                    primary_engine = max(engine_counts, key=engine_counts.get) if engine_counts else "unknown"

                    # Average costs
                    spot_costs = [r.cost_spot for r in results if r.cost_spot > 0]
                    savings_costs = [r.cost_savings_1yr for r in results if r.cost_savings_1yr > 0]
                    avg_spot = statistics.mean(spot_costs) if spot_costs else 0
                    avg_savings = statistics.mean(savings_costs) if savings_costs else 0

                    print(f"  {lang:<10} {primary_engine:<15} {count:>6} {total_audio_min:>11.1f} ${avg_spot:>9.2f} ${avg_savings:>13.2f}")

                print("  " + "-" * 75)

        # Language Detection Statistics
        successful = [r for r in self.stats.results if r.status == "success"]
        if successful:
            lang_results: dict[str, list[RequestResult]] = {}
            for r in successful:
                lang = r.language or "unknown"
                if lang not in lang_results:
                    lang_results[lang] = []
                lang_results[lang].append(r)

            if lang_results:
                print(f"\nLanguage Detection Statistics:")
                print(f"  {'Language':<10} {'Avg Confidence':>15} {'Detection Model':<25}")
                print("  " + "-" * 55)

                sorted_langs = sorted(
                    lang_results.items(),
                    key=lambda x: sum(r.audio_duration_sec for r in x[1]),
                    reverse=True,
                )

                for lang, results in sorted_langs:
                    # Average confidence
                    confidences = [r.lang_confidence for r in results if r.lang_confidence > 0]
                    avg_conf = statistics.mean(confidences) if confidences else 0.0

                    # Get most common model for this language
                    model_counts: dict[str, int] = {}
                    for r in results:
                        if r.lang_model:
                            model_counts[r.lang_model] = model_counts.get(r.lang_model, 0) + 1
                    primary_model = max(model_counts, key=model_counts.get) if model_counts else "N/A"

                    print(f"  {lang:<10} {avg_conf:>14.1%} {primary_model:<25}")

                print("  " + "-" * 55)

        # Low-Cost Transcriptions (at or below $0.06 per 1000 minutes)
        low_cost_threshold = 0.06
        successful = [r for r in self.stats.results if r.status == "success"]
        low_cost_files = [r for r in successful if r.cost_spot > 0 and r.cost_spot <= low_cost_threshold]

        if low_cost_files:
            # Deduplicate by filename (take first occurrence)
            seen_files: dict[str, RequestResult] = {}
            for r in low_cost_files:
                if r.filename not in seen_files:
                    seen_files[r.filename] = r

            print(f"\nLow-Cost Transcriptions (<= ${low_cost_threshold:.2f}/1000 min):")
            print(f"  {'Filename':<40} {'Duration':>10} {'Time':>8} {'Engine':<15} {'Cost':>10}")
            print("  " + "-" * 90)

            sorted_files = sorted(seen_files.values(), key=lambda r: r.cost_spot)
            for r in sorted_files:
                duration_str = f"{r.audio_duration_sec/60:.1f} min"
                time_str = f"{r.elapsed_sec:.2f}s"
                print(f"  {r.filename:<40} {duration_str:>10} {time_str:>8} {r.engine:<15} ${r.cost_spot:>8.4f}")

            print("  " + "-" * 90)
            print(f"  Total: {len(seen_files)} files qualify for low-cost transcription")

        # Latency statistics
        successful = [r for r in self.stats.results if r.status == "success"]
        if successful:
            latencies = [r.elapsed_sec for r in successful]
            print(f"\nLatency Statistics:")
            print(f"  Min:     {min(latencies):.2f}s")
            print(f"  Max:     {max(latencies):.2f}s")
            print(f"  Mean:    {statistics.mean(latencies):.2f}s")
            print(f"  Median:  {statistics.median(latencies):.2f}s")
            if len(latencies) > 1:
                print(f"  Std Dev: {statistics.stdev(latencies):.2f}s")
                # P95/P99
                sorted_latencies = sorted(latencies)
                p95_idx = int(len(sorted_latencies) * 0.95)
                p99_idx = int(len(sorted_latencies) * 0.99)
                print(f"  P95:     {sorted_latencies[p95_idx]:.2f}s")
                print(f"  P99:     {sorted_latencies[p99_idx]:.2f}s")

        # Loop-over-loop stability
        if len(self.stats.loop_stats) > 1:
            loop_throughputs = [ls.throughput for ls in self.stats.loop_stats]
            loop_capacities = [ls.audio_capacity_min_per_sec for ls in self.stats.loop_stats]

            print(f"\nLoop-over-Loop Stability:")
            print(f"  Throughput (req/s):")
            print(f"    Min: {min(loop_throughputs):.3f}, Max: {max(loop_throughputs):.3f}, "
                  f"Mean: {statistics.mean(loop_throughputs):.3f}")
            print(f"  Audio capacity (min/s):")
            print(f"    Min: {min(loop_capacities):.2f}, Max: {max(loop_capacities):.2f}, "
                  f"Mean: {statistics.mean(loop_capacities):.2f}")

        # Error summary
        errors = [r for r in self.stats.results if r.status != "success"]
        if errors:
            print(f"\nError Summary:")
            error_types: dict[str, int] = {}
            for e in errors:
                key = e.error[:50] if e.error else "Unknown"
                error_types[key] = error_types.get(key, 0) + 1

            for error_msg, count in sorted(error_types.items(), key=lambda x: -x[1])[:5]:
                print(f"  [{count:>3}x] {error_msg}")

        print("\n" + "=" * 80)

        # Final verdict
        if self.stats.success_rate >= 99:
            print("PASSED - Excellent stability")
        elif self.stats.success_rate >= 95:
            print("PASSED - Good stability")
        elif self.stats.success_rate >= 90:
            print("WARNING - Some failures detected")
        else:
            print("FAILED - High failure rate")

        print("=" * 80 + "\n")

    def _should_continue(self, loop_id: int) -> bool:
        """Check if we should continue running loops."""
        if self._shutdown_requested:
            return False

        if self.time_based_mode:
            elapsed = time.time() - self.stats.start_time
            return elapsed < self.duration_seconds
        else:
            return loop_id < self.num_loops

    async def run(self) -> None:
        """Run the complete loop load test."""
        print("=" * 80)
        print("LOOP LOAD TEST")
        print("=" * 80)
        print(f"URL:          {self.url}")
        print(f"Test dir:     {self.test_dir}/")
        print(f"Files:        {len(self.audio_files)}")
        for f in self.audio_files:
            print(f"  - {f.name}")

        if self.time_based_mode:
            print(f"Mode:         Time-based ({self.duration_minutes:.0f} minutes)")
            print(f"Files/loop:   {len(self.audio_files)}")
        else:
            print(f"Mode:         Loop-based ({self.num_loops} loops)")
            print(f"Total reqs:   {self.num_loops * len(self.audio_files)}")

        print(f"Started:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")

        self.stats.start_time = time.time()

        timeout = aiohttp.ClientTimeout(total=600)  # 10 minute timeout per request
        async with aiohttp.ClientSession(timeout=timeout) as session:
            loop_id = 0
            while self._should_continue(loop_id):
                loop_stats = await self._run_loop(session, loop_id)
                self._print_loop_summary(loop_stats, loop_id + 1)
                loop_id += 1

                if self._shutdown_requested:
                    logger.warning("Shutdown requested, stopping after current loop")
                    break

        self._print_final_report()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Loop-based load test for multi-hour stability testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 10 loops (quick test)
  python test_load_loop.py --loops 10

  # Run for 60 minutes (1 hour)
  python test_load_loop.py --minutes 60

  # Run for 4 hours
  python test_load_loop.py --minutes 240

  # Run 100 loops
  python test_load_loop.py --loops 100

  # Use custom test directory
  python test_load_loop.py --minutes 30 --test-dir audio_samples/

  # Custom server URL
  python test_load_loop.py --minutes 60 --url http://localhost:8000/transcribe

Note: Press Ctrl+C to gracefully stop after the current loop completes.
      Use --minutes OR --loops (if both specified, --minutes takes precedence).
        """,
    )

    parser.add_argument(
        "--loops",
        type=int,
        default=None,
        help="Number of loops to run (default: 10 if --minutes not specified)",
    )

    parser.add_argument(
        "--minutes",
        type=float,
        default=None,
        help="Run for specified number of minutes (takes precedence over --loops)",
    )

    parser.add_argument(
        "--test-dir",
        default="test",
        help="Directory containing test audio files (default: test/)",
    )

    parser.add_argument(
        "--url",
        default="http://localhost:8003/transcribe",
        help="Server URL (default: http://localhost:8003/transcribe)",
    )

    parser.add_argument(
        "-f", "--format",
        choices=["text", "json", "srt", "vtt"],
        default="json",
        help="Response format (default: json)",
    )

    args = parser.parse_args()

    # Determine mode: --minutes takes precedence, otherwise use --loops (default 10)
    duration_minutes = args.minutes
    num_loops = args.loops if args.loops is not None else (None if args.minutes else 10)

    try:
        tester = LoopLoadTester(
            url=args.url,
            test_dir=args.test_dir,
            num_loops=num_loops,
            duration_minutes=duration_minutes,
            output_format=args.format,
        )
    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nMake sure '{args.test_dir}/' exists and contains audio files.")
        print("Supported formats: .wav, .mp3, .flac, .m4a, .ogg, .opus")
        sys.exit(1)

    asyncio.run(tester.run())


if __name__ == "__main__":
    main()
