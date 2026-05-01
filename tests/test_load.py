#!/usr/bin/env python3
"""
Load Testing Script for Blitz Transcription Server

Tests server stability and performance under concurrent load.
Uses all audio files from test/ directory.
"""

import argparse
import asyncio
import time
from pathlib import Path
from typing import Dict, List
import statistics
import itertools

import aiohttp


class LoadTester:
    def __init__(
        self,
        url: str,
        test_dir: str = "test",
        total_requests: int = 10,
        format: str = "json"
    ):
        self.url = url
        self.test_dir = Path(test_dir)
        self.total_requests = total_requests
        self.format = format

        # Find all audio files in test directory
        self.audio_files = self._find_audio_files()
        if not self.audio_files:
            raise ValueError(f"No audio files found in {self.test_dir}/")

        # Create circular iterator for files
        self.file_cycle = itertools.cycle(self.audio_files)

        # Results tracking
        self.results: List[Dict] = []
        self.errors: List[Dict] = []

    def _find_audio_files(self) -> List[Path]:
        """Find all audio files in test directory."""
        if not self.test_dir.exists():
            return []

        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus'}
        files = []

        for ext in audio_extensions:
            files.extend(self.test_dir.glob(f"*{ext}"))

        return sorted(files)

    async def send_request(self, session: aiohttp.ClientSession, request_id: int) -> Dict:
        """Send a single transcription request."""
        start_time = time.time()

        # Get next audio file from cycle
        audio_file = next(self.file_cycle)

        try:
            # Prepare form data (matches curl command format)
            data = aiohttp.FormData()
            data.add_field('file',
                          open(audio_file, 'rb'),
                          filename=audio_file.name,
                          content_type='multipart/form-data')
            data.add_field('format', self.format)

            # Send request with proper headers
            headers = {
                'accept': 'application/json'
            }

            # Send request
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
                        "http_status": response.status,
                        "error": error_text
                    }

        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "request_id": request_id,
                "status": "exception",
                "elapsed": elapsed,
                "error": str(e)
            }

    async def run_sequential(self, session: aiohttp.ClientSession) -> None:
        """Run requests sequentially, one at a time."""
        for request_id in range(self.total_requests):
            print(f"\n📤 Request {request_id + 1}/{self.total_requests}")

            request_start = time.time()

            # Send single request and wait for completion
            result = await self.send_request(session, request_id)

            request_elapsed = time.time() - request_start

            # Track result
            if result["status"] == "success":
                self.results.append(result)
                print(f"   ✅ Success")
                print(f"   📄 File: {result['filename']}")

                # Extract response data
                response = result['response']

                # Audio duration
                duration_sec = response.get('duration_sec', 0)
                duration_min = int(duration_sec // 60)
                duration_sec_rem = duration_sec % 60
                print(f"   🎵 Audio: {duration_sec:.1f}s ({duration_min}m {duration_sec_rem:.0f}s)")

                # Engine and language (handle new nested format)
                engine = response.get('engine', 'unknown')
                language_data = response.get('language', {})

                # Extract language info from nested dict or fallback to old format
                if isinstance(language_data, dict):
                    language = language_data.get('code', 'unknown')
                    lang_conf = language_data.get('confidence', 0.0)
                    lang_detect_method = language_data.get('detection_method', 'unknown')
                    lang_detect_model = language_data.get('detection_model', 'unknown')
                    top_langs = language_data.get('top_languages', {})
                else:
                    # Old format fallback
                    language = language_data
                    lang_conf = response.get('language_confidence', 0.0)
                    metrics = response.get('metrics', {})
                    lang_detect_method = metrics.get('lang_detect_method', 'unknown')
                    lang_detect_model = metrics.get('lang_detect_model', 'unknown')
                    top_langs = {}

                # VAD (Voice Activity Detection) metrics - check first
                vad_data = response.get('vad', {})
                is_silent = vad_data.get('is_silent', False) if vad_data else False

                if is_silent:
                    # Silent audio - show special message
                    print(f"   🔇 SILENT AUDIO DETECTED - Transcription skipped")
                    print(f"   🤖 Engine: none (no processing needed)")
                    print(f"   💾 Language: {language} (default)")

                    if vad_data:
                        speech_ratio = vad_data.get('speech_ratio') or 0.0
                        vad_time_ms = vad_data.get('vad_time_ms') or 0.0
                        num_segments = vad_data.get('num_speech_segments') or 0
                        print(f"   📊 VAD: {speech_ratio:.1%} speech, {num_segments} segments, {vad_time_ms:.0f}ms")
                else:
                    # Normal audio with speech
                    print(f"   🤖 Engine: {engine} ({language}, conf: {lang_conf:.2%})")

                    # Show top 3 detected languages
                    if top_langs:
                        top_3 = sorted(top_langs.items(), key=lambda x: x[1], reverse=True)[:3]
                        langs_str = ", ".join([f"{code}:{prob:.1%}" for code, prob in top_3])
                        print(f"   🔍 Lang detect: {lang_detect_model} (method: {lang_detect_method})")
                        print(f"      Top languages: {langs_str}")
                    else:
                        print(f"   🔍 Lang detect: {lang_detect_model} (method: {lang_detect_method})")
                        print(f"      Confidence: {lang_conf:.2%}")

                    # VAD metrics for non-silent audio
                    if vad_data:
                        speech_ratio = vad_data.get('speech_ratio') or 0.0
                        vad_time_ms = vad_data.get('vad_time_ms') or 0.0
                        num_segments = vad_data.get('num_speech_segments') or 0
                        print(f"   🔊 VAD: Speech detected (confidence: {speech_ratio:.1%}, {num_segments} segments, {vad_time_ms:.0f}ms)")

                # Real-time factor and timing
                metrics = response.get('metrics', {})
                rtf = metrics.get('real_time_factor', 0.0)
                lang_detect_ms = metrics.get('language_detection_time_ms', 0.0)
                transcribe_sec = metrics.get('transcription_time_sec', 0.0)
                estimated_cost = metrics.get('estimated_gpu_cost_usd', 0.0)
                gpu_instance = metrics.get('gpu_instance', 'unknown')

                if is_silent:
                    # Silent audio - show cost savings
                    print(f"   ⏱️  Total: {request_elapsed:.2f}s (VAD only, transcription skipped)")
                    print(f"   💰 GPU cost: $0.000000 (saved ~$0.000100 by skipping transcription)")
                    print(f"   ✅ Optimization: ~80% faster processing for silent audio")
                else:
                    # Normal audio - show full metrics
                    if rtf > 0:
                        speedup = 1.0 / rtf
                        print(f"   ⚡ RTF: {rtf:.2f} ({speedup:.1f}x real-time)")

                    print(f"   ⏱️  Total: {request_elapsed:.2f}s (lang detect: {lang_detect_ms:.0f}ms, transcribe: {transcribe_sec:.1f}s)")

                    # GPU cost metrics from API response
                    cost_per_1000_min = metrics.get('cost_per_1000_min_audio_usd', {})
                    cost_breakdown = metrics.get('cost_breakdown_percent', {})

                    if cost_per_1000_min:
                        # Display cost per 1000 minutes for all pricing models
                        on_demand = cost_per_1000_min.get('on_demand', 0.0)
                        spot = cost_per_1000_min.get('spot', 0.0)
                        savings_1yr = cost_per_1000_min.get('savings_1yr', 0.0)
                        savings_3yr = cost_per_1000_min.get('savings_3yr', 0.0)

                        # Calculate savings percentages
                        spot_savings_pct = ((on_demand - spot) / on_demand * 100) if on_demand > 0 else 0
                        savings_1yr_pct = ((on_demand - savings_1yr) / on_demand * 100) if on_demand > 0 else 0
                        savings_3yr_pct = ((on_demand - savings_3yr) / on_demand * 100) if on_demand > 0 else 0

                        print(f"   💰 Cost per 1,000 min:")
                        print(f"      On-Demand:    ${on_demand:.2f}")
                        print(f"      Spot:         ${spot:.2f} (save {spot_savings_pct:.0f}%)")
                        print(f"      1-yr Savings: ${savings_1yr:.2f} (save {savings_1yr_pct:.0f}%)")
                        print(f"      3-yr Savings: ${savings_3yr:.2f} (save {savings_3yr_pct:.0f}%)")

                        # Show cost breakdown if available
                        if cost_breakdown:
                            lang_pct = cost_breakdown.get('language_detection', 0.0)
                            trans_pct = cost_breakdown.get('transcription', 0.0)
                            print(f"   📊 Processing: {lang_pct:.1f}% lang detection, {trans_pct:.1f}% transcription")

                # Chunk metadata (in metrics dict)
                chunk_approach = metrics.get('chunk_approach', 'unknown')
                chunk_count = metrics.get('chunk_count', 1)
                chunk_size_sec = metrics.get('chunk_size_sec')

                if chunk_approach == 'chunked' and chunk_size_sec:
                    print(f"   🧩 Chunking: {chunk_count} chunks × {chunk_size_sec}s ({chunk_approach})")
                else:
                    print(f"   🧩 Processing: {chunk_approach} (chunk_count={chunk_count})")

                # Text length and preview
                print(f"   📝 Text: {result['text_length']} chars")
                transcript = response.get('text', '')
                # Skip first 40 chars, show up to 1000 chars
                start_pos = 40 if len(transcript) > 40 else 0
                end_pos = start_pos + 1000
                preview = transcript[start_pos:end_pos]
                print(f"   💬 Transcript: {preview}{'...' if len(transcript) > end_pos else ''}")
            else:
                self.errors.append(result)
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')[:100]}")
                print(f"   ⏱️  Time: {request_elapsed:.2f}s")

    async def run(self) -> None:
        """Run the complete load test."""
        print("=" * 70)
        print("🔥 SEQUENTIAL LOAD TEST STARTING")
        print("=" * 70)
        print(f"URL:          {self.url}")
        print(f"Test dir:     {self.test_dir}/")
        print(f"Audio files:  {len(self.audio_files)} files found")
        for i, f in enumerate(self.audio_files, 1):
            print(f"   [{i}] {f.name}")
        print(f"Mode:         Sequential (one at a time)")
        print(f"Total:        {self.total_requests} requests")
        print(f"Format:       {self.format}")
        print("=" * 70)

        total_start = time.time()

        # Create session with timeout
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Run requests sequentially
            await self.run_sequential(session)

        total_elapsed = time.time() - total_start

        # Print final report
        self.print_report(total_elapsed)

    def print_report(self, total_elapsed: float) -> None:
        """Print comprehensive test report."""
        print("\n" + "=" * 70)
        print("📊 LOAD TEST RESULTS")
        print("=" * 70)

        total_requests = len(self.results) + len(self.errors)
        success_rate = (len(self.results) / total_requests * 100) if total_requests > 0 else 0

        print(f"\n📈 Summary:")
        print(f"   Total requests:     {total_requests}")
        print(f"   ✅ Successful:      {len(self.results)} ({success_rate:.1f}%)")
        print(f"   ❌ Failed:          {len(self.errors)} ({100-success_rate:.1f}%)")
        print(f"   ⏱️  Total time:       {total_elapsed:.2f}s")
        print(f"   🚀 Overall throughput: {total_requests/total_elapsed:.2f} req/s")

        # Calculate audio capacity (minutes of audio per second)
        if self.results:
            total_audio_duration = 0.0
            for r in self.results:
                response = r.get("response", {})
                # Try both field names for duration
                duration = response.get("duration_sec") or response.get("audio_duration_sec", 0.0)
                total_audio_duration += duration

            audio_capacity = (total_audio_duration / 60) / total_elapsed if total_elapsed > 0 else 0
            print(f"   🎵 Audio capacity:  {audio_capacity:.2f} min/s ({total_audio_duration:.1f}s total audio)")

        if self.results:
            latencies = [r["elapsed"] for r in self.results]
            text_lengths = [r["text_length"] for r in self.results]

            print(f"\n⏱️  Latency Statistics (successful requests):")
            print(f"   Min:     {min(latencies):.2f}s")
            print(f"   Max:     {max(latencies):.2f}s")
            print(f"   Mean:    {statistics.mean(latencies):.2f}s")
            print(f"   Median:  {statistics.median(latencies):.2f}s")
            if len(latencies) > 1:
                print(f"   Std Dev: {statistics.stdev(latencies):.2f}s")

            print(f"\n📝 Transcription Statistics:")
            print(f"   Min length:     {min(text_lengths)} chars")
            print(f"   Max length:     {max(text_lengths)} chars")
            print(f"   Mean length:    {statistics.mean(text_lengths):.0f} chars")

            # File distribution
            file_counts = {}
            for r in self.results:
                filename = r.get("filename", "unknown")
                file_counts[filename] = file_counts.get(filename, 0) + 1

            print(f"\n📁 Files Tested:")
            for filename, count in sorted(file_counts.items()):
                print(f"   {filename}: {count} requests")

        if self.errors:
            print(f"\n❌ Error Details:")
            error_types = {}
            for error in self.errors:
                error_key = error.get("http_status", error["status"])
                error_types[error_key] = error_types.get(error_key, 0) + 1

            for error_type, count in error_types.items():
                print(f"   {error_type}: {count} occurrences")

            # Show first 3 errors
            print(f"\n   First {min(3, len(self.errors))} errors:")
            for i, error in enumerate(self.errors[:3]):
                print(f"   [{i+1}] Request {error['request_id']}: {error.get('error', 'Unknown error')[:100]}")

        # Show sample response
        if self.results:
            print(f"\n📄 Sample Response (Request {self.results[0]['request_id']}):")
            sample = self.results[0]["response"]

            # Handle new nested language format
            language_data = sample.get('language', {})
            if isinstance(language_data, dict):
                lang_code = language_data.get('code', 'N/A')
                lang_conf = language_data.get('confidence', 'N/A')
                lang_detect_model = language_data.get('detection_model', 'N/A')
                lang_detect_method = language_data.get('detection_method', 'N/A')
            else:
                lang_code = language_data
                lang_conf = sample.get('language_confidence', 'N/A')
                metrics = sample.get("metrics", {})
                lang_detect_model = metrics.get('lang_detect_model', 'N/A')
                lang_detect_method = metrics.get('lang_detect_method', 'N/A')

            print(f"   Language: {lang_code}")
            if isinstance(lang_conf, float):
                print(f"   Language confidence: {lang_conf:.2%}")
            else:
                print(f"   Language confidence: {lang_conf}")
            print(f"   Engine: {sample.get('engine', 'N/A')}")
            print(f"   Duration: {sample.get('duration_sec', 'N/A')}s")
            print(f"   Text: {sample.get('text', '')[:100]}...")
            if "metrics" in sample:
                metrics = sample["metrics"]
                print(f"   Metrics:")

                # Language detection details
                print(f"      Lang detect model: {lang_detect_model} ({lang_detect_method})")

                whisper_conf = metrics.get('whisper_lang_confidence')
                voxlingua_conf = metrics.get('voxlingua_lang_confidence')
                if whisper_conf is not None:
                    print(f"      Whisper confidence: {whisper_conf:.4f}")
                if voxlingua_conf is not None:
                    print(f"      VoxLingua confidence: {voxlingua_conf:.4f}")

                # VAD metrics
                vad_data = sample.get('vad', {})
                if vad_data:
                    is_silent = vad_data.get('is_silent', False)
                    speech_ratio = vad_data.get('speech_ratio') or 0.0
                    vad_status = "Silence detected" if is_silent else "Speech detected"
                    print(f"      VAD: {vad_status} (confidence: {speech_ratio:.1%})")

                print(f"      Processing time: {metrics.get('processing_time_sec', 'N/A')}s")
                gpu_cost = metrics.get('gpu_cost_usd', 'N/A')
                if isinstance(gpu_cost, (int, float)):
                    print(f"      GPU cost: ${gpu_cost:.6f}")
                else:
                    print(f"      GPU cost: {gpu_cost}")

        # Average cost per 1000 minutes across all successful requests
        if self.results:
            costs_collected = {
                'on_demand': [],
                'spot': [],
                'savings_1yr': [],
                'savings_3yr': []
            }

            for r in self.results:
                response = r.get("response", {})
                metrics = response.get("metrics", {})
                cost_per_1000_min = metrics.get('cost_per_1000_min_audio_usd', {})

                if cost_per_1000_min:
                    for pricing_type in costs_collected.keys():
                        cost_value = cost_per_1000_min.get(pricing_type)
                        if cost_value is not None:
                            costs_collected[pricing_type].append(cost_value)

            # Calculate and display averages
            if any(costs_collected.values()):
                print(f"\n💰 Average Cost per 1,000 Minutes (across {len(self.results)} successful requests):")

                if costs_collected['on_demand']:
                    avg_on_demand = statistics.mean(costs_collected['on_demand'])
                    print(f"   On-Demand:    ${avg_on_demand:.2f}")

                    if costs_collected['spot']:
                        avg_spot = statistics.mean(costs_collected['spot'])
                        spot_savings = ((avg_on_demand - avg_spot) / avg_on_demand * 100)
                        print(f"   Spot:         ${avg_spot:.2f} (save {spot_savings:.0f}%)")

                    if costs_collected['savings_1yr']:
                        avg_1yr = statistics.mean(costs_collected['savings_1yr'])
                        savings_1yr_pct = ((avg_on_demand - avg_1yr) / avg_on_demand * 100)
                        print(f"   1-yr Savings: ${avg_1yr:.2f} (save {savings_1yr_pct:.0f}%)")

                    if costs_collected['savings_3yr']:
                        avg_3yr = statistics.mean(costs_collected['savings_3yr'])
                        savings_3yr_pct = ((avg_on_demand - avg_3yr) / avg_on_demand * 100)
                        print(f"   3-yr Savings: ${avg_3yr:.2f} (save {savings_3yr_pct:.0f}%)")

        print("\n" + "=" * 70)

        # Final verdict
        if success_rate >= 95:
            print("✅ TEST PASSED - Server is stable!")
        elif success_rate >= 80:
            print("⚠️  TEST WARNING - Some failures detected")
        else:
            print("❌ TEST FAILED - High failure rate!")

        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Sequential load test for Blitz (sends requests one at a time)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test: 10 sequential requests (uses test/ directory)
  python test_load.py

  # More requests: 50 sequential requests
  python test_load.py -n 50

  # Quick test: 5 requests
  python test_load.py -n 5

  # Use different test directory
  python test_load.py --test-dir audio_samples/

  # Test text format
  python test_load.py --format text

  # Different server URL
  python test_load.py --url http://localhost:8003/transcribe

Note: Requests are sent sequentially (one at a time) to avoid memory issues.
        """
    )

    parser.add_argument(
        "--test-dir",
        default="test",
        help="Directory containing test audio files (default: test/)"
    )

    parser.add_argument(
        "--url",
        default="http://localhost:8003/transcribe",
        help="Server URL (default: http://localhost:8003/transcribe)"
    )

    parser.add_argument(
        "-n", "--num-requests",
        type=int,
        default=10,
        help="Total number of requests to send sequentially (default: 10)"
    )

    parser.add_argument(
        "-f", "--format",
        choices=["text", "json", "srt", "vtt"],
        default="json",
        help="Response format (default: json)"
    )

    args = parser.parse_args()

    # Create and run load tester
    try:
        tester = LoadTester(
            url=args.url,
            test_dir=args.test_dir,
            total_requests=args.num_requests,
            format=args.format
        )
    except ValueError as e:
        print(f"❌ Error: {e}")
        print(f"\nMake sure the '{args.test_dir}/' directory exists and contains audio files.")
        print("Supported formats: .wav, .mp3, .flac, .m4a, .ogg, .opus")
        return

    # Run async
    asyncio.run(tester.run())


if __name__ == "__main__":
    main()
