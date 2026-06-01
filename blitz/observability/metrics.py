"""
Prometheus metrics for Blitz ASR.

Exposes custom metrics for:
- Request throughput and latency
- Model routing distribution (tier1/tier2/tier3)
- Audio processing characteristics
- GPU memory and utilization
- Language detection accuracy
- Error rates and types
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi import FastAPI, Response


class BlitzMetrics:
    """Container for all Blitz Prometheus metrics."""

    def __init__(self):
        # ====================================================================
        # Request Metrics
        # ====================================================================
        self.requests_total = Counter(
            "blitz_requests_total",
            "Total number of transcription requests",
            ["method", "endpoint", "status_code"],
        )

        self.requests_in_progress = Gauge(
            "blitz_requests_in_progress",
            "Number of requests currently being processed",
        )

        self.request_duration_seconds = Histogram(
            "blitz_request_duration_seconds",
            "End-to-end request duration in seconds",
            ["engine", "status"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
        )

        # ====================================================================
        # Transcription Metrics
        # ====================================================================
        self.transcriptions_total = Counter(
            "blitz_transcriptions_total",
            "Total number of successful transcriptions",
            ["engine", "language"],
        )

        self.transcription_errors_total = Counter(
            "blitz_transcription_errors_total",
            "Total number of transcription errors",
            ["engine", "error_type"],
        )

        self.audio_duration_seconds = Histogram(
            "blitz_audio_duration_seconds",
            "Duration of input audio files in seconds",
            ["engine"],
            buckets=[5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600],
        )

        self.transcription_time_seconds = Histogram(
            "blitz_transcription_time_seconds",
            "Time spent on transcription (excluding language detection)",
            ["engine"],
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
        )

        self.real_time_factor = Histogram(
            "blitz_real_time_factor",
            "Real-time factor (processing_time / audio_duration)",
            ["engine"],
            buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        )

        # ====================================================================
        # Model Routing Metrics
        # ====================================================================
        self.tier_routing_total = Counter(
            "blitz_tier_routing_total",
            "Number of requests routed to each tier",
            ["tier"],
        )

        self.tier_routing_duration_seconds = Histogram(
            "blitz_tier_routing_duration_seconds",
            "Processing time by tier",
            ["tier"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        )

        # ====================================================================
        # Language Detection Metrics
        # ====================================================================
        self.language_detection_total = Counter(
            "blitz_language_detection_total",
            "Language detection results",
            ["detected_language", "detection_method"],
        )

        self.language_detection_duration_ms = Histogram(
            "blitz_language_detection_duration_ms",
            "Language detection time in milliseconds",
            ["method"],
            buckets=[10, 25, 50, 100, 150, 200, 300, 500, 750, 1000],
        )

        self.language_detection_confidence = Histogram(
            "blitz_language_detection_confidence",
            "Language detection confidence scores",
            ["method"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        )

        # ====================================================================
        # VAD Metrics
        # ====================================================================
        self.vad_triggered_total = Counter(
            "blitz_vad_triggered_total",
            "Number of times VAD was triggered",
            ["result"],  # "speech", "silence", "skipped"
        )

        self.silence_detected_total = Counter(
            "blitz_silence_detected_total",
            "Number of silent audio files detected (GPU resources saved)",
        )

        # ====================================================================
        # GPU Metrics
        # ====================================================================
        self.gpu_memory_used_bytes = Gauge(
            "blitz_gpu_memory_used_bytes",
            "GPU memory currently in use (bytes)",
            ["gpu_id"],
        )

        self.gpu_memory_total_bytes = Gauge(
            "blitz_gpu_memory_total_bytes",
            "Total GPU memory available (bytes)",
            ["gpu_id"],
        )

        self.gpu_utilization_percent = Gauge(
            "blitz_gpu_utilization_percent",
            "GPU compute utilization percentage",
            ["gpu_id"],
        )

        self.gpu_temperature_celsius = Gauge(
            "blitz_gpu_temperature_celsius",
            "GPU temperature in Celsius",
            ["gpu_id"],
        )

        # ====================================================================
        # Batch Processing Metrics
        # ====================================================================
        self.batch_size = Histogram(
            "blitz_batch_size",
            "Number of requests processed per batch",
            ["tier"],
            buckets=[1, 2, 3, 4, 5, 6, 7, 8],
        )

        self.batch_wait_time_ms = Histogram(
            "blitz_batch_wait_time_ms",
            "Time requests wait in batch queue before processing",
            buckets=[1, 2, 5, 10, 15, 20, 30, 50, 100],
        )

        # ====================================================================
        # Cost Metrics
        # ====================================================================
        self.estimated_cost_usd = Counter(
            "blitz_estimated_cost_usd_total",
            "Estimated cumulative processing cost in USD (spot pricing)",
        )

        self.audio_minutes_processed = Counter(
            "blitz_audio_minutes_processed_total",
            "Total audio minutes processed",
            ["engine"],
        )

        # ====================================================================
        # System Info
        # ====================================================================
        self.system_info = Info(
            "blitz_system",
            "Blitz system information",
        )


# Singleton metrics instance
METRICS = BlitzMetrics()


def setup_metrics(app: FastAPI) -> None:
    """Setup Prometheus metrics endpoint on the FastAPI app."""

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint():
        """Prometheus metrics scraping endpoint."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    # Set system info
    METRICS.system_info.info(
        {
            "version": "1.0.0",
            "tier1_model": "nvidia/parakeet-tdt_ctc-110m",
            "tier2_model": "nvidia/parakeet-tdt-0.6b-v3",
            "tier3_model": "openai/whisper-large-v3-turbo",
        }
    )


def update_gpu_metrics():
    """Update GPU metrics from nvidia-smi. Call periodically."""
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 5:
                    gpu_id = parts[0]
                    mem_used_mb = float(parts[1])
                    mem_total_mb = float(parts[2])
                    util_pct = float(parts[3])
                    temp_c = float(parts[4])

                    METRICS.gpu_memory_used_bytes.labels(gpu_id=gpu_id).set(
                        mem_used_mb * 1024 * 1024
                    )
                    METRICS.gpu_memory_total_bytes.labels(gpu_id=gpu_id).set(
                        mem_total_mb * 1024 * 1024
                    )
                    METRICS.gpu_utilization_percent.labels(gpu_id=gpu_id).set(util_pct)
                    METRICS.gpu_temperature_celsius.labels(gpu_id=gpu_id).set(temp_c)
    except Exception:
        pass  # Non-critical — metrics just won't update
