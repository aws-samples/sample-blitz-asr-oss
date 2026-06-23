from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Response

class BlitzMetrics:
    def __init__(self):
        self.requests_total = Counter("blitz_requests_total", "Total requests", ["method", "endpoint", "status_code"])
        self.requests_in_progress = Gauge("blitz_requests_in_progress", "Requests in progress")
        self.transcriptions_total = Counter("blitz_transcriptions_total", "Successful transcriptions", ["engine", "language"])
        self.transcription_errors_total = Counter("blitz_transcription_errors_total", "Errors", ["engine", "error_type"])
        self.audio_duration_seconds = Histogram("blitz_audio_duration_seconds", "Audio duration", ["engine"], buckets=[5,10,30,60,120,300,600,1200,3600])
        self.transcription_time_seconds = Histogram("blitz_transcription_time_seconds", "Transcription time", ["engine"], buckets=[0.05,0.1,0.25,0.5,1,2,5,10,30,60])
        self.real_time_factor = Histogram("blitz_real_time_factor", "RTF", ["engine"], buckets=[0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0])
        self.tier_routing_total = Counter("blitz_tier_routing_total", "Routing per tier", ["tier"])
        self.language_detection_total = Counter("blitz_language_detection_total", "Detection results", ["detected_language", "detection_method"])
        self.language_detection_duration_ms = Histogram("blitz_language_detection_duration_ms", "Detection time ms", ["method"], buckets=[10,25,50,100,150,200,300,500,1000])
        self.gpu_memory_used_bytes = Gauge("blitz_gpu_memory_used_bytes", "GPU VRAM used", ["gpu_id"])
        self.gpu_memory_total_bytes = Gauge("blitz_gpu_memory_total_bytes", "GPU VRAM total", ["gpu_id"])
        self.gpu_utilization_percent = Gauge("blitz_gpu_utilization_percent", "GPU util %", ["gpu_id"])
        self.audio_minutes_processed = Counter("blitz_audio_minutes_processed_total", "Audio minutes", ["engine"])
        self.system_info = Info("blitz_system", "System info")

METRICS = BlitzMetrics()

def setup_metrics(app: FastAPI):
    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    METRICS.system_info.info({"version": "1.0.0", "tier1": "parakeet-ctc-110m", "tier2": "parakeet-0.6b", "tier3": "whisper-large-v3-turbo"})

def update_gpu_metrics():
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 4:
                    METRICS.gpu_memory_used_bytes.labels(gpu_id=parts[0]).set(float(parts[1])*1024*1024)
                    METRICS.gpu_memory_total_bytes.labels(gpu_id=parts[0]).set(float(parts[2])*1024*1024)
                    METRICS.gpu_utilization_percent.labels(gpu_id=parts[0]).set(float(parts[3]))
    except Exception:
        pass
