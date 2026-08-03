# Blitz ASR — Observability

Production observability for Blitz ASR: Prometheus metrics, structured JSON logging, and pre-built Grafana dashboards.

## Quick Start

### 1. Add Python dependency

Add to `requirements.txt`:
```
prometheus-client>=0.20.0
```

Or in the Dockerfile, add to the pip install block:
```dockerfile
RUN pip install --no-cache-dir prometheus-client>=0.20.0
```

### 2. Wire into the server

In `blitz/__init__.py` (inside `create_app()`) or in `blitz/server.py` after app creation:

```python
from blitz.observability.integration import integrate_observability

app = FastAPI(...)
integrate_observability(app)
```

That's it! This adds:
- `GET /metrics` — Prometheus scraping endpoint
- Structured JSON logging to stdout
- Request correlation IDs (`X-Request-ID` header)
- Automatic request duration and error tracking

### 3. Record transcription metrics

In the `/transcribe` endpoint, after a successful transcription:

```python
from blitz.observability.metrics import METRICS

# After successful transcription:
METRICS.transcriptions_total.labels(engine=engine, language=lang_code).inc()
METRICS.tier_routing_total.labels(tier=engine).inc()
METRICS.audio_duration_seconds.labels(engine=engine).observe(duration_sec)
METRICS.transcription_time_seconds.labels(engine=engine).observe(transcribe_time)
METRICS.real_time_factor.labels(engine=engine).observe(rtf)
METRICS.language_detection_duration_ms.labels(method=method).observe(lang_detect_ms)
METRICS.language_detection_confidence.labels(method=method).observe(confidence)
METRICS.audio_minutes_processed.labels(engine=engine).inc(duration_sec / 60.0)
```

### 4. Deploy monitoring stack (optional)

```bash
# Grafana admin password is required (stack won't start without it)
export GRAFANA_ADMIN_PASSWORD="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"

# Start Prometheus + Grafana alongside Blitz
docker compose -f docker-compose.yml \
  -f observability/docker-compose.observability.yml up -d

# Access (Grafana and Prometheus bind to localhost only):
# - Blitz API:    http://localhost:8000
# - Metrics:      http://localhost:8000/metrics
# - Prometheus:   http://localhost:9090
# - Grafana:      http://localhost:3000 (user: admin, or $GRAFANA_ADMIN_USER)
```

## What's Measured

### Request Metrics
| Metric | Type | Description |
|--------|------|-------------|
| `blitz_requests_total` | Counter | Total requests by endpoint/status |
| `blitz_requests_in_progress` | Gauge | Currently processing |
| `blitz_request_duration_seconds` | Histogram | End-to-end latency |

### Transcription Metrics
| Metric | Type | Description |
|--------|------|-------------|
| `blitz_transcriptions_total` | Counter | Successful transcriptions by engine/language |
| `blitz_transcription_errors_total` | Counter | Errors by engine/type |
| `blitz_audio_duration_seconds` | Histogram | Input audio length distribution |
| `blitz_transcription_time_seconds` | Histogram | GPU processing time |
| `blitz_real_time_factor` | Histogram | RTF distribution (lower = faster) |
| `blitz_audio_minutes_processed_total` | Counter | Cumulative audio processed |

### Model Routing
| Metric | Type | Description |
|--------|------|-------------|
| `blitz_tier_routing_total` | Counter | Requests per tier |
| `blitz_tier_routing_duration_seconds` | Histogram | Latency per tier |

### Language Detection
| Metric | Type | Description |
|--------|------|-------------|
| `blitz_language_detection_total` | Counter | Detections by language/method |
| `blitz_language_detection_duration_ms` | Histogram | Detection latency |
| `blitz_language_detection_confidence` | Histogram | Confidence distribution |

### GPU
| Metric | Type | Description |
|--------|------|-------------|
| `blitz_gpu_memory_used_bytes` | Gauge | Current VRAM usage |
| `blitz_gpu_memory_total_bytes` | Gauge | Total VRAM |
| `blitz_gpu_utilization_percent` | Gauge | Compute utilization |
| `blitz_gpu_temperature_celsius` | Gauge | Temperature |

### Batch Processing
| Metric | Type | Description |
|--------|------|-------------|
| `blitz_batch_size` | Histogram | Requests per micro-batch |
| `blitz_batch_wait_time_ms` | Histogram | Queue wait time |

## Structured Logging

Every log entry is JSON with these fields:
```json
{
  "timestamp": "2026-05-06T17:00:00.000000+00:00",
  "level": "INFO",
  "logger": "blitz.request",
  "message": "Request completed: 200 in 1.234s",
  "service": "blitz-asr",
  "instance_type": "g6.xlarge",
  "region": "us-east-1",
  "hostname": "blitz-abc123",
  "request_id": "a1b2c3d4",
  "method": "POST",
  "path": "/transcribe",
  "client_ip": "10.0.1.5",
  "processing_time_sec": 1.234
}
```

**CloudWatch integration**: JSON logs go to stdout → Docker captures them → CloudWatch Logs agent ingests. Use CloudWatch Insights to query:
```
fields @timestamp, request_id, engine, processing_time_sec
| filter path = "/transcribe"
| sort processing_time_sec desc
| limit 20
```

## Key Grafana Dashboard Panels

The pre-provisioned dashboard includes:
1. **Request Rate** — requests/sec over time
2. **Latency Percentiles** — P50, P95, P99
3. **Tier Distribution** — pie chart of tier1/tier2/tier3 usage
4. **Language Distribution** — top detected languages
5. **GPU Memory** — VRAM usage vs. total
6. **GPU Utilization** — compute % over time
7. **Error Rate** — errors/min with type breakdown
8. **Throughput** — audio minutes processed/hour
9. **Cost Tracker** — estimated $/hour based on audio volume

## Useful PromQL Queries

```promql
# Request rate (5-min window)
rate(blitz_requests_total{endpoint="/transcribe"}[5m])

# P99 latency
histogram_quantile(0.99, rate(blitz_request_duration_seconds_bucket{engine!=""}[5m]))

# Tier routing ratio
blitz_tier_routing_total / ignoring(tier) group_left sum(blitz_tier_routing_total)

# Audio throughput (minutes/hour)
rate(blitz_audio_minutes_processed_total[1h]) * 3600

# GPU memory utilization %
blitz_gpu_memory_used_bytes / blitz_gpu_memory_total_bytes * 100

# Error rate
rate(blitz_transcription_errors_total[5m])

# Average RTF by tier
rate(blitz_real_time_factor_sum[5m]) / rate(blitz_real_time_factor_count[5m])
```

## File Structure

```
blitz/observability/
├── __init__.py          # Module exports
├── metrics.py           # Prometheus metric definitions
├── logging_config.py    # Structured JSON logging
├── middleware.py        # FastAPI middleware (auto-instrumentation)
└── integration.py       # One-call setup + usage guide

observability/
├── README.md                          # This file
├── prometheus.yml                     # Prometheus scrape config
├── docker-compose.observability.yml   # Prometheus + Grafana stack
└── grafana/
    ├── provisioning/
    │   ├── datasources/prometheus.yml # Auto-connect to Prometheus
    │   └── dashboards/dashboard.yml   # Dashboard provisioning
    └── dashboards/                    # Pre-built dashboard JSONs
```
