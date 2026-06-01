"""
Integration guide for wiring observability into the existing Blitz server.

Usage:
    In blitz/server.py (or blitz/__init__.py where create_app() is defined),
    add the following after app creation:

    ```python
    from blitz.observability import setup_metrics, setup_logging, ObservabilityMiddleware, METRICS

    # Setup structured logging (call before anything else)
    setup_logging(log_level="INFO", json_format=True)

    # Add observability middleware to app
    app.add_middleware(ObservabilityMiddleware)

    # Setup /metrics endpoint
    setup_metrics(app)
    ```

    Then in the /transcribe endpoint, record metrics after each successful transcription:

    ```python
    from blitz.observability.metrics import METRICS

    # After successful transcription:
    METRICS.transcriptions_total.labels(engine=engine, language=best_lang_code).inc()
    METRICS.tier_routing_total.labels(tier=engine).inc()
    METRICS.audio_duration_seconds.labels(engine=engine).observe(audio_duration_sec)
    METRICS.transcription_time_seconds.labels(engine=engine).observe(transcribe_time)
    METRICS.real_time_factor.labels(engine=engine).observe(rtf)
    METRICS.language_detection_duration_ms.labels(method=detection_method).observe(lang_detect_ms)
    METRICS.language_detection_confidence.labels(method=detection_method).observe(confidence)
    METRICS.language_detection_total.labels(
        detected_language=best_lang_code, detection_method=detection_method
    ).inc()
    METRICS.audio_minutes_processed.labels(engine=engine).inc(audio_duration_sec / 60.0)

    # After VAD:
    METRICS.vad_triggered_total.labels(result=vad_result).inc()  # "speech", "silence", "skipped"

    # For batch processing:
    METRICS.batch_size.labels(tier=engine).observe(batch_size)
    ```
"""


def integrate_observability(app):
    """
    One-call integration: adds all observability to a FastAPI app.

    Call this in create_app() or after app creation:
        from blitz.observability.integration import integrate_observability
        integrate_observability(app)
    """
    from blitz.observability import setup_metrics, setup_logging, ObservabilityMiddleware

    # Setup structured logging
    setup_logging(
        log_level="INFO",
        json_format=True,  # Set to False for local dev (human-readable logs)
    )

    # Add middleware (request tracking, metrics, correlation IDs)
    app.add_middleware(ObservabilityMiddleware)

    # Setup /metrics endpoint for Prometheus scraping
    setup_metrics(app)

    import logging
    logger = logging.getLogger("blitz.observability")
    logger.info("Observability initialized: metrics=/metrics, structured logging=enabled")
