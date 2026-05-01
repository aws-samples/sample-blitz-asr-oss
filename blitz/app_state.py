"""
Application state management with dependency injection.

Provides a centralized application state container that holds all models,
services, queues, and configuration. Integrates with FastAPI's dependency
injection system to eliminate global state.
"""

import asyncio
from typing import Dict, Optional
from fastapi import FastAPI

from blitz.config import ServerConfig
from blitz.models import WhisperModel, VoxLinguaModel
from blitz.models.parakeet_factory import (
    create_tier2_model_from_config,
    create_tier1_model_from_config,
    ParakeetModelProtocol,
)
from blitz.services import (
    LanguageDetectionService,
    TranscriptionService,
    VADService,
    create_batch_worker,
)


class ApplicationState:
    """
    Application state container for dependency injection.

    Holds all models, services, queues, and configuration in a single
    thread-safe container. Eliminates global state and enables easier
    testing through dependency injection.
    """

    def __init__(
        self,
        config: Optional[ServerConfig] = None,
        tier2_model: Optional[ParakeetModelProtocol] = None,
        tier1_model: Optional[ParakeetModelProtocol] = None,
        whisper_model: Optional[WhisperModel] = None,
        voxlingua_model: Optional[VoxLinguaModel] = None,
        language_detection_service: Optional[LanguageDetectionService] = None,
        transcription_service: Optional[TranscriptionService] = None,
        vad_service: Optional[VADService] = None,
    ):
        """
        Initialize application state.

        Args:
            config: Server configuration (defaults to ServerConfig())
            tier2_model: Tier 2 Parakeet ASR model (0.6B, EU languages)
            tier1_model: Tier 1 Parakeet CTC model (110M, English-only, fastest)
            whisper_model: Tier 3 Whisper ASR model (catch-all)
            voxlingua_model: VoxLingua language detection model
            language_detection_service: Language detection service
            transcription_service: Transcription orchestration service
            vad_service: Voice activity detection service
        """
        # Configuration
        self.config = config or ServerConfig()

        # Models (3-tier architecture)
        self.tier1_model = tier1_model  # Fastest: CTC-110M for English
        self.tier2_model = tier2_model  # Fast multilingual: Parakeet 0.6B
        self.whisper_model = whisper_model  # Tier 3: Catch-all
        self.voxlingua_model = voxlingua_model

        # Services
        self.language_detection_service = language_detection_service
        self.transcription_service = transcription_service
        self.vad_service = vad_service

        # Queue for micro-batching
        self.transcription_queue: Optional[asyncio.Queue] = None

        # Results dictionary: job_id -> asyncio.Future
        self.pending_results: Dict[str, asyncio.Future] = {}

        # Batch worker tasks (one per tier)
        self.tier1_worker_task: Optional[asyncio.Task] = None
        self.tier2_worker_task: Optional[asyncio.Task] = None
        self.whisper_worker_task: Optional[asyncio.Task] = None

    @classmethod
    def _initialize_models_and_services(
        cls, config: ServerConfig, batch_mode: bool = False
    ) -> dict:
        """
        Initialize all models and services from config.

        This is the shared initialization logic used by both create_with_defaults()
        and create_for_batch(). Extracted to eliminate code duplication.

        Args:
            config: Server configuration
            batch_mode: If True, uses simpler logging for batch processing

        Returns:
            Dictionary with all initialized models and services
        """
        tier1_langs = config.get_tier1_languages()
        tier2_langs = config.get_tier2_languages()

        # Initialize models (3-tier architecture)
        mode_suffix = " for batch processing" if batch_mode else ""
        print(f"Initializing 3-tier model architecture{mode_suffix}...")

        # Tier 2: Fast multilingual (Parakeet 0.6B)
        print(
            f"    Tier 2: {config.tier2_model} (backend={config.tier2_backend}, precision={config.tier2_precision})"
        )
        print(f"           Languages: {', '.join(sorted(tier2_langs))}")
        tier2_model = create_tier2_model_from_config(config)

        # Tier 1: Fastest (CTC-110M for English)
        tier1_model = None
        if tier1_langs:
            print(
                f"    Tier 1: {config.tier1_model} (precision={config.tier1_precision})"
            )
            print(f"           Languages: {', '.join(sorted(tier1_langs))}")
            tier1_model = create_tier1_model_from_config(config)

        # Tier 3: Catch-all (Whisper)
        print(f"    Tier 3: {config.tier3_model} (catch-all)")
        whisper_model = WhisperModel(
            primary_model_name=config.tier3_model,
            language_detection_model=config.language_detection_model,
            device="cuda" if config.device == "cuda" else "cpu",
            primary_revision=config.tier3_model_revision or None,
            tiny_model_name=config.whisper_tiny_model_name,
            tiny_revision=config.whisper_tiny_revision or None,
        )

        # Initialize VoxLingua (required for VoxLingua-primary detection)
        voxlingua_model = None
        try:
            voxlingua_model = VoxLinguaModel(
                device="cpu"  # VoxLingua always runs on CPU to free GPU
            )
            if batch_mode:
                print("VoxLingua107 loaded (CPU)")
            elif config.primary_language_detector == "voxlingua":
                print(
                    f"VoxLingua107 loaded as PRIMARY detector (CPU, threshold={config.voxlingua_confidence_threshold:.2f})"
                )
            else:
                print(
                    f"VoxLingua107 loaded as fallback detector (CPU, threshold={config.voxlingua_confidence_threshold:.2f})"
                )
        except Exception as e:
            print(f"VoxLingua107 not available: {e}")
            if not batch_mode and config.primary_language_detector == "voxlingua":
                print("Falling back to Whisper as primary detector")

        # Initialize services
        print("Initializing services...")
        language_detection_service = LanguageDetectionService(
            whisper_model=whisper_model,
            voxlingua_model=voxlingua_model,
            primary_detector=config.primary_language_detector,
            secondary_detector=config.secondary_language_detector,
            voxlingua_threshold=config.voxlingua_confidence_threshold,
            whisper_threshold=config.whisper_confidence_threshold,
            min_confidence_threshold=config.min_language_confidence,
        )

        transcription_service = TranscriptionService(
            parakeet_model=tier2_model,
            whisper_model=whisper_model,
            language_detection_service=language_detection_service,
            short_audio_threshold_sec=config.short_audio_threshold_sec,
            chunk_length_sec=config.chunk_length_sec,
            chunk_overlap_sec=config.chunk_overlap_sec,
        )

        # Initialize VAD service (CPU is fast enough for VAD)
        vad_service = VADService(device="cpu", revision=config.silero_vad_revision)
        print("VAD service initialized (Silero VAD)")

        return {
            "tier1_model": tier1_model,
            "tier2_model": tier2_model,
            "whisper_model": whisper_model,
            "voxlingua_model": voxlingua_model,
            "language_detection_service": language_detection_service,
            "transcription_service": transcription_service,
            "vad_service": vad_service,
        }

    @classmethod
    def create_with_defaults(cls) -> "ApplicationState":
        """
        Create application state with default models and services.

        This factory method initializes all models and services with
        default configurations, suitable for production use.

        Returns:
            ApplicationState with fully initialized models and services
        """
        config = ServerConfig()
        components = cls._initialize_models_and_services(config, batch_mode=False)
        return cls(config=config, **components)

    @classmethod
    def create_for_batch(cls, config: ServerConfig) -> "ApplicationState":
        """
        Create application state for AWS Batch processing.

        Unlike create_with_defaults(), this method:
        - Accepts an externally-provided config (typically BatchConfig)
        - Does NOT initialize async queues or batch workers
        - Is designed for synchronous, single-job processing

        Args:
            config: ServerConfig or BatchConfig instance

        Returns:
            ApplicationState with models and services (no async components)
        """
        components = cls._initialize_models_and_services(config, batch_mode=True)
        print("Batch application state ready (no async queues)")
        return cls(config=config, **components)

    async def startup(self) -> None:
        """
        Initialize async resources on application startup.

        Creates the transcription queue and starts batch worker tasks.
        """
        print("=" * 80)
        print("Starting 3-Tier ASR Router with Micro-Batching")
        print("=" * 80)

        # Initialize queue
        self.transcription_queue = asyncio.Queue()
        print("Transcription queue initialized")

        # Import batch processing functions
        from blitz.batch_processing import (
            process_parakeet_batch,
            process_whisper_batch,
            aggressive_cleanup,
        )

        # Create Tier 2 (Parakeet 0.6B) batch processing function
        def tier2_processor(jobs):
            return process_parakeet_batch(
                jobs=jobs,
                parakeet_model=self.tier2_model,
                short_audio_threshold_sec=self.config.short_audio_threshold_sec,
                chunk_length_sec=self.config.chunk_length_sec,
                chunk_overlap_sec=self.config.chunk_overlap_sec,
                local_attention_threshold_sec=self.config.local_attention_threshold_sec,
            )

        # Create Tier 3 (Whisper) batch processing function
        def tier3_processor(jobs):
            return process_whisper_batch(
                jobs=jobs,
                whisper_model=self.whisper_model.model,
                whisper_processor=self.whisper_model.processor,
            )

        # Start batch workers
        print(
            f"\nStarting batch workers (timeout={self.config.batch_timeout_ms}ms, max_batch={self.config.max_batch_size})"
        )

        self.tier2_worker_task = create_batch_worker(
            engine_name="tier2",
            transcription_queue=self.transcription_queue,
            pending_results=self.pending_results,
            process_batch_func=tier2_processor,
            cleanup_func=lambda batch, results: aggressive_cleanup(batch, results),
            max_batch_size=self.config.max_batch_size,
            batch_timeout_ms=self.config.batch_timeout_ms,
        )

        self.whisper_worker_task = create_batch_worker(
            engine_name="tier3",
            transcription_queue=self.transcription_queue,
            pending_results=self.pending_results,
            process_batch_func=tier3_processor,
            cleanup_func=lambda batch, results: aggressive_cleanup(batch, results),
            max_batch_size=self.config.max_batch_size,
            batch_timeout_ms=self.config.batch_timeout_ms,
        )

        # Start Tier 1 batch worker if tier1 languages are configured
        if self.tier1_model is not None:

            def tier1_processor(jobs):
                return process_parakeet_batch(
                    jobs=jobs,
                    parakeet_model=self.tier1_model,
                    short_audio_threshold_sec=self.config.short_audio_threshold_sec,
                    chunk_length_sec=self.config.chunk_length_sec,
                    chunk_overlap_sec=self.config.chunk_overlap_sec,
                    local_attention_threshold_sec=self.config.local_attention_threshold_sec,
                )

            self.tier1_worker_task = create_batch_worker(
                engine_name="tier1",
                transcription_queue=self.transcription_queue,
                pending_results=self.pending_results,
                process_batch_func=tier1_processor,
                cleanup_func=lambda batch, results: aggressive_cleanup(batch, results),
                max_batch_size=self.config.max_batch_size,
                batch_timeout_ms=self.config.batch_timeout_ms,
            )

        print("Application state initialized")

    async def shutdown(self) -> None:
        """
        Clean up async resources on application shutdown.

        Cancels batch worker tasks and performs cleanup.
        """
        print("\n" + "=" * 80)
        print("Shutting down 3-Tier ASR Router")
        print("=" * 80)

        # Cancel batch workers
        if self.tier1_worker_task:
            self.tier1_worker_task.cancel()
            try:
                await self.tier1_worker_task
            except asyncio.CancelledError:
                pass

        if self.tier2_worker_task:
            self.tier2_worker_task.cancel()
            try:
                await self.tier2_worker_task
            except asyncio.CancelledError:
                pass

        if self.whisper_worker_task:
            self.whisper_worker_task.cancel()
            try:
                await self.whisper_worker_task
            except asyncio.CancelledError:
                pass

        print("Batch workers stopped")
        print("Application state cleaned up")


# Global application state instance (initialized in lifespan)
_app_state: Optional[ApplicationState] = None


def get_app_state() -> ApplicationState:
    """
    Get the current application state.

    This function is used as a FastAPI dependency to inject
    the application state into route handlers.

    Returns:
        ApplicationState instance

    Raises:
        RuntimeError: If application state is not initialized

    Example:
        ```python
        from fastapi import Depends

        @app.post("/transcribe")
        async def transcribe(
            state: ApplicationState = Depends(get_app_state)
        ):
            # Use state.transcription_service, state.config, etc.
            pass
        ```
    """
    if _app_state is None:
        raise RuntimeError(
            "Application state not initialized. "
            "Ensure lifespan context manager is used."
        )
    return _app_state


def set_app_state(state: ApplicationState) -> None:
    """
    Set the global application state.

    This should only be called once during application startup
    in the lifespan context manager.

    Args:
        state: ApplicationState instance to set
    """
    global _app_state
    _app_state = state


def create_app() -> FastAPI:
    """
    Create FastAPI application with dependency injection.

    Returns:
        Configured FastAPI application with lifespan management

    Example:
        ```python
        app = create_app()

        # Application state is automatically initialized on startup
        # and cleaned up on shutdown
        ```
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan with startup and shutdown."""
        # Startup
        state = ApplicationState.create_with_defaults()
        set_app_state(state)
        await state.startup()

        yield

        # Shutdown
        await state.shutdown()

    app = FastAPI(
        title="Blitz ASR Service",
        description="High-performance ASR with intelligent language routing",
        version="1.0.0",
        lifespan=lifespan,
    )

    return app
