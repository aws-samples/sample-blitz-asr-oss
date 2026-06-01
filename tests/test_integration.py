#!/usr/bin/env python3
"""
Integration test for refactored modular architecture.

Verifies that all modules can be imported and basic integration works
without requiring GPU resources.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_module_imports():
    """Test that all refactored modules can be imported."""
    print("🔍 Testing module imports...")

    try:
        # Core modules
        from blitz import ApplicationState, get_app_state, create_app, ServerConfig
        print("  ✅ blitz core imports")

        # Config
        from blitz.config import ServerConfig
        print("  ✅ blitz.config")

        # Models
        from blitz.models import ParakeetModel, WhisperModel, VoxLinguaModel
        print("  ✅ blitz.models")

        # Services
        from blitz.services import (
            LanguageDetectionService,
            TranscriptionService,
            create_batch_worker
        )
        print("  ✅ blitz.services")

        # Schemas
        from blitz.schemas.api import (
            TranscriptionResponse,
            SRTResponse,
            HealthResponse,
            ModelsInfoResponse
        )
        print("  ✅ blitz.schemas")

        # Utils
        from blitz.utils.audio import (
            get_audio_duration,
            resample_audio,
            chunk_audio
        )
        from blitz.utils.formatting import (
            generate_srt_content,
            generate_vtt_content
        )
        print("  ✅ blitz.utils")

        # Batch processing
        from blitz.batch_processing import (
            process_parakeet_batch,
            process_whisper_batch,
            aggressive_cleanup
        )
        print("  ✅ blitz.batch_processing")

        # App state
        from blitz.app_state import ApplicationState, create_app
        print("  ✅ blitz.app_state")

        print("\n✅ All module imports successful!")
        return True

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test that configuration loads correctly."""
    print("\n🔍 Testing configuration loading...")

    try:
        from blitz.config import ServerConfig

        config = ServerConfig()

        # Verify key configuration values
        assert config.batch_timeout_ms > 0, "batch_timeout_ms must be positive"
        assert config.max_batch_size > 0, "max_batch_size must be positive"
        assert config.chunk_length_sec > 0, "chunk_length_sec must be positive"
        assert config.tier1_model, "tier1_model must be set"
        assert config.tier2_model, "tier2_model must be set"
        assert config.tier3_model, "tier3_model must be set"

        print(f"  Configuration loaded successfully")
        print(f"     - Batch timeout: {config.batch_timeout_ms}ms")
        print(f"     - Max batch size: {config.max_batch_size}")
        print(f"     - Chunk length: {config.chunk_length_sec}s")
        print(f"     - Tier 1 model: {config.tier1_model}")
        print(f"     - Tier 2 model: {config.tier2_model}")
        print(f"     - Tier 3 model: {config.tier3_model}")

        return True

    except Exception as e:
        print(f"\n❌ Configuration error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_schema_validation():
    """Test that Pydantic schemas validate correctly."""
    print("\n🔍 Testing schema validation...")

    try:
        from blitz.schemas.api import (
            LanguageInfo,
            ProcessingMetrics,
            TranscriptionResponse
        )

        # Test LanguageInfo
        lang_info = LanguageInfo(code="en", confidence=0.95)
        assert lang_info.code == "en"
        assert lang_info.confidence == 0.95
        print("  ✅ LanguageInfo schema")

        # Test ProcessingMetrics
        metrics = ProcessingMetrics(
            processing_time_sec=1.5,
            gpu_cost_usd=0.0001
        )
        assert metrics.processing_time_sec == 1.5
        print("  ✅ ProcessingMetrics schema")

        # Test TranscriptionResponse
        response = TranscriptionResponse(
            text="Hello world",
            language=lang_info,
            engine="tier2",
            duration_sec=10.0,
            timestamps=None,
            metrics=metrics
        )
        assert response.text == "Hello world"
        assert response.engine == "tier2"
        print("  TranscriptionResponse schema")

        print("\n✅ All schema validations passed!")
        return True

    except Exception as e:
        print(f"\n❌ Schema validation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fastapi_app_creation():
    """Test that FastAPI app can be created."""
    print("\n🔍 Testing FastAPI app creation...")

    try:
        from blitz import create_app

        app = create_app()

        # Verify app is created
        assert app is not None, "App should be created"
        assert hasattr(app, 'routes'), "App should have routes"

        # Check routes exist (they should be registered when blitz.server imports app)
        # Note: Routes are registered in blitz/server.py, not in create_app()
        print(f"  ✅ FastAPI app created successfully")
        print(f"     - App title: {app.title}")
        print(f"     - App version: {app.version}")

        return True

    except Exception as e:
        print(f"\n❌ FastAPI app creation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utility_functions():
    """Test utility functions with mock data."""
    print("\n🔍 Testing utility functions...")

    try:
        import numpy as np
        from blitz.utils.audio import get_audio_duration
        from blitz.utils.formatting import format_srt_time, format_vtt_time

        # Test get_audio_duration
        audio = np.random.randn(16000)  # 1 second of audio at 16kHz
        duration = get_audio_duration(audio, 16000)
        assert 0.9 < duration < 1.1, f"Duration should be ~1.0s, got {duration}s"
        print("  ✅ get_audio_duration()")

        # Test format_srt_time
        srt_time = format_srt_time(65.5)  # 1 min 5.5 sec
        assert "01:05,500" in srt_time, f"SRT time format incorrect: {srt_time}"
        print("  ✅ format_srt_time()")

        # Test format_vtt_time
        vtt_time = format_vtt_time(65.5)
        assert "01:05.500" in vtt_time, f"VTT time format incorrect: {vtt_time}"
        print("  ✅ format_vtt_time()")

        print("\n✅ All utility functions working!")
        return True

    except Exception as e:
        print(f"\n❌ Utility function error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("=" * 70)
    print("🧪 INTEGRATION TESTS - Modular Architecture")
    print("=" * 70)
    print("\nThese tests verify the refactored modules can be imported and")
    print("basic functionality works without requiring GPU resources.\n")

    results = []

    # Run tests
    results.append(("Module Imports", test_module_imports()))
    results.append(("Configuration Loading", test_config_loading()))
    results.append(("Schema Validation", test_schema_validation()))
    results.append(("FastAPI App Creation", test_fastapi_app_creation()))
    results.append(("Utility Functions", test_utility_functions()))

    # Print summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")

    print("=" * 70)

    if passed == total:
        print("✅ ALL TESTS PASSED - Integration successful!")
        print("\nNext steps:")
        print("  1. Start the integrated server: python3 -m blitz.server")
        print("  2. Run load tests: python3 tests/test_load.py -n 5")
        print("  3. Verify GPU memory cleanup and batch processing")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Fix errors before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())
