"""
Blitz: High-performance ASR with intelligent language routing.

A modular transcription service combining NVIDIA Parakeet and OpenAI Whisper models
with micro-batching, intelligent language detection, and memory-efficient processing.
"""

from .app_state import ApplicationState, get_app_state, create_app
from .config import ServerConfig

__version__ = "1.0.0"

__all__ = [
    "ApplicationState",
    "get_app_state",
    "create_app",
    "ServerConfig",
]
