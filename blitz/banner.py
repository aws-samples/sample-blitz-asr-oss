"""
Blitz ASCII Art Banner for Terminal Startup
"""

BLITZ_BANNER = r"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║     ██████╗ ██╗     ██╗████████╗███████╗    ⚡                           ║
║     ██╔══██╗██║     ██║╚══██╔══╝╚══███╔╝                                 ║
║     ██████╔╝██║     ██║   ██║     ███╔╝     Lightning-Fast               ║
║     ██╔══██╗██║     ██║   ██║    ███╔╝      Audio Transcription          ║
║     ██████╔╝███████╗██║   ██║   ███████╗                                 ║
║     ╚═════╝ ╚══════╝╚═╝   ╚═╝   ╚══════╝    GPU-Accelerated ASR          ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

BLITZ_BANNER_COMPACT = r"""
    ██████╗ ██╗     ██╗████████╗███████╗  ⚡
    ██╔══██╗██║     ██║╚══██╔══╝╚══███╔╝
    ██████╔╝██║     ██║   ██║     ███╔╝   Lightning-Fast ASR
    ██╔══██╗██║     ██║   ██║    ███╔╝
    ██████╔╝███████╗██║   ██║   ███████╗  GPU-Accelerated
    ╚═════╝ ╚══════╝╚═╝   ╚═╝   ╚══════╝
"""

def print_banner(compact: bool = False):
    """Print the Blitz banner to terminal"""
    if compact:
        print(BLITZ_BANNER_COMPACT)
    else:
        print(BLITZ_BANNER)
    print("    🚀 High-Performance Audio Transcription Service")
    print("    📊 Real-Time Factor: 0.069-0.217 | Capacity: 0.75 min/s")
    print("    ⚡ Server-side micro-batching enabled")
    print()


def print_startup_info(host: str = "0.0.0.0", port: int = 8000):
    """Print startup information"""
    print(f"    🌐 Server running on http://{host}:{port}")
    print(f"    📚 API docs: http://{host}:{port}/docs")
    print(f"    ❤️  Health check: http://{host}:{port}/health")
    print()
    print("=" * 80)
    print()
