#!/usr/bin/env python3
"""
Download FP16 ONNX Parakeet models from HuggingFace.

This script downloads the FP16 quantized ONNX models for Parakeet TDT 0.6B v3,
providing 50% smaller model size and faster inference compared to FP32.

Models are sourced from:
- FP16 models: https://huggingface.co/grikdotnet/parakeet-tdt-0.6b-fp16
- Supporting files: https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx

Usage:
    python scripts/download_fp16_models.py [--output-dir ./models/parakeet]
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# HuggingFace repositories
FP16_REPO = "grikdotnet/parakeet-tdt-0.6b-fp16"
FP32_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"

# Files to download
FP16_FILES = [
    "encoder-model.fp16.onnx",      # 1.2 GB - Main encoder
    "decoder_joint-model.fp16.onnx", # 35 MB - Decoder/joint network
]

SUPPORT_FILES = [
    "config.json",    # Model configuration
    "vocab.txt",      # Vocabulary file
    "nemo128.onnx",   # Feature extractor
]

# Optional FP32 files (for comparison/fallback)
FP32_FILES = [
    "encoder-model.onnx",
    "decoder_joint-model.onnx",
]


def check_huggingface_hub():
    """Check if huggingface_hub is installed."""
    try:
        from huggingface_hub import hf_hub_download
        return True
    except ImportError:
        return False


def download_file(repo_id: str, filename: str, output_dir: Path) -> bool:
    """Download a single file from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    try:
        logger.info(f"  Downloading {filename} from {repo_id}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False
        )
        return True
    except Exception as e:
        logger.error(f"  Failed to download {filename}: {e}")
        return False


def verify_download(output_dir: Path, files: list) -> bool:
    """Verify all required files were downloaded."""
    missing = []
    for filename in files:
        if not (output_dir / filename).exists():
            missing.append(filename)

    if missing:
        logger.error(f"Missing files: {', '.join(missing)}")
        return False

    return True


def get_file_sizes(output_dir: Path, files: list) -> dict:
    """Get file sizes in human-readable format."""
    sizes = {}
    for filename in files:
        filepath = output_dir / filename
        if filepath.exists():
            size_bytes = filepath.stat().st_size
            if size_bytes > 1_000_000_000:
                sizes[filename] = f"{size_bytes / 1_000_000_000:.2f} GB"
            elif size_bytes > 1_000_000:
                sizes[filename] = f"{size_bytes / 1_000_000:.1f} MB"
            else:
                sizes[filename] = f"{size_bytes / 1_000:.1f} KB"
    return sizes


def download_fp16_models(output_dir: Path, include_fp32: bool = False) -> bool:
    """
    Download FP16 ONNX models and supporting files.

    Args:
        output_dir: Directory to save models
        include_fp32: Also download FP32 models for comparison

    Returns:
        True if all required files downloaded successfully
    """
    if not check_huggingface_hub():
        logger.error("huggingface-hub is required. Install with: pip install huggingface-hub")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading models to: {output_dir}")

    success = True

    # Download FP16 models
    logger.info("\n📦 Downloading FP16 quantized models...")
    for filename in FP16_FILES:
        if not download_file(FP16_REPO, filename, output_dir):
            success = False

    # Download supporting files
    logger.info("\n📦 Downloading supporting files...")
    for filename in SUPPORT_FILES:
        if not download_file(FP32_REPO, filename, output_dir):
            success = False

    # Optionally download FP32 models
    if include_fp32:
        logger.info("\n📦 Downloading FP32 models (for comparison)...")
        for filename in FP32_FILES:
            if not download_file(FP32_REPO, filename, output_dir):
                # FP32 is optional, don't fail
                logger.warning(f"  FP32 download failed (optional): {filename}")

    # Verify required files
    required_files = FP16_FILES + SUPPORT_FILES
    if not verify_download(output_dir, required_files):
        success = False

    # Print summary
    if success:
        logger.info("\n✅ Download complete!")
        sizes = get_file_sizes(output_dir, FP16_FILES + SUPPORT_FILES)
        logger.info("\nDownloaded files:")
        for filename, size in sizes.items():
            logger.info(f"  {filename}: {size}")

        total_size = sum(
            (output_dir / f).stat().st_size
            for f in required_files
            if (output_dir / f).exists()
        )
        logger.info(f"\nTotal size: {total_size / 1_000_000_000:.2f} GB")
    else:
        logger.error("\n❌ Download incomplete. Please check errors above.")

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Download FP16 ONNX Parakeet models from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/parakeet",
        help="Directory to save models (default: ./models/parakeet)"
    )
    parser.add_argument(
        "--include-fp32",
        action="store_true",
        help="Also download FP32 models for comparison benchmarking"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing files, don't download"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.verify_only:
        logger.info(f"Verifying models in: {output_dir}")
        required = FP16_FILES + SUPPORT_FILES
        if verify_download(output_dir, required):
            sizes = get_file_sizes(output_dir, required)
            logger.info("✅ All required files present:")
            for filename, size in sizes.items():
                logger.info(f"  {filename}: {size}")
            sys.exit(0)
        else:
            sys.exit(1)

    success = download_fp16_models(
        output_dir=output_dir,
        include_fp32=args.include_fp32
    )

    if success:
        print("\n" + "=" * 60)
        print("🎉 FP16 models ready!")
        print("=" * 60)
        print(f"\nTo use FP16 models, set these environment variables:")
        print(f"  export PARAKEET_BACKEND=onnx")
        print(f"  export PARAKEET_ONNX_MODEL_PATH={output_dir.absolute()}")
        print(f"\nOr pass to create_parakeet_model():")
        print(f"  model = create_parakeet_model(")
        print(f"      backend='onnx',")
        print(f"      onnx_model_path='{output_dir}'")
        print(f"  )")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
