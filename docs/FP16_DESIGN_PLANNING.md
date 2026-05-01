# FP16 Parakeet Model Design & Planning

## Executive Summary

This document outlines the design and implementation plan for adding FP16 (Half-Precision) Parakeet model support to Blitz. The goal is to provide an optional flag to switch between FP32/BFloat16 (current NeMo-based) and FP16 (ONNX-based) model variants.

**Key Benefits of FP16:**
- **50% smaller model size**: 1.25GB vs 2.4GB
- **Faster inference**: FP16 Tensor Core acceleration on modern GPUs
- **Lower memory**: Reduced VRAM usage for batching
- **Same accuracy**: Minimal quality loss from quantization

---

## Background

### Current Architecture (NeMo + BFloat16)

The current Blitz implementation uses:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CURRENT PARAKEET STACK                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Model: nvidia/parakeet-tdt-0.6b-v3 (HuggingFace)             │
│   Framework: NVIDIA NeMo 2.4.0                                  │
│   Precision: BFloat16 (16-bit brain floating point)            │
│   Runtime: PyTorch native                                       │
│   Model Size: ~2.4GB (FP32 weights loaded, cast to BFloat16)   │
│                                                                 │
│   Loading:                                                      │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ model = nemo_asr.models.ASRModel.from_pretrained()      │  │
│   │ model = model.to("cuda").to(dtype=torch.bfloat16)       │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   Memory: ~5.0GB VRAM (BFloat16)                               │
│   Speed: ~1.5-1.8s for 24s audio                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**File:** `blitz/models/parakeet.py`

### Proposed FP16 ONNX Architecture

The FP16 variant uses a completely different stack:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FP16 ONNX PARAKEET STACK                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Model: grikdotnet/parakeet-tdt-0.6b-fp16 (HuggingFace)       │
│   Framework: ONNX Runtime (onnx-asr library)                    │
│   Precision: FP16 (IEEE 754 half-precision)                    │
│   Runtime: ONNX Runtime with CUDA ExecutionProvider            │
│   Model Size: ~1.25GB (natively FP16)                          │
│                                                                 │
│   Model Files:                                                  │
│   ├── encoder-model.fp16.onnx (1.2GB)                          │
│   ├── decoder_joint-model.fp16.onnx (35MB)                     │
│   ├── config.json                                               │
│   ├── vocab.txt                                                 │
│   └── nemo128.onnx (FeatureExtractor)                          │
│                                                                 │
│   Loading:                                                      │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ import onnx_asr                                         │  │
│   │ model = onnx_asr.load_model(                            │  │
│   │     'nemo-parakeet-tdt-0.6b-v3',                        │  │
│   │     model_path,                                         │  │
│   │     quantization='fp16',                                │  │
│   │     providers=['CUDAExecutionProvider']                 │  │
│   │ )                                                       │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   Memory: ~2.5GB VRAM (true FP16)                              │
│   Speed: TBD (benchmarking required)                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Precision Format Comparison

| Format | Exponent | Mantissa | Range | Model Size | GPU Support |
|--------|----------|----------|-------|------------|-------------|
| FP32 | 8 bits | 23 bits | ~1e-38 to 1e38 | 2.4GB | All GPUs |
| BFloat16 | 8 bits | 7 bits | ~1e-38 to 1e38 | 1.2GB* | Ampere+ |
| **FP16** | **5 bits** | **10 bits** | **~6e-8 to 6e4** | **1.2GB** | **All GPUs** |

*BFloat16 loaded from FP32 weights with dynamic casting

### Key Differences

| Aspect | BFloat16 (Current) | FP16 ONNX (Proposed) |
|--------|-------------------|---------------------|
| **Runtime** | PyTorch/NeMo | ONNX Runtime |
| **Loading** | Dynamic cast at runtime | Pre-converted models |
| **Precision** | Same range as FP32 | Narrower range (overflow risk) |
| **GPU Support** | Ampere+ native, emulated on older | Native on all modern GPUs |
| **File Size** | 2.4GB (FP32 on disk) | 1.25GB (FP16 on disk) |
| **Memory** | ~5GB VRAM | ~2.5GB VRAM |
| **Timestamp Support** | Native NeMo 2.4+ | Requires implementation |
| **Batch Transcribe** | Supported | May need adaptation |

---

## Implementation Options

### Option 1: ONNX Runtime with onnx-asr Library (Recommended)

Use the pre-converted FP16 ONNX models from HuggingFace with the `onnx-asr` library.

**Pros:**
- Pre-optimized FP16 models (tested and validated)
- Smaller disk footprint (1.25GB)
- Lower memory usage (~2.5GB VRAM)
- Works on all CUDA GPUs (including T4, V100)
- `onnx-asr` provides clean API

**Cons:**
- New dependency (`onnx-asr`)
- Different inference API than NeMo
- Timestamp extraction may differ
- Need to implement adapter for Blitz interface

**Implementation:**

```python
# blitz/models/parakeet_onnx.py
import onnx_asr
import numpy as np
from typing import Optional, Tuple, Dict

class ParakeetONNXModel:
    """ONNX-based Parakeet model with FP16 support."""

    def __init__(
        self,
        model_path: str,
        quantization: str = "fp16",  # "fp16" or None for FP32
        device: str = "cuda"
    ):
        self.model = onnx_asr.load_model(
            'nemo-parakeet-tdt-0.6b-v3',
            model_path,
            quantization=quantization,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            if device == "cuda" else ['CPUExecutionProvider'],
            cpu_preprocessing=False
        )

    def transcribe(
        self,
        audio: np.ndarray,
        timestamps: bool = True
    ) -> Tuple[str, Optional[Dict]]:
        """Transcribe audio with optional timestamps."""
        text = self.model.recognize(audio)
        # Note: onnx-asr timestamp format may differ from NeMo
        timestamp_data = None  # TBD: implement timestamp extraction
        return text, timestamp_data
```

### Option 2: PyTorch Native FP16 (torch.float16)

Convert existing NeMo models to FP16 at runtime.

**Pros:**
- No new dependencies
- Same codebase/API as current
- Easy to implement (2 lines)

**Cons:**
- Still loads FP32 from disk (2.4GB)
- Potential numerical instability (narrower range than BFloat16)
- May cause NaN/Inf issues in some layers
- Not tested with NeMo ASR models

**Implementation:**

```python
# In blitz/models/parakeet.py
if precision == "fp16":
    self.model = self.model.to(dtype=torch.float16)
elif precision == "bfloat16":
    self.model = self.model.to(dtype=torch.bfloat16)
```

### Option 3: Hybrid Approach

Support both NeMo (BFloat16) and ONNX (FP16) backends with a unified interface.

**Pros:**
- Best of both worlds
- User can choose based on their GPU/requirements
- Fallback if one fails

**Cons:**
- More complex codebase
- Two code paths to maintain
- Need to align timestamp formats

---

## Recommended Architecture

### Unified Model Interface

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROPOSED ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Configuration (blitz/config.py):                              │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ tier2_backend: str = "nemo"    # "nemo" or "onnx"       │  │
│   │ tier2_precision: str = "bfloat16"  # "fp32", "fp16",    │  │
│   │                                       # "bfloat16"      │  │
│   │ tier2_onnx_model_path: str = "./models/parakeet"        │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   Model Factory (blitz/models/parakeet_factory.py):             │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ def create_parakeet_model(config):                      │  │
│   │     if config.tier2_backend == "onnx":                  │  │
│   │         return ParakeetONNXModel(                       │  │
│   │             model_path=config.tier2_onnx_model_path,    │  │
│   │             quantization="fp16"                         │  │
│   │         )                                               │  │
│   │     else:  # "nemo"                                     │  │
│   │         return ParakeetModel(                           │  │
│   │             model_name=config.tier2_model,              │  │
│   │             precision=config.tier2_precision            │  │
│   │         )                                               │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   Unified Interface:                                            │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ class ParakeetModelInterface(ABC):                      │  │
│   │     def transcribe(audio, timestamps) -> (text, ts)     │  │
│   │     def set_attention_mode(duration) -> None            │  │
│   │     @property supported_languages -> Set[str]           │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration Changes

### New Environment Variables

```bash
# Tier 2 Backend Selection
TIER2_BACKEND=nemo           # "nemo" (default) or "onnx"

# Precision Control
TIER2_PRECISION=bfloat16     # "fp32", "bfloat16", "fp16"

# ONNX Model Path (only for backend=onnx)
TIER2_ONNX_MODEL_PATH=./models/parakeet

# Auto-download FP16 models from HuggingFace
TIER2_AUTO_DOWNLOAD=true     # Download FP16 models if missing
```

### Config Class Updates

```python
# blitz/config.py

class ServerConfig(BaseSettings):
    # ... existing fields ...

    # Tier 2 Backend Configuration
    tier2_backend: str = Field(
        default="nemo",
        description="Tier 2 backend: 'nemo' (PyTorch/NeMo) or 'onnx' (ONNX Runtime)"
    )

    tier2_precision: str = Field(
        default="bfloat16",
        description="Model precision: 'fp32', 'bfloat16', 'fp16'"
    )

    tier2_onnx_model_path: str = Field(
        default="./models/parakeet",
        description="Path to ONNX model files (for backend='onnx')"
    )

    tier2_auto_download: bool = Field(
        default=True,
        description="Auto-download FP16 ONNX models from HuggingFace"
    )

    @field_validator("tier2_backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        v = v.lower()
        if v not in ["nemo", "onnx"]:
            raise ValueError(f"tier2_backend must be 'nemo' or 'onnx', got: {v}")
        return v

    @field_validator("tier2_precision")
    @classmethod
    def validate_precision(cls, v: str) -> str:
        v = v.lower()
        if v not in ["fp32", "bfloat16", "fp16"]:
            raise ValueError(f"tier2_precision must be 'fp32', 'bfloat16', or 'fp16', got: {v}")
        return v
```

---

## Model Download & Setup

### ONNX FP16 Model Sources

**HuggingFace Repository:**
- URL: https://huggingface.co/grikdotnet/parakeet-tdt-0.6b-fp16
- Files:
  - `encoder-model.fp16.onnx` (1.2GB)
  - `decoder_joint-model.fp16.onnx` (35MB)

**Additional Required Files (from FP32 repo):**
- URL: https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx
- Files:
  - `config.json`
  - `vocab.txt`
  - `nemo128.onnx` (FeatureExtractor)

### Download Script

```python
# scripts/download_fp16_models.py
from huggingface_hub import hf_hub_download
from pathlib import Path

def download_fp16_parakeet(output_dir: str = "./models/parakeet"):
    """Download FP16 Parakeet ONNX models."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # FP16 models
    fp16_repo = "grikdotnet/parakeet-tdt-0.6b-fp16"
    for file in ["encoder-model.fp16.onnx", "decoder_joint-model.fp16.onnx"]:
        hf_hub_download(repo_id=fp16_repo, filename=file, local_dir=output_path)

    # Supporting files from FP32 repo
    fp32_repo = "istupakov/parakeet-tdt-0.6b-v3-onnx"
    for file in ["config.json", "vocab.txt", "nemo128.onnx"]:
        hf_hub_download(repo_id=fp32_repo, filename=file, local_dir=output_path)

    print(f"✅ FP16 Parakeet models downloaded to {output_path}")
```

---

## Implementation Plan

### Phase 1: Research & Baseline (Week 1)

1. **Set up benchmarking environment**
   - Create test audio corpus (various lengths, languages)
   - Define metrics: latency, throughput, memory, accuracy (WER)

2. **Baseline current BFloat16 performance**
   - Memory usage at various batch sizes
   - Inference latency per audio length
   - Throughput at concurrency 1, 5, 10, 20

3. **Install and test onnx-asr**
   - Install `onnx-asr` library
   - Download FP16 models
   - Run basic inference tests

### Phase 2: ONNX Integration (Week 2)

1. **Create ParakeetONNXModel class**
   - Implement unified interface
   - Handle model loading with `onnx-asr`
   - Match transcribe() signature with NeMo version

2. **Implement timestamp extraction**
   - Research onnx-asr timestamp format
   - Create adapter for Blitz timestamp format
   - Test with SRT/VTT generation

3. **Create model factory**
   - Implement backend selection logic
   - Add config validation
   - Handle auto-download

### Phase 3: Benchmarking (Week 3)

1. **Performance comparison**
   - BFloat16 (NeMo) vs FP16 (ONNX)
   - Latency: various audio lengths
   - Throughput: various concurrency levels
   - Memory: various batch sizes

2. **Accuracy comparison**
   - WER on LibriSpeech test-clean
   - Quality on multi-language corpus
   - Edge cases: accents, background noise

3. **GPU compatibility testing**
   - Test on L4, A10G, A100
   - Test on T4, V100 (older GPUs)
   - Identify any hardware-specific issues

### Phase 4: Integration & Deployment (Week 4)

1. **Update Dockerfile**
   - Add onnx-asr to requirements
   - Add model download step (optional)
   - Update environment variable documentation

2. **Documentation**
   - Update README with FP16 option
   - Add benchmarking results
   - Document GPU compatibility

3. **Testing**
   - Integration tests for both backends
   - Load tests with FP16 backend
   - Regression tests for existing functionality

---

## Benchmarking Plan

### Test Corpus

| Category | Audio Length | Languages | Samples |
|----------|-------------|-----------|---------|
| Short | 5-10s | en, de, fr, es | 100 |
| Medium | 30-60s | en, de, fr, es | 50 |
| Long | 5-10min | en, de | 20 |
| Very Long | 30-60min | en | 10 |

### Metrics

```python
# Benchmark output format
{
    "backend": "onnx",  # or "nemo"
    "precision": "fp16",  # or "bfloat16", "fp32"
    "audio_duration_sec": 24.0,
    "metrics": {
        "latency_ms": {
            "mean": 1234.5,
            "p50": 1200.0,
            "p95": 1450.0,
            "p99": 1580.0
        },
        "throughput_req_per_sec": 1.87,
        "memory_mb": {
            "model_load": 2500,
            "peak_inference": 3200
        },
        "accuracy": {
            "wer": 4.2,
            "cer": 1.8
        }
    }
}
```

### Benchmark Script

```bash
# scripts/benchmark_fp16.py
python scripts/benchmark_fp16.py \
    --backends nemo,onnx \
    --precisions bfloat16,fp16 \
    --audio-dir ./test_audio \
    --concurrency 1,5,10,20 \
    --output-file results/fp16_benchmark.json
```

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| onnx-asr API changes | Medium | Pin version in requirements |
| Timestamp format mismatch | Medium | Create adapter layer |
| FP16 numerical instability | Low | Fallback to BFloat16 |
| ONNX model unavailable | Low | Auto-download from HuggingFace |
| Performance regression | High | Keep NeMo as default, FP16 opt-in |
| Incompatible GPUs | Low | Test matrix, document compatibility |

---

## Success Criteria

1. **Performance**: FP16 ONNX matches or exceeds BFloat16 NeMo throughput
2. **Accuracy**: WER within 0.5% of BFloat16 baseline
3. **Memory**: FP16 uses ≤50% of BFloat16 memory
4. **Compatibility**: Works on L4, A10G, A100, T4, V100
5. **Stability**: No NaN/Inf errors in 1000+ transcriptions
6. **Timestamps**: SRT/VTT generation matches NeMo output quality

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `blitz/models/parakeet_onnx.py` | Create | ONNX-based Parakeet model |
| `blitz/models/parakeet_factory.py` | Create | Model factory for backend selection |
| `blitz/models/parakeet_interface.py` | Create | Abstract base interface |
| `blitz/models/parakeet.py` | Modify | Add precision parameter |
| `blitz/config.py` | Modify | Add backend/precision config |
| `blitz/app_state.py` | Modify | Use factory for model creation |
| `scripts/download_fp16_models.py` | Create | Model download utility |
| `scripts/benchmark_fp16.py` | Create | Benchmarking script |
| `tests/test_parakeet_onnx.py` | Create | ONNX backend tests |
| `Dockerfile` | Modify | Add onnx-asr, optional model download |
| `requirements.txt` | Modify | Add onnx-asr dependency |
| `docs/FP16_BENCHMARKS.md` | Create | Benchmark results |
| `README.md` | Modify | Document FP16 option |

---

## Dependencies

### New Dependencies

```txt
# requirements.txt additions
onnx-asr>=0.10.0
onnx>=1.14.0
onnxruntime-gpu>=1.16.0
```

### Existing Dependencies

```txt
# Already in requirements.txt
nemo-toolkit>=2.4.0
torch>=2.0.0
huggingface-hub>=0.20.0
```

---

## References

- [FP16 Parakeet Model (HuggingFace)](https://huggingface.co/grikdotnet/parakeet-tdt-0.6b-fp16)
- [FP16 Conversion Scripts (GitHub)](https://github.com/grikdotnet/ai-stenographer/tree/master/misc)
- [onnx-asr Library (PyPI)](https://pypi.org/project/onnx-asr/)
- [ONNX FP32 Parakeet (HuggingFace)](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)
- [Original Parakeet Model (NVIDIA)](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

---

## Appendix: FP16 Conversion Scripts

The FP16 models were created using these conversion scripts from `grikdotnet/ai-stenographer`:

### convert_to_fp16.py

Converts FP32 ONNX models to FP16:

```python
from onnxconverter_common import float16
import onnx

model = onnx.load('encoder-model.onnx', load_external_data=True)
model_fp16 = float16.convert_float_to_float16(
    model,
    keep_io_types=True,           # Keep I/O as FP32 for compatibility
    disable_shape_infer=True      # Preserve weights in external data models
)
onnx.save(model_fp16, 'encoder-model.fp16.onnx')
```

### fix_fp16_casts.py

Fixes Cast operations that cause type mismatches:

```python
import onnx

model = onnx.load('encoder-model.fp16.onnx')

# Convert internal Cast(to=FLOAT) to Cast(to=FLOAT16)
FLOAT, FLOAT16 = 1, 10
for node in model.graph.node:
    if node.op_type == 'Cast':
        for attr in node.attribute:
            if attr.name == 'to' and attr.i == FLOAT:
                attr.i = FLOAT16

onnx.save(model, 'encoder-model.fp16.onnx')
```

---

## Changelog

| Date | Version | Author | Description |
|------|---------|--------|-------------|
| 2025-01-02 | 0.1 | - | Initial design document |
