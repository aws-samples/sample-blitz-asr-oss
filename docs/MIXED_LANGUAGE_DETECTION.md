# Mixed Language Detection Fix

## Problem

Language detection routes all "English" audio to Tier 1 (English-only model) regardless of confidence. Mixed-language audio with English intros gets misrouted.

### Test Cases (from load test)

**Spanish podcasts misrouted:**

| File | Duration | Detected | Engine | Expected |
|------|----------|----------|--------|----------|
| `spanish-286-pasando.mp3` | 45.7 min | en | tier1 | es → tier2 |
| `spanish-291-cerditos.mp3` | 28.4 min | en | tier1 | es → tier2 |
| `spanish-292-prestardinero.mp3` | 39.4 min | la (Latin!) | whisper | es → tier2 |

**Chinese/Japanese AP exam files (verify actual content):**

| File | Duration | Detected | Engine | Verify |
|------|----------|----------|--------|--------|
| `ap25-chinese-cultural-presenta*.mp3` | 0.3 min | en | tier1 | zh? or actually English narration |
| `ap25-japanese-cultural-present*.mp3` | 6.6 min | en | tier1 | ja? or actually English narration |

**No engine routing issues:**

| File | Duration | Detected | Engine | Issue |
|------|----------|----------|--------|-------|
| `czech-2_mono.wav` | 1.25 min | en | none | Czech misdetected as English, no engine handles it |
| `beeping_only_test_audio_mono.wav` | 0.17 min | en | none | Beeping sound, no speech (expected - not a bug) |

**Failure modes:**
1. **English intro** (286, 291): VoxLingua detects English intro → routes to English-only CTC
2. **Latin confusion** (292): Whisper confuses formal Spanish with Latin
3. **Mixed content** (AP files): May contain English narration about foreign cultures - need to verify
4. **Czech misdetection** (czech-2): Czech audio misdetected as English with low confidence, falls through routing
5. **Non-speech audio** (english-test_audio): Beeping/tones with no speech - expected to have no valid engine

### `spanish-286-pasando.mp3` (primary test case)

| Segment | Time | Language |
|---------|------|----------|
| Intro | 0-40s | English |
| Main | 40s-45min | Spanish |

**Current**: Detects "en" → routes to `tier1` → Spanish content garbled

**Expected**: Low confidence "en" → routes to `tier2` → code-switching handled

---

## Root Cause

```python
# blitz/server.py - No confidence check
tier1_langs = state.config.get_tier1_languages()
if (
    tier1_langs
    and state.tier1_model is not None
    and best_lang_code in tier1_langs  # BUG: No confidence threshold
):
    engine = "tier1"
```

---

## Fix

### Config (1 new field)

```python
# blitz/config.py
tier1_min_confidence: float = Field(
    default=0.85,
    description="Minimum language detection confidence to route to Tier 1 (English-only). "
                "Lower confidence suggests mixed content, routes to Tier 2 instead.",
    ge=0.5,
    le=1.0,
)
```

### Routing (1 line change)

```python
# blitz/server.py
tier1_langs = state.config.get_tier1_languages()
if (
    tier1_langs
    and state.tier1_model is not None
    and best_lang_code in tier1_langs
    and detection_result.confidence >= state.config.tier1_min_confidence  # NEW
):
    engine = "tier1"
```

---

## Why This Works

| Audio Type | Detection | Confidence | Route | Result |
|------------|-----------|------------|-------|--------|
| Pure English | "en" | 0.95 | `tier1` | Fast, correct |
| English intro + Spanish | "en" | 0.72 | `tier2` | Code-switching handled |
| Pure Spanish | "es" | 0.93 | `tier2` | Correct |

Parakeet v3 (0.6B) handles code-switching at the tokenizer level. CTC (110M) is English-only.

---

## Test

### Evaluation Files

**Spanish podcasts** (`tests/fixtures/podcasts/`):

| File | Issue | Pass Criteria |
|------|-------|---------------|
| `spanish-286-pasando.mp3` | English intro → detected "en" | engine=tier2, language=es |
| `spanish-291-cerditos.mp3` | English intro → detected "en" | engine=tier2, language=es |
| `spanish-292-prestardinero.mp3` | Detected as Latin | engine=tier2, language=es |

**AP exam files** (`test/` - verify actual language first):

| File | Issue | Pass Criteria |
|------|-------|---------------|
| `ap25-chinese-cultural-presenta*.mp3` | Detected "en" | If zh content: engine=tier3, language=zh |
| `ap25-japanese-cultural-present*.mp3` | Detected "en" | If ja content: engine=tier3, language=ja |

**No engine issues** (`tests/fixtures/`):

| File | Issue | Pass Criteria |
|------|-------|---------------|
| `czech-2.wav` | Detected "en", engine=none | engine=tier2, language=cs |
| `beeping_only_test_audio.wav` | Beeping sound, no speech | engine=none is acceptable (non-speech audio) |

### Manual Test

```bash
# Spanish podcasts (before: tier1/en, after: tier2/es)
for f in spanish-286-pasando spanish-291-cerditos spanish-292-prestardinero; do
  echo "=== $f ==="
  curl -s -X POST http://localhost:8000/transcribe \
    -F "file=@tests/fixtures/podcasts/${f}.mp3" \
    -F "format=json" | jq '{engine, lang: .language.code, conf: .language.confidence}'
done

# AP exam files (verify actual content first)
for f in ap25-chinese-cultural-presenta ap25-japanese-cultural-present; do
  echo "=== $f ==="
  curl -s -X POST http://localhost:8000/transcribe \
    -F "file=@test/${f}*.mp3" \
    -F "format=json" | jq '{engine, lang: .language.code, conf: .language.confidence, text: .text[:100]}'
done

# Czech file (before: none/en, after: tier2/cs)
echo "=== czech-2.wav ==="
curl -s -X POST http://localhost:8000/transcribe \
  -F "file=@tests/fixtures/czech-2.wav" \
  -F "format=json" | jq '{engine, lang: .language.code, conf: .language.confidence}'
```

### Unit Test

```python
import pytest

SPANISH_PODCAST_FILES = [
    "tests/fixtures/podcasts/spanish-286-pasando.mp3",
    "tests/fixtures/podcasts/spanish-291-cerditos.mp3",
    "tests/fixtures/podcasts/spanish-292-prestardinero.mp3",
]

@pytest.mark.parametrize("audio_file", SPANISH_PODCAST_FILES)
def test_spanish_podcasts_route_correctly(audio_file):
    """Spanish podcasts with English intros should route to Parakeet, not CTC."""
    audio, sr = load_audio(audio_file)
    result = transcribe(audio, sr)

    # Should NOT use English-only CTC
    assert result["engine"] != "tier1", \
        f"{audio_file}: routed to tier1 (English-only)"

    # Should detect Spanish (not English or Latin)
    assert result["language"]["code"] == "es", \
        f"{audio_file}: detected as {result['language']['code']}, expected 'es'"


def test_czech_routes_correctly():
    """Czech audio should route to Tier 2, not fall through to no engine."""
    audio, sr = load_audio("tests/fixtures/czech-2.wav")
    result = transcribe(audio, sr)

    # Should have an engine (not none)
    assert result["engine"] != "none", \
        "czech-2.wav: no engine assigned"

    # Should route to tier2 (Czech is supported)
    assert result["engine"] == "tier2", \
        f"czech-2.wav: routed to {result['engine']}, expected 'tier2'"

    # Should detect Czech
    assert result["language"]["code"] == "cs", \
        f"czech-2.wav: detected as {result['language']['code']}, expected 'cs'"
```
