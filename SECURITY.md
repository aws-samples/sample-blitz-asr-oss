# Security Policy

## Reporting a Vulnerability

Please report suspected security vulnerabilities privately via GitHub
Security Advisories (**Security** tab → **Report a vulnerability**) rather
than opening a public issue.

We ask that you give us a reasonable window to investigate and publish
a fix before any public disclosure.

## Deployment Security Model

Blitz is a reference ASR service. A few deployment characteristics are
intentional and should be accounted for when running it in untrusted
environments:

### No built-in authentication

The FastAPI server in `blitz/server.py` does not authenticate incoming
requests. Anyone who can reach the HTTP port can submit audio, consume
GPU, and read back transcripts.

**Recommendation:** put Blitz behind an authenticating reverse proxy
(API gateway, ALB with Cognito/OIDC, nginx with mTLS, etc.) before
exposing it beyond `localhost` or a private subnet.

### Listens on `0.0.0.0` by default

The container/uvicorn invocation binds to all interfaces so it works
inside Docker and ECS/EKS. Bandit's `B104` advisory flags this on the
raw process.

**Recommendation:** in VM/bare-metal deployments that are not inside a
container network, override the bind host (e.g., `--host 127.0.0.1`) and
front the service with a proxy.

### Model supply chain

Several subsystems download model weights from HuggingFace Hub and
PyTorch Hub at first run:

- `blitz/models/whisper.py` → `openai/whisper-*`
- `blitz/models/parakeet.py` → `nvidia/parakeet-*` (NeMo)
- `blitz/models/parakeet_onnx.py` → `grikdotnet/parakeet-tdt-0.6b-fp16`, `istupakov/parakeet-tdt-0.6b-v3-onnx`
- `blitz/services/vad.py` → `snakers4/silero-vad`

By default the latest revision on each repo's default branch is used.
For reproducible, auditable model loads, set the corresponding
environment variables to a known-good commit SHA:

| Env var | Applies to |
| --- | --- |
| `TIER1_MODEL_REVISION` | Tier 1 Parakeet CTC |
| `TIER2_MODEL_REVISION` | Tier 2 Parakeet 0.6B (NeMo or ONNX) |
| `TIER3_MODEL_REVISION` | Tier 3 Whisper (primary) |
| `WHISPER_TINY_REVISION` | Whisper tiny (optional lang-detect model) |
| `SILERO_VAD_REVISION` | Silero VAD (torch.hub) |

Empty string (the default) means "latest" and is convenient for local
development. Production deployments should pin all of them.

### Uploaded file handling

`/transcribe` accepts arbitrary audio uploads, which are buffered to a
`tempfile.NamedTemporaryFile` and (for non-native formats) run through
`ffmpeg`. `ffmpeg` is invoked with a fixed argv (no shell), and the
user-supplied filename is only reflected back in response headers after
sanitization (see `_safe_attachment_filename` in `blitz/server.py`).
`ffmpeg` itself must be present on `$PATH`; keep it up to date for
upstream CVE fixes.

## Known `bandit` advisories we accept

- **B104** (bind to `0.0.0.0`) — documented above.
- **B404 / B603 / B607** (`subprocess.run` with `ffmpeg`) — invoked
  with list-form argv, no `shell=True`, and fixed flags. User input
  reaches the process only as the input/output file paths, which we
  control.
- **B615 / B614** (HuggingFace / `torch.load` without pinning) — the
  ability to pin is exposed via the env vars above. We do not enforce
  it by default so first-time users can get running without lookups.
