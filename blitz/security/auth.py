"""
API Key authentication for Blitz ASR.

Supports:
- Header-based API keys (X-API-Key)
- Multiple clients with individual keys
- Key rotation (multiple valid keys per client)
- Public endpoints exemption (/health, /metrics)
- Optional: disabled entirely via BLITZ_AUTH_ENABLED=false

Keys are stored in environment variable BLITZ_API_KEYS as a JSON string:
    BLITZ_API_KEYS='{"client1": ["key1", "key2"], "client2": ["key3"]}'

Or as a simple comma-separated list (all keys map to "default" client):
    BLITZ_API_KEYS='sk-abc123,sk-def456'
"""

import os
import json
import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger("blitz.security")

# Header name for API key
API_KEY_HEADER = "X-API-Key"

# Endpoints that don't require authentication
PUBLIC_ENDPOINTS: Set[str] = {"/health", "/metrics", "/docs", "/openapi.json", "/"}


@dataclass
class ClientInfo:
    """Information about an authenticated client."""
    client_id: str
    key_hash: str  # SHA-256 hash of the key used (for audit logging)


@dataclass
class APIKeyAuth:
    """
    API Key authentication manager.

    Supports two formats for BLITZ_API_KEYS env var:
    1. JSON object: {"client_name": ["key1", "key2"], ...}
    2. Simple list: "key1,key2,key3" (all mapped to "default" client)
    """

    enabled: bool = True
    keys: Dict[str, List[str]] = field(default_factory=dict)
    # Reverse lookup: key_hash -> client_id
    _key_lookup: Dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_env(cls) -> "APIKeyAuth":
        """Create auth manager from environment variables."""
        enabled = os.environ.get("BLITZ_AUTH_ENABLED", "true").lower() in ("true", "1", "yes")
        keys_raw = os.environ.get("BLITZ_API_KEYS", "")

        if not enabled:
            logger.info("API key authentication DISABLED (BLITZ_AUTH_ENABLED=false)")
            return cls(enabled=False)

        if not keys_raw:
            logger.warning(
                "No API keys configured (BLITZ_API_KEYS not set). "
                "Authentication is DISABLED. Set BLITZ_API_KEYS to enable."
            )
            return cls(enabled=False)

        # Try parsing as JSON first
        keys: Dict[str, List[str]] = {}
        try:
            parsed = json.loads(keys_raw)
            if isinstance(parsed, dict):
                keys = {k: (v if isinstance(v, list) else [v]) for k, v in parsed.items()}
            elif isinstance(parsed, list):
                keys = {"default": parsed}
        except json.JSONDecodeError:
            # Treat as comma-separated list
            key_list = [k.strip() for k in keys_raw.split(",") if k.strip()]
            keys = {"default": key_list}

        # Build reverse lookup (hash -> client_id)
        key_lookup = {}
        total_keys = 0
        for client_id, client_keys in keys.items():
            for key in client_keys:
                key_hash = hashlib.sha256(key.encode()).hexdigest()
                key_lookup[key_hash] = client_id
                total_keys += 1

        logger.info(
            f"API key authentication ENABLED: {len(keys)} client(s), {total_keys} key(s)"
        )

        instance = cls(enabled=True, keys=keys)
        instance._key_lookup = key_lookup
        return instance

    def authenticate(self, api_key: Optional[str]) -> Optional[ClientInfo]:
        """
        Authenticate an API key.

        Returns ClientInfo if valid, None if auth is disabled, raises HTTPException if invalid.
        """
        if not self.enabled:
            return None

        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="Missing API key. Provide X-API-Key header.",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        client_id = self._key_lookup.get(key_hash)

        if not client_id:
            logger.warning(f"Invalid API key attempt (hash prefix: {key_hash[:8]}...)")
            raise HTTPException(
                status_code=403,
                detail="Invalid API key.",
            )

        return ClientInfo(client_id=client_id, key_hash=key_hash[:16])

    @staticmethod
    def generate_key(prefix: str = "blitz") -> str:
        """Generate a new random API key."""
        random_part = secrets.token_urlsafe(32)
        return f"{prefix}_{random_part}"


# Singleton auth manager
_auth_manager: Optional[APIKeyAuth] = None


def get_auth_manager() -> APIKeyAuth:
    """Get or create the global auth manager."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = APIKeyAuth.from_env()
    return _auth_manager


async def get_current_client(request: Request) -> Optional[ClientInfo]:
    """
    FastAPI dependency that authenticates the request.

    Usage in endpoints:
        @app.post("/transcribe")
        async def transcribe(client: ClientInfo = Depends(get_current_client)):
            ...

    Skip auth for public endpoints automatically.
    """
    # Skip auth for public endpoints
    if request.url.path in PUBLIC_ENDPOINTS:
        return None

    auth = get_auth_manager()
    api_key = request.headers.get(API_KEY_HEADER)
    return auth.authenticate(api_key)
