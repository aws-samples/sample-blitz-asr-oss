import os, json, hashlib, secrets, logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from fastapi import Request, HTTPException

logger = logging.getLogger("blitz.security")
API_KEY_HEADER = "X-API-Key"
PUBLIC_ENDPOINTS = {"/health", "/metrics", "/docs", "/openapi.json", "/"}

@dataclass
class ClientInfo:
    client_id: str
    key_hash: str

@dataclass
class APIKeyAuth:
    enabled: bool = True
    keys: Dict[str, List[str]] = field(default_factory=dict)
    _key_lookup: Dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_env(cls):
        enabled = os.environ.get("BLITZ_AUTH_ENABLED", "true").lower() in ("true", "1", "yes")
        keys_raw = os.environ.get("BLITZ_API_KEYS", "")
        if not enabled:
            logger.info("API key authentication DISABLED")
            return cls(enabled=False)
        if not keys_raw:
            logger.warning("No API keys configured (BLITZ_API_KEYS not set). Auth DISABLED.")
            return cls(enabled=False)
        keys = {}
        try:
            parsed = json.loads(keys_raw)
            if isinstance(parsed, dict):
                keys = {k: (v if isinstance(v, list) else [v]) for k, v in parsed.items()}
            elif isinstance(parsed, list):
                keys = {"default": parsed}
        except json.JSONDecodeError:
            key_list = [k.strip() for k in keys_raw.split(",") if k.strip()]
            keys = {"default": key_list}
        key_lookup = {}
        total = 0
        for client_id, client_keys in keys.items():
            for key in client_keys:
                key_lookup[hashlib.sha256(key.encode()).hexdigest()] = client_id
                total += 1
        logger.info(f"API key auth ENABLED: {len(keys)} client(s), {total} key(s)")
        instance = cls(enabled=True, keys=keys)
        instance._key_lookup = key_lookup
        return instance

    def authenticate(self, api_key: Optional[str]) -> Optional[ClientInfo]:
        if not self.enabled:
            return None
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing API key. Provide X-API-Key header.", headers={"WWW-Authenticate": "ApiKey"})
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        client_id = self._key_lookup.get(key_hash)
        if not client_id:
            logger.warning(f"Invalid API key attempt (hash prefix: {key_hash[:8]})")
            raise HTTPException(status_code=403, detail="Invalid API key.")
        return ClientInfo(client_id=client_id, key_hash=key_hash[:16])

    @staticmethod
    def generate_key(prefix="blitz"):
        return f"{prefix}_{secrets.token_urlsafe(32)}"

_auth_manager = None
def get_auth_manager():
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = APIKeyAuth.from_env()
    return _auth_manager

async def get_current_client(request: Request) -> Optional[ClientInfo]:
    if request.url.path in PUBLIC_ENDPOINTS:
        return None
    auth = get_auth_manager()
    api_key = request.headers.get(API_KEY_HEADER)
    return auth.authenticate(api_key)
