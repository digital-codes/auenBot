#!/usr/bin/env python
"""
Sealed state tokens for the botCtl API.

Ported from the CodeforKarlsruhe/auenBot fork (botQa.py, seal/open
half only; the QaEngine slot-filling FSM was not ported). The
conversation context is sealed into an opaque token the frontend
carries between requests: the client can neither read nor modify it.

Token format:

    "v1." + base64url( nonce(12) || ciphertext )

AES-256-GCM provides authenticity (tamper detection). The key is
derived per session from the server secret via HMAC-SHA256, and the
session id is bound as associated data, so a token is useless for any
other session and any modification fails the auth tag.

The server secret is read from the STATE_TOKEN_SECRET environment
variable at import time. When unset, a random secret is generated and
a warning is printed: tokens will not survive a restart.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from typing import Any, Dict, Optional, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# ---- Key derivation (session-bound) ----
# DO NOT use session_id alone as a key.
# Use a server secret + session_id so tokens can't be forged client-side.

if os.environ.get("STATE_TOKEN_SECRET") is None:
    print(
        "Warning: using random server secret for state tokens "
        "(tokens won't be valid across restarts)"
    )
    SERVER_SECRET: Optional[bytes] = os.urandom(32)  # 32+ bytes recommended
else:
    SERVER_SECRET = os.environ["STATE_TOKEN_SECRET"].encode("utf-8")


def _derive_key(session_id: str, secret: bytes) -> bytes:
    """Derive the AES-256 key for a session from the server secret.

    Simple KDF via HMAC-SHA256 (OK for deriving a symmetric key here).
    """
    return hmac.new(secret, session_id.encode("utf-8"), hashlib.sha256).digest()


def seal_state(state: Dict[str, Any], session_id: str, *, secret: Optional[bytes] = None) -> str:
    """Seal a context dict into an opaque state token.

    ``secret`` defaults to the module-level server secret derived from
    the STATE_TOKEN_SECRET environment variable (random when unset).
    Passing an explicit secret lets callers (e.g. the API test harness)
    derive keys without depending on import-time environment reads.
    """
    key = _derive_key(session_id, secret if secret is not None else SERVER_SECRET)
    aesgcm = AESGCM(key)

    nonce = os.urandom(12)
    plaintext = json.dumps(state, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    # bind token to session_id as associated data (extra safety)
    aad = session_id.encode("utf-8")

    ciphertext = aesgcm.encrypt(nonce, plaintext, aad)  # includes auth tag
    blob = nonce + ciphertext
    token = base64.urlsafe_b64encode(blob).decode("ascii").rstrip("=")
    return f"v1.{token}"


def open_state(token: str, session_id: str, *, secret: Optional[bytes] = None) -> Dict[str, Any]:
    """Open a state token back into its context dict.

    Raises ValueError (unsupported version, bad payload) or
    cryptography.exceptions.InvalidTag (tampered, wrong session, wrong
    secret) - callers translate that into an HTTP 400.
    """
    if not token.startswith("v1."):
        raise ValueError("Unsupported state token version")

    b64 = token[3:]
    # restore padding
    pad = "=" * (-len(b64) % 4)
    blob = base64.urlsafe_b64decode(b64 + pad)

    nonce, ciphertext = blob[:12], blob[12:]
    key = _derive_key(session_id, secret if secret is not None else SERVER_SECRET)
    aesgcm = AESGCM(key)

    aad = session_id.encode("utf-8")
    plaintext = aesgcm.decrypt(nonce, ciphertext, aad)  # raises if tampered/wrong session
    return json.loads(plaintext.decode("utf-8"))


# ---- Per-event sealing (kept for internal reuse, e.g. by QaEngine) ----

def seal_state_tuple(state: Tuple[Any, ...], session_id: str, *, secret: Optional[bytes] = None) -> str:
    """Seal a tuple of values (order preserved) into a state token."""
    key = _derive_key(session_id, secret if secret is not None else SERVER_SECRET)
    aesgcm = AESGCM(key)

    nonce = os.urandom(12)
    payload = {"v": list(state)}  # preserve order
    plaintext = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    aad = session_id.encode("utf-8")  # bind to session
    ciphertext = aesgcm.encrypt(nonce, plaintext, aad)

    blob = nonce + ciphertext
    token = base64.urlsafe_b64encode(blob).decode("ascii").rstrip("=")
    return f"v1.{token}"


def open_state_tuple(token: str, session_id: str, *, secret: Optional[bytes] = None) -> Tuple[Any, ...]:
    """Open a per-event token sealed by ``seal_state_tuple``."""
    if not token.startswith("v1."):
        raise ValueError("Unsupported state token version")

    b64 = token[3:]
    pad = "=" * (-len(b64) % 4)
    blob = base64.urlsafe_b64decode(b64 + pad)

    nonce, ciphertext = blob[:12], blob[12:]
    key = _derive_key(session_id, secret if secret is not None else SERVER_SECRET)
    aesgcm = AESGCM(key)

    aad = session_id.encode("utf-8")
    plaintext = aesgcm.decrypt(nonce, ciphertext, aad)

    payload = json.loads(plaintext.decode("utf-8"))
    if not isinstance(payload, dict) or "v" not in payload or not isinstance(payload["v"], list):
        raise ValueError("Bad state payload format")

    return tuple(payload["v"])
