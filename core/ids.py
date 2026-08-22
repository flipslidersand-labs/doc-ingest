"""Deterministic UUID generation for Qdrant point IDs."""
import hashlib
import uuid


def make_id(key: str) -> str:
    """Return a deterministic UUID string derived from key via SHA-256.

    Uses the first 16 bytes of the digest — 128-bit entropy with no modular
    truncation, eliminating the birthday-paradox collision risk of % 2^63.
    """
    digest = hashlib.sha256(key.encode()).digest()
    return str(uuid.UUID(bytes=digest[:16]))
