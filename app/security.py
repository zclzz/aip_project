"""API-key authentication and a small in-memory rate limiter."""
from __future__ import annotations

import secrets
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

from .config import settings


async def require_api_key(request: Request) -> None:
    """Accept the key via ``X-API-Key: <key>`` or ``Authorization: Bearer <key>``."""
    key = request.headers.get("x-api-key", "")
    if not key:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            key = auth[7:].strip()
    # Constant-time compare so response timing does not leak key prefixes.
    if not key or not secrets.compare_digest(key.encode(), settings.api_key.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Send it as 'X-API-Key' or 'Authorization: Bearer'.",
            headers={"WWW-Authenticate": "Bearer"},
        )


class RateLimiter:
    """Sliding-window limiter keyed by client address. Good enough for a single-process service."""

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client_id: str) -> tuple[bool, float]:
        """Return (allowed, seconds_until_slot_frees)."""
        if self.limit <= 0:
            return True, 0.0
        now = time.monotonic()
        with self._lock:
            hits = self._hits[client_id]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, max(0.0, self.window - (now - hits[0]))
            hits.append(now)
            return True, 0.0


rate_limiter = RateLimiter(settings.rate_limit_per_minute)


async def enforce_rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    allowed, retry_after = rate_limiter.check(client)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit of {settings.rate_limit_per_minute} requests/minute exceeded.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )
