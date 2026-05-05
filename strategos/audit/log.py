"""Append-only audit log with HMAC-SHA256 hash chaining."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from strategos.types import AuditEvent

_GENESIS = "0" * 64


def _canonical(payload: Any) -> bytes:
    """Stable JSON encoding for hashing — sorted keys, no whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).isoformat()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


class AuditLog:
    """Append-only JSONL audit log with HMAC-SHA256 chaining."""

    def __init__(self, path: str | None = None, secret_key: bytes | None = None) -> None:
        self.path: Path | None = Path(path) if path else None
        if secret_key is None:
            warnings.warn(
                "AuditLog: no secret_key supplied; generating an ephemeral key. "
                "This is fine for tests but logs CANNOT be re-verified across processes.",
                stacklevel=2,
            )
            secret_key = secrets.token_bytes(32)
        if not isinstance(secret_key, (bytes, bytearray)):
            raise TypeError("secret_key must be bytes")
        self._key: bytes = bytes(secret_key)
        self._mem: list[dict] = []
        self._seq: int = 0
        self._last_hash: str = _GENESIS

        if self.path is not None and self.path.exists():
            self._load_existing()

    def _compute_hash(
        self,
        seq: int,
        timestamp: str,
        kind: str,
        payload: dict,
        prev_hash: str,
    ) -> str:
        body = {
            "seq": seq,
            "timestamp": timestamp,
            "kind": kind,
            "payload": payload,
            "prev_hash": prev_hash,
        }
        msg = bytes.fromhex(prev_hash) + _canonical(body)
        return hmac.new(self._key, msg, hashlib.sha256).hexdigest()

    def _load_existing(self) -> None:
        assert self.path is not None
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._mem.append(rec)
                self._seq = max(self._seq, int(rec["seq"]) + 1)
                self._last_hash = rec["hash"]

    def append(self, kind: str, payload: dict) -> AuditEvent:
        if not isinstance(kind, str) or not kind:
            raise ValueError("kind must be a non-empty string")
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")

        ts = datetime.now(timezone.utc).isoformat()
        seq = self._seq
        prev = self._last_hash
        canon_payload = json.loads(_canonical(payload).decode("utf-8"))
        h = self._compute_hash(seq, ts, kind, canon_payload, prev)
        event = AuditEvent(seq=seq, timestamp=ts, kind=kind, payload=canon_payload, prev_hash=prev, hash=h)

        rec = asdict(event)
        self._mem.append(rec)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, separators=(",", ":")) + os.linesep)

        self._seq += 1
        self._last_hash = h
        return event

    def __len__(self) -> int:
        return len(self._mem)

    def __iter__(self) -> Iterator[dict]:
        return iter(list(self._mem))

    def tail(self, n: int) -> list[dict]:
        if n <= 0:
            return []
        return list(self._mem[-n:])

    def all(self) -> list[dict]:
        return list(self._mem)

    def verify_chain(self) -> bool:
        """Walk every event and verify the HMAC chain."""
        prev = _GENESIS
        for i, rec in enumerate(self._mem):
            try:
                seq = int(rec["seq"])
                ts = str(rec["timestamp"])
                kind = str(rec["kind"])
                payload = rec["payload"]
                prev_hash = str(rec["prev_hash"])
                stored = str(rec["hash"])
            except (KeyError, ValueError, TypeError):
                return False
            if seq != i:
                return False
            if prev_hash != prev:
                return False
            recomputed = self._compute_hash(seq, ts, kind, payload, prev_hash)
            if not hmac.compare_digest(recomputed, stored):
                return False
            prev = stored
        return True


__all__ = ["AuditLog"]
