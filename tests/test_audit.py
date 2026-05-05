"""Audit-log chain integrity and tamper detection."""
from __future__ import annotations

import json

import pytest

from strategos import AuditLog


def _key() -> bytes:
    return b"\x01" * 32


def test_chain_intact_in_memory():
    log = AuditLog(secret_key=_key())
    log.append("decision", {"agent": "a", "pos": 0.5})
    log.append("verdict", {"ok": True})
    log.append("execution", {"weights": {"AAPL": 0.5}})
    assert log.verify_chain() is True
    assert len(log) == 3


def test_chain_detects_payload_tampering():
    log = AuditLog(secret_key=_key())
    log.append("decision", {"agent": "a", "pos": 0.5})
    log.append("verdict", {"ok": True})
    log._mem[0]["payload"]["pos"] = 0.99
    assert log.verify_chain() is False


def test_chain_detects_hash_tampering():
    log = AuditLog(secret_key=_key())
    log.append("decision", {"agent": "a"})
    log.append("verdict", {"ok": True})
    log._mem[1]["hash"] = "0" * 64
    assert log.verify_chain() is False


def test_chain_detects_reordering():
    log = AuditLog(secret_key=_key())
    log.append("decision", {"i": 0})
    log.append("verdict", {"i": 1})
    log._mem[0], log._mem[1] = log._mem[1], log._mem[0]
    assert log.verify_chain() is False


def test_persists_to_disk_and_reloads(tmp_path):
    p = tmp_path / "audit.jsonl"
    key = _key()
    log = AuditLog(path=str(p), secret_key=key)
    log.append("decision", {"agent": "a"})
    log.append("execution", {"w": {"AAPL": 0.4}})
    assert log.verify_chain() is True

    log2 = AuditLog(path=str(p), secret_key=key)
    assert len(log2) == 2
    assert log2.verify_chain() is True


def test_disk_tamper_breaks_chain(tmp_path):
    p = tmp_path / "audit.jsonl"
    key = _key()
    log = AuditLog(path=str(p), secret_key=key)
    log.append("decision", {"agent": "a", "pos": 0.5})
    log.append("execution", {"weights": {"AAPL": 0.5}})

    lines = p.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["payload"]["pos"] = 0.99
    lines[0] = json.dumps(rec, separators=(",", ":"))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log2 = AuditLog(path=str(p), secret_key=key)
    assert log2.verify_chain() is False


def test_tail_returns_last_n():
    log = AuditLog(secret_key=_key())
    for i in range(5):
        log.append("decision", {"i": i})
    last = log.tail(2)
    assert len(last) == 2
    assert last[-1]["payload"]["i"] == 4


def test_no_secret_key_warns():
    with pytest.warns(UserWarning):
        AuditLog()
