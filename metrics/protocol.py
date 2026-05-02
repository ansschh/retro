"""Protocol loader and SHA computation.

Every leaderboard row records protocol_version + protocol_sha so reruns are
traceable and rankings can't drift silently if the protocol changes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "protocol.yaml"
)


def _resolve_path(path: str | Path | None) -> Path:
    return Path(path) if path is not None else _DEFAULT_PROTOCOL_PATH


def load_protocol(path: str | Path | None = None) -> dict[str, Any]:
    """Load `configs/protocol.yaml` (or the file at `path`) as a dict."""
    with open(_resolve_path(path)) as f:
        return yaml.safe_load(f)


def protocol_sha(path: str | Path | None = None) -> str:
    """SHA-256 of the protocol file (full 64-char hex). Use `[:12]` for
    leaderboard column display."""
    with open(_resolve_path(path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
