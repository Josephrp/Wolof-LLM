"""Private-only Hugging Face publication validation helpers."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".jinja",
    ".py",
    ".toml",
}
FORBIDDEN_PARTS = {
    ".env",
    ".git",
    ".venv",
    "__pycache__",
    "data",
    "datasets",
    "logs",
}
SECRET_PATTERNS = {
    "hugging_face_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "credential_assignment": re.compile(
        r"(?i)\b(password|passwd|token|secret|api[_-]?key)\s*[:=]\s*[\"']?[^\s\"']{8,}"
    ),
    "ephemeral_private_path": re.compile(r"/mnt/(?:volume|data)[^/\s]*/"),
}


def select_namespace(identity: dict, preferred: str = "Tonic") -> str:
    username = identity.get("name")
    if not isinstance(username, str) or not username:
        raise ValueError("HfApi.whoami() returned no authenticated username")
    for organization in identity.get("orgs", []):
        role = str(organization.get("roleInOrg", "")).lower()
        if organization.get("name") == preferred and role in {
            "admin",
            "write",
            "contributor",
        }:
            return preferred
    return username


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publishable_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"artifact directory does not exist: {folder}")
    files = [
        path
        for path in folder.rglob("*")
        if path.is_file() and not set(path.relative_to(folder).parts) & FORBIDDEN_PARTS
    ]
    if not files:
        raise ValueError("artifact directory contains no publishable files")
    return sorted(files)


def scan_for_secrets(folder: Path, files: Iterable[Path] | None = None) -> None:
    violations: list[str] = []
    for path in files or publishable_files(folder):
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 10 * 1024 * 1024:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                violations.append(f"{path.relative_to(folder)}:{name}")
    if violations:
        raise ValueError(f"secret/private-coordinate scan failed: {sorted(violations)}")


def build_checksums(folder: Path, files: Iterable[Path]) -> dict[str, dict[str, int | str]]:
    return {
        path.relative_to(folder).as_posix(): {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in files
    }
