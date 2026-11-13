"""Small Hub helpers for Qwen family GGUF/ONNX publication."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

TEXT_SUFFIXES = {".json", ".md", ".txt", ".yaml", ".yml", ".jinja"}
SECRET_PATTERNS = {
    "hugging_face_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_checksums(folder: Path, files: Iterable[Path]) -> dict[str, dict[str, int | str]]:
    return {
        path.relative_to(folder).as_posix(): {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in files
    }


def scan_for_secrets(folder: Path, files: Iterable[Path]) -> None:
    violations: list[str] = []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 10 * 1024 * 1024:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                violations.append(f"{path.relative_to(folder)}:{name}")
    if violations:
        raise ValueError(f"secret scan failed: {sorted(violations)}")


def select_namespace(whoami: dict, preferred: str = "Tonic") -> str:
    name = whoami.get("name") or whoami.get("fullname")
    if name == preferred:
        return preferred
    orgs = {org.get("name") for org in whoami.get("orgs", []) if isinstance(org, dict)}
    if preferred in orgs:
        return preferred
    if name:
        return str(name)
    raise RuntimeError("cannot resolve Hub namespace")
