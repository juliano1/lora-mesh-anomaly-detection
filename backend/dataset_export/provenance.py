"""Proveniência: hashes, revisão do código e dependências."""
from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent if BACKEND_ROOT.name == "backend" else BACKEND_ROOT

# Fontes cuja revisão identifica o formato/semântica do pacote.
RELEVANT_SOURCES = (
    "backend/dataset_export/request.py",
    "backend/dataset_export/codec.py",
    "backend/dataset_export/snapshot.py",
    "backend/dataset_export/slice.py",
    "backend/dataset_export/package.py",
    "backend/dataset_export/interpretation.py",
    "backend/dataset_export/importer.py",
    "backend/ai/feature_engineering.py",
    "backend/ai/anomaly_definitions.py",
    "backend/ai/reference_policy.py",
    "backend/ai/sequence_builder.py",
    "scripts/export_dataset.py",
    "scripts/import_dataset.py",
    "backend/models.py",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_revision() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def source_hashes() -> dict:
    hashes = {}
    for rel in RELEVANT_SOURCES:
        path = REPO_ROOT / rel
        if not path.is_file() and rel.startswith("backend/"):
            path = BACKEND_ROOT / rel.removeprefix("backend/")
        if path.is_file():
            hashes[rel] = sha256_file(path)
    return hashes


def dependency_versions() -> dict:
    import sqlalchemy
    import sqlmodel
    return {"python": platform.python_version(), "sqlalchemy": sqlalchemy.__version__,
            "sqlmodel": sqlmodel.__version__}
