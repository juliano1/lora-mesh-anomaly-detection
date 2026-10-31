"""
Pacotes congelados dos modelos online.

Um pacote guarda os modelos treinados, normalizadores, scaler e os scores de
calibração de τ de um OnlinePipeline, mais um manifesto JSON com identidade,
versões e SHA-256 do payload. Com ML_ONLINE_FREEZE_MODELS os mesmos modelos são
reutilizados em todos os blocos da campanha e sobrevivem a reinícios.

O payload é serializado com joblib (pickle) e carregá-lo executa código: o diretório
de pacotes deve ser gravável só pelo usuário do backend. O SHA-256 do manifesto detecta
corrupção, não autoria. Links simbólicos e nomes de payload fora do padrão são recusados.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PAYLOAD = "bundle.joblib"
MANIFEST = "manifest.json"


def _sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "sklearn", "torch", "joblib"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    return versions


def _feature_identity() -> dict:
    from ai.anomaly_definitions import FEATURE_ORDER
    from ai.feature_engineering import FEATURE_SCHEMA_VERSION
    return {"feature_schema_version": FEATURE_SCHEMA_VERSION, "feature_order": list(FEATURE_ORDER)}


def save_bundle(pipeline, network_id: int, base_dir: str, reason: str) -> dict:
    """Grava pacote + manifesto em <base_dir>/network_<id>/<bundle_id>/; devolve o manifesto."""
    import joblib
    bundle = pipeline.export_bundle()
    created = datetime.utcnow()
    root = Path(base_dir) / f"network_{network_id}"
    bundle_id = created.strftime("%Y%m%dT%H%M%S%fZ")
    folder = root / bundle_id
    folder.mkdir(parents=True, exist_ok=False)
    payload = folder / PAYLOAD
    joblib.dump(bundle, payload)
    status = pipeline.status()
    manifest = {
        "format": bundle["format"], "bundle_id": bundle_id, "network_id": network_id,
        "created_at": created.isoformat(), "reason": reason,
        "primary_id": bundle["primary_id"], "secondary_id": bundle["secondary_id"],
        "model_signature": bundle["model_signature"], "config": bundle["config"],
        **_feature_identity(),
        "tau_status": status["tau_status"], "tau_stats": status["tau_stats"],
        "calibration": status["calibration"],
        "primary": status["primary"], "secondary": status["secondary"],
        "versions": _versions(),
        "payload": PAYLOAD, "payload_sha256": _sha256(payload),
    }
    (folder / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    logger.info(f"pacote {bundle_id} salvo para a rede {network_id} ({reason})")
    return manifest


def load_latest_bundle(network_id: int, model_signature, base_dir: str) -> Optional[tuple[dict, dict]]:
    """(manifesto, pacote) mais recente compatível com a assinatura dos modelos e o
    vetor de features atuais; None se não houver. Payload adulterado é recusado."""
    import joblib
    from ai.online.pipeline import _jsonable
    root = Path(base_dir) / f"network_{int(network_id)}"
    if root.is_symlink() or not root.is_dir():
        return None
    wanted = _jsonable(model_signature)
    identity = _feature_identity()
    for folder in sorted((p for p in root.iterdir() if p.is_dir() and not p.is_symlink()), reverse=True):
        manifest_path, payload = folder / MANIFEST, folder / PAYLOAD
        if manifest_path.is_symlink() or payload.is_symlink() or not payload.is_file():
            logger.warning(f"pacote {folder.name} ignorado — arquivos ausentes ou links")
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict) or manifest.get("payload", PAYLOAD) != PAYLOAD:
            logger.warning(f"pacote {folder.name} ignorado — manifesto fora do formato")
            continue
        if manifest.get("model_signature") != wanted:
            continue
        if any(manifest.get(k) != v for k, v in identity.items()):
            logger.warning(f"pacote {folder.name} ignorado — vetor de features diferente")
            continue
        if _sha256(payload) != manifest.get("payload_sha256"):
            logger.error(f"pacote {folder.name} recusado — SHA-256 não confere")
            continue
        return manifest, joblib.load(payload)
    return None
