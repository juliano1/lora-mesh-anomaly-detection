"""
Importador: pacote de dados exportado → SQLite com o esquema atual.

Confere hashes SHA-256, tamanhos e linhas contra o manifesto do próprio pacote
antes de ler (detecta corrupção ou pacote incompleto, não adulteração intencional:
quem altera um arquivo pode regravar o manifesto); reconstrói as tabelas com o mesmo
escritor da materialização (`write_rows`); confere `PRAGMA foreign_key_check`;
grava ao lado do banco `<nome>.dataset.json` com o manifesto de origem e os
parâmetros de interpretação, que recálculos posteriores devem reaplicar
(`resolve_for_computation`). O SQLite resultante pode ser usado como snapshot
para novos recálculos e exportações.
"""
from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import zipfile
from artifact_versions import version_matches
from datetime import datetime
from pathlib import Path

from dataset_export.codec import decode, read_csv
from dataset_export.interpretation import InterpretationParams
from dataset_export.package import EXTRA_COLUMNS, RAW_FILES, PackageError, verify_package, build_schema
from dataset_export.provenance import code_revision
from dataset_export.request import iso_z
from dataset_export.slice import create_target, finalize_sqlite, table, write_rows

IMPORT_FORMAT = "import-v1"


def sidecar_path(db_path) -> Path:
    return Path(db_path).with_suffix(".dataset.json")


def read_table_csv(fh, table_name: str, source_id=None):
    columns = list(table(table_name).columns)
    reader = read_csv(fh)
    header = next(reader, None)
    expected = [c.name for c in columns] + list(EXTRA_COLUMNS)
    if header != expected:
        raise PackageError(f"{table_name}: cabeçalho inesperado {header}")
    for record in reader:
        if len(record) != len(expected):
            raise PackageError(f"{table_name}: quantidade de colunas inválida")
        if (source_id is not None and record[-2] != source_id) or record[-1] not in ("requested", "context"):
            raise PackageError(f"{table_name}: source_id ou record_role inválido")
        yield {c.name: decode(value, c) for c, value in zip(columns, record)}


def import_dataset(package, out_db) -> dict:
    out_db = Path(out_db)
    if out_db.exists() or sidecar_path(out_db).exists():
        raise FileExistsError(f"{out_db} (ou seu .dataset.json) já existe")
    manifest, root = verify_package(package)
    if manifest["export_kind"] not in ("raw", "full"):
        raise PackageError(f"pacote {manifest['export_kind']!r} não contém dados brutos")
    # O schema deve ser interpretável pelo código instalado, sem descartar colunas.
    with zipfile.ZipFile(package) as zf:
        incoming_schema = json.loads(zf.read(f"{root}/schema.json"))
    current_schema = build_schema()
    if not version_matches(incoming_schema.get("format"), current_schema["format"]):
        raise PackageError("schema de formato desconhecido")
    for name in RAW_FILES:
        old = incoming_schema.get("files", {}).get(name, {}).get("columns", [])
        current = current_schema["files"][name]["columns"]
        signature = lambda cols: [(c["name"], c["type"], c["nullable"]) for c in cols]
        if signature(old) != signature(current):
            raise PackageError(f"{name}: schema incompatível com o código instalado")
    out_db.parent.mkdir(parents=True, exist_ok=True)
    own_sidecar = False
    with tempfile.TemporaryDirectory(prefix="dataset_import_", dir=out_db.parent) as directory:
        staged_db = Path(directory) / "dataset.db"
        engine = create_target(staged_db)
        try:
            with zipfile.ZipFile(package) as zf:
                for file_name, table_name in RAW_FILES.items():
                    with zf.open(f"{root}/{file_name}") as raw:
                        fh = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                        written = write_rows(engine, table_name, read_table_csv(
                            fh, table_name, manifest["interpretation"]["source_id"]))
                    if written != manifest["files"][file_name]["rows"]:
                        raise PackageError(f"{file_name}: linhas importadas diferem do manifesto")
            violations = finalize_sqlite(engine)
            if violations:
                raise PackageError(f"foreign_key_check: {len(violations)} violação(ões)")
        finally:
            engine.dispose()
        info = {
            "format": IMPORT_FORMAT,
            "imported_at": iso_z(datetime.utcnow()),
            "database": out_db.name,
            "interpretation": manifest["interpretation"],
            "importer_code": code_revision(),
            "source_manifest": manifest,
        }
        staged_sidecar = Path(directory) / "dataset.json"
        staged_sidecar.write_text(json.dumps(info, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        try:
            os.link(staged_sidecar, sidecar_path(out_db))
            own_sidecar = True
            os.link(staged_db, out_db)
        except BaseException:
            if own_sidecar:
                sidecar_path(out_db).unlink(missing_ok=True)
            raise
        return info


def load_interpretation(db_path) -> InterpretationParams:
    """Parâmetros de interpretação gravados na importação de `db_path`."""
    data = json.loads(sidecar_path(db_path).read_text(encoding="utf-8"))
    return InterpretationParams.from_dict(data["interpretation"])
