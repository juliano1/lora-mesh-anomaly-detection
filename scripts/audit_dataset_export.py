#!/usr/bin/env python
"""
Auditoria de reconciliação de um pacote de exportação.

Reconcilia o ZIP contra (1) o banco de origem — conjuntos de IDs, papéis e valores,
com as regras do exportador para grade, contexto e recorte reimplementadas
aqui, independentes de dataset_export.request/slice —, (2) o extrator do projeto
aplicado diretamente à origem (features) e (3) a documentação embutida (schema,
manifesto, README), e reimporta o pacote para conferir `PRAGMA foreign_key_check`.
Grava um JSON com uma entrada por verificação. Não altera banco nem pacote: a
origem é aberta somente leitura e a importação usa um diretório temporário.

Uso:
  audit_dataset_export.py --package X.zip --sqlite origem.db --out VALIDACAO.json
      [--expect-source-id ID] [--expect-network-id N] [--expect-kind raw|features|full]
      [--expect-start T] [--expect-end T] [--expect-window S] [--expect-step S]
      [--expect-nodes 1,2,3|all] [--expect-grid-origin T]
      [--expect-tracked-since T|none] [--expect-margin MIN] [--expect-campaign-id ID|none]
      [--expect-commit REV] [--allow-dirty] [--no-import-check]
      [--compare Y.zip --compare-out COMPARE.json]
Instantes (T) em ISO 8601 com Z ou offset. Sem --expect-*, `request:matches_expected`
fica `info`. `manifest:code` exige commit do pacote = --expect-commit (padrão: HEAD do
auditor) e dirty=false (--allow-dirty relaxa só o dirty).
Saída: 0 sem falhas; 1 com falha (VALIDACAO.json gravado); 2 erro de uso ou
pacote/banco ilegível.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import event, select  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402
from sqlmodel import Session  # noqa: E402

from dataset_export.codec import decode, encode, kind as codec_kind, read_csv  # noqa: E402
from dataset_export.interpretation import InterpretationParams  # noqa: E402
from dataset_export.package import LIMITATIONS, PackageError, verify_package  # noqa: E402
from dataset_export.provenance import RELEVANT_SOURCES  # noqa: E402
from dataset_export.slice import table  # noqa: E402
from dataset_export.snapshot import build_source_engine  # noqa: E402

SCIENTIFIC_KEYS = ("export_kind", "campaign_id", "request", "resolved", "nodes", "networks",
                   "simulation_runs", "tables", "slice_rows", "features", "interpretation",
                   "validation", "code", "source_hashes", "limitations", "labels_status")
VARIABLE_KEYS = ("dataset_id", "package_root", "created_at", "timings_seconds", "timing_scope",
                 "source", "dependencies")
# Definidos aqui, não importados do gerador.
RAW_TABLES = {"telemetry.csv": "telemetry", "neighbors.csv": "neighbor", "nodes.csv": "node",
              "networks.csv": "network", "simulation_runs.csv": "simulation_run"}
FEATURE_FILES = {"features.csv", "coverage.csv"}
COMMON_FILES = {"schema.json", "README.md", "events_template.csv", "manifest.json"}
EXTRA_COLUMNS = ("source_id", "record_role")
EVENTS_HEADER = ["event_id", "campaign_id", "node_id", "event_type", "origin",
                 "start_at_utc", "end_at_utc", "observer", "evidence", "notes"]
RAW_KEY = {"telemetry": "id", "neighbor": "id", "node": "node_id", "network": "id", "simulation_run": "id"}
# Ordem documentada em schema.json (conventions.ordering).
RAW_ORDER = {"telemetry": ("node_id", "timestamp", "id"), "neighbor": ("node_id", "timestamp", "id"),
             "node": ("node_id",), "network": ("id",), "simulation_run": ("node_id", "started_at", "id")}
COUNT_COLUMNS = ("neighbor_record_count", "distinct_neighbor_count", "telemetry_record_count",
                 "reverse_record_count")
EXTRACTION_POLICY = "all_origins_v1"      # v1 preserva todas as origens
LABELS_STATUS = "not_provided"
LEGACY_PREFIX = "legacy_node:"
FEATURE_STATUSES = ["valid", "insufficient_data", "invalid_numeric", "error"]
ORIGIN_STATUSES = ["real", "synthetic", "mixed", "unknown", "no_data"]
ENUMS = {("features.csv", "feature_status"): FEATURE_STATUSES,
         ("features.csv", "origin_status"): ORIGIN_STATUSES}
SCHEMA_TYPES = {"integer", "float", "boolean", "datetime_utc", "string", "json_array", "json_object"}
CODEC_TO_SCHEMA = {"integer": "integer", "float": "float", "boolean": "boolean",
                   "datetime": "datetime_utc", "string": "string"}
INTEGER_TEXT = re.compile(r"-?(0|[1-9][0-9]*)")
# Cabeçalho do README do pacote (package.py: _README_HEAD e KIND_LABELS).
KIND_LABELS = {"raw": "Bruto", "features": "Features por janela", "full": "Completo"}
CASE_NAMES = ("real", "unknown", "mixed", "synthetic", "reverse_synthetic", "reverse_simulation",
              "margin_only", "legacy")
ORIGIN_CASES_TEXT = (
    "Janelas por caso de origem/simulação (0 = recorte não contém o caso). reverse_synthetic: linhas "
    "sintéticas sem simulação do próprio nó; reverse_simulation: simulação só de nó fonte (enlace "
    "reverso), sem linha sintética; margin_only: sem linha sintética, excluída por simulation_overlap "
    "e só intersecta com a margem; legacy: intervalo Node.sim_* sem SimulationRun do mesmo nó na "
    "janela. Exemplos: (nó, fim, exclusão, simulações)")


class Audit:
    def __init__(self):
        self.checks = []
        self.compared = {}  # quantidade efetivamente comparada por verificação (evita aprovação vazia)

    def add(self, cid, description, expected, observed, ok=None, status=None):
        if status is None:
            status = "pass" if (ok if ok is not None else expected == observed) else "fail"
        self.checks.append({"id": cid, "description": description, "expected": expected,
                            "observed": observed, "status": status})
        return status == "pass"

    def info(self, cid, description, observed):
        self.checks.append({"id": cid, "description": description, "expected": None,
                            "observed": observed, "status": "info"})


# Prefixo do nome padronizado `<opção>_<dataset_id>.zip` (download_filename do exportador),
# declarado aqui, sem importar o gerador.
API_FILE_PREFIX = {"raw": "bruto", "features": "feature_por_janela", "full": "completo"}


# ─── instantes: mesmo formato do codec (ISO 8601 UTC naive + "Z") ─────────────────
def iso_z(value: datetime) -> str:
    return value.isoformat() + "Z"


def parse_z(text) -> datetime:
    """Instante canônico do pacote: termina em Z, sem offset, e reescrito igual por iso_z."""
    if not isinstance(text, str) or not text.endswith("Z"):
        raise ValueError(f"instante sem Z: {text!r}")
    value = datetime.fromisoformat(text[:-1])
    if value.tzinfo is not None or iso_z(value) != text:
        raise ValueError(f"instante fora do formato canônico: {text!r}")
    return value


def parse_arg(text: str) -> datetime:
    """Instante informado ao auditor (ISO 8601 com Z ou offset) → UTC naive."""
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError(f"instante sem fuso: {text!r}")
    return value.astimezone(timezone.utc).replace(tzinfo=None)


@dataclass(frozen=True)
class Req:
    """Pedido lido do manifesto; grade e início do bruto calculados aqui."""
    source_id: str
    network_id: int
    node_ids: Optional[tuple]
    start: datetime
    end: datetime
    window_seconds: int
    step_seconds: int
    grid_origin: datetime
    export_kind: str

    def grid(self) -> list:
        """t = grid_origin + k·step para start < t ≤ end (aritmética inteira em µs)."""
        step = self.step_seconds * 1_000_000
        offset = (self.start - self.grid_origin) // timedelta(microseconds=1)
        k = offset // step + 1          # menor k com grid_origin + k·step > start
        instants = []
        while True:
            t = self.grid_origin + timedelta(microseconds=k * step)
            if t > self.end:
                return instants
            instants.append(t)
            k += 1

    def data_start(self, grid: list) -> datetime:
        """Início (exclusivo) do bruto: contexto da 1ª janela, mais a leitura de bateria anterior
        a ela que o extrator consulta (BATTERY_LOOKBACK). A margem de referência não amplia
        o bruto, só a seleção de simulações."""
        from ai.feature_engineering import BATTERY_LOOKBACK
        if not grid:
            return self.start
        return min(self.start, grid[0] - timedelta(seconds=self.window_seconds) - BATTERY_LOOKBACK)


def request_from_manifest(m) -> Req:
    r = m["request"]
    return Req(source_id=m["source"]["source_id"], network_id=int(r["network_id"]),
               node_ids=tuple(int(n) for n in r["node_ids"]) if r["node_ids"] is not None else None,
               start=parse_z(r["start"]), end=parse_z(r["end"]),
               window_seconds=int(r["window_seconds"]), step_seconds=int(r["step_seconds"]),
               grid_origin=parse_z(r["grid_origin"]), export_kind=m["export_kind"])


# ─── leitura ─────────────────────────────────────────────────────────────────────
def _csv_rows(zf, root, name):
    with zf.open(f"{root}/{name}") as raw:
        it = read_csv(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
        header = next(it)
        return header, [dict(zip(header, r)) for r in it]


def _loads(text):
    """JSON de uma célula; nulo ou JSON inválido → None (nunca igual a lista/dict esperado)."""
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _int(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _close(text, value) -> bool:
    try:
        return text is not None and math.isclose(float(text), float(value), rel_tol=0, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def _sort_key(values):
    return tuple((v is None, v) for v in values)


def _first_unsorted(keys) -> Optional[int]:
    return next((i for i in range(1, len(keys)) if keys[i] < keys[i - 1]), None)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_source(path: Path):
    """Origem somente leitura (URI mode=ro + query_only) com cache de leitura maior nesta conexão.

    cache_size/mmap_size só afetam a conexão do auditor; não gravam no arquivo nem
    mudam resultados (a auditoria chama o extrator ~1× por nó × instante).
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"banco de origem inexistente: {path}")
    engine = build_source_engine(f"sqlite:///{path.as_uri()}?mode=ro&uri=true")

    @event.listens_for(engine, "connect")
    def _read_cache(dbapi_connection, _record):
        dbapi_connection.execute("PRAGMA cache_size=-524288")      # 512 MiB
        dbapi_connection.execute("PRAGMA mmap_size=1073741824")    # 1 GiB

    return engine


def read_manifest_unverified(path: Path):
    """Manifesto e raiz sem validação (para auditar um pacote que falhou em verify_package)."""
    with zipfile.ZipFile(path) as zf:
        roots = {n.split("/", 1)[0] for n in zf.namelist()}
        if len(roots) != 1:
            raise PackageError(f"pacote sem raiz única: {sorted(roots)[:5]}")
        root = roots.pop()
        return json.loads(zf.read(f"{root}/manifest.json")), root


def resolve_commit(rev: Optional[str]) -> Optional[str]:
    """SHA completo de REV (padrão HEAD); None se o git não resolver."""
    proc = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{rev or 'HEAD'}^{{commit}}"],
                          cwd=ROOT, capture_output=True, text=True)
    return proc.stdout.strip() or None


# ─── origem: regras de recorte do exportador ──────────────────────────────────────
def expected_slice(session, req: Req, grid: list, margin_minutes: float):
    """Regras de recorte do exportador em consultas diretas: devolve {tabela: {id: linha}} e os nós pedidos."""
    node_t, nb_t, tel_t, sim_t, net_t = (table(n) for n in
                                         ("node", "neighbor", "telemetry", "simulation_run", "network"))
    lo, hi = req.data_start(grid), req.end
    rows = lambda stmt: [dict(r) for r in session.execute(stmt).mappings()]  # noqa: E731
    in_net = {r["node_id"]: r for r in rows(select(node_t).where(node_t.c.network_id == req.network_id))}
    requested = sorted(in_net) if req.node_ids is None else sorted(set(req.node_ids))
    addresses = sorted({in_net[n]["lora_address"] for n in requested
                        if in_net.get(n, {}).get("lora_address")})
    tel = {r["id"]: r for r in rows(select(tel_t).where(
        tel_t.c.node_id.in_(requested), tel_t.c.timestamp > lo, tel_t.c.timestamp <= hi))}
    nb = {r["id"]: r for r in rows(select(nb_t).where(
        nb_t.c.timestamp > lo, nb_t.c.timestamp <= hi).where(
        (nb_t.c.node_id.in_(requested)) | (nb_t.c.neighbor_address.in_(addresses))))}
    reporters = {r["node_id"] for r in nb.values() if r["node_id"] not in set(requested)}
    heard = {r["neighbor_address"] for r in nb.values() if r["node_id"] in set(requested)}
    nodes = {r["node_id"]: r for r in rows(select(node_t).where(
        (node_t.c.node_id.in_(sorted(set(requested) | reporters)))
        | (node_t.c.lora_address.in_(sorted(heard)))))}
    margin = timedelta(minutes=margin_minutes)
    sims = {r["id"]: r for r in rows(select(sim_t).where(
        sim_t.c.node_id.in_(sorted(nodes)), sim_t.c.started_at < hi + margin).where(
        (sim_t.c.ended_at.is_(None)) | (sim_t.c.ended_at > lo - margin)))}
    nets = {}
    while True:  # fecho referencial
        need_net = {n["network_id"] for n in nodes.values()} - set(nets) - {None}
        need_sim = {n["current_sim_run_id"] for n in nodes.values()} - set(sims) - {None}
        need_node = {s["node_id"] for s in sims.values()} - set(nodes)
        before = (len(nets), len(sims), len(nodes))
        if need_net:
            nets.update({r["id"]: r for r in rows(select(net_t).where(net_t.c.id.in_(sorted(need_net))))})
        if need_sim:
            sims.update({r["id"]: r for r in rows(select(sim_t).where(sim_t.c.id.in_(sorted(need_sim))))})
        if need_node:
            nodes.update({r["node_id"]: r for r in rows(
                select(node_t).where(node_t.c.node_id.in_(sorted(need_node))))})
        if (len(nets), len(sims), len(nodes)) == before:
            break
    return {"telemetry": tel, "neighbor": nb, "node": nodes, "network": nets, "simulation_run": sims}, requested


def expected_role(tname, row, req, requested):
    inside = lambda ts: req.start < ts <= req.end  # noqa: E731
    if tname in ("telemetry", "neighbor"):
        return "requested" if row["node_id"] in requested and inside(row["timestamp"]) else "context"
    if tname == "node":
        return "requested" if row["node_id"] in requested else "context"
    if tname == "network":
        return "requested" if row["id"] == req.network_id else "context"
    ended = row["ended_at"]
    overlaps = row["started_at"] < req.end and (ended is None or ended > req.start)
    return "requested" if row["node_id"] in requested and overlaps else "context"


def files_for_kind(kind: str) -> set:
    files = set(COMMON_FILES)
    if kind in ("raw", "full"):
        files |= set(RAW_TABLES)
    if kind in ("features", "full"):
        files |= FEATURE_FILES
    return files


# ─── seções ──────────────────────────────────────────────────────────────────────
def audit_request(a, args, manifest):
    """Pedido registrado pelo auditor (independente do manifesto): o pacote deve corresponder a ele."""
    r, interp = manifest["request"], manifest["interpretation"]
    observed = {"source_id": manifest["source"]["source_id"], "network_id": r["network_id"],
                "start": r["start"], "end": r["end"], "window_seconds": r["window_seconds"],
                "step_seconds": r["step_seconds"], "export_kind": manifest["export_kind"],
                "node_ids": sorted(r["node_ids"]) if r["node_ids"] is not None else None,
                "grid_origin": r["grid_origin"],
                "data_origin_tracked_since": interp.get("data_origin_tracked_since"),
                "reference_margin_minutes": interp.get("reference_margin_minutes"),
                "campaign_id": manifest.get("campaign_id")}
    informed = {}
    for key, value in (("source_id", args.expect_source_id), ("network_id", args.expect_network_id),
                       ("window_seconds", args.expect_window), ("step_seconds", args.expect_step),
                       ("export_kind", args.expect_kind)):
        if value is not None:
            informed[key] = value
    for key, value in (("start", args.expect_start), ("end", args.expect_end),
                       ("grid_origin", args.expect_grid_origin)):
        if value is not None:
            informed[key] = iso_z(parse_arg(value))
    if args.expect_nodes is not None:
        informed["node_ids"] = (None if args.expect_nodes == "all"
                                else sorted(int(n) for n in args.expect_nodes.split(",")))
    if args.expect_tracked_since is not None:
        informed["data_origin_tracked_since"] = (None if args.expect_tracked_since.lower() == "none"
                                                 else iso_z(parse_arg(args.expect_tracked_since)))
    if args.expect_margin is not None:
        informed["reference_margin_minutes"] = float(args.expect_margin)
    if args.expect_campaign_id is not None:
        informed["campaign_id"] = None if args.expect_campaign_id.lower() == "none" else args.expect_campaign_id
    got = {k: observed[k] for k in informed}
    if "reference_margin_minutes" in got and got["reference_margin_minutes"] is not None:
        got["reference_margin_minutes"] = float(got["reference_margin_minutes"])
    a.add("request:matches_expected", "manifesto = pedido registrado pelo auditor (--expect-*)",
          informed, got, status="info" if not informed else None)


def audit_manifest(a, zf, root, manifest, req, grid, expected, requested, args):
    bad_hash = []
    for name, info in manifest["files"].items():
        try:
            data = zf.read(f"{root}/{name}")
        except KeyError:
            bad_hash.append(name)
            continue
        if hashlib.sha256(data).hexdigest() != info.get("sha256") or len(data) != info.get("bytes"):
            bad_hash.append(name)
    a.add("manifest:hashes", "SHA-256 e tamanhos recalculados", [], bad_hash)
    a.add("manifest:nodes_requested", "nós pedidos no manifesto = regra", list(requested),
          manifest["nodes"]["requested"])
    a.add("manifest:nodes_context", "nós de contexto = fecho − pedidos",
          sorted(set(expected["node"]) - set(requested)), manifest["nodes"]["context"])
    a.add("manifest:simulation_runs", "simulações = regra (período/margem/fecho)",
          sorted(expected["simulation_run"]), manifest["simulation_runs"])
    a.add("manifest:networks", "redes = fecho referencial", sorted(expected["network"]), manifest.get("networks"))
    a.add("manifest:slice_rows", "linhas do recorte materializado = regra de recorte na origem",
          {t: len(expected[t]) for t in sorted(expected)}, manifest.get("slice_rows"))
    a.add("manifest:resolved", "limites resolvidos = grade e contexto recalculados",
          {"interval_convention": "(start, end]", "data_start": iso_z(req.data_start(grid)),
           "end": iso_z(req.end), "grid_instants": len(grid),
           "first_instant": iso_z(grid[0]) if grid else None,
           "last_instant": iso_z(grid[-1]) if grid else None},
          manifest.get("resolved"))
    a.add("manifest:labels_status", "labels_status = not_provided (pacote sem rótulos)", LABELS_STATUS,
          manifest.get("labels_status"))
    limitations = manifest.get("limitations")
    well_formed = (isinstance(limitations, list) and bool(limitations)
                   and all(isinstance(x, str) and x.strip() for x in limitations))
    a.add("manifest:limitations", "limitações: lista não vazia de textos não vazios = LIMITATIONS do código",
          LIMITATIONS, limitations, ok=well_formed and limitations == LIMITATIONS)
    if manifest["export_kind"] == "raw":
        a.add("manifest:features", "pacote raw sem bloco de features", None, manifest.get("features"))
    a.add("manifest:zip_root", "ZIP com raiz única = package_root", manifest.get("package_root"), root)
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", req.source_id)
    pattern = f"dataset_{re.escape(safe)}_{req.start:%Y%m%dT%H%M%S}Z_{req.end:%Y%m%dT%H%M%S}Z_[0-9a-f]{{8}}"
    a.add("zip:name", "raiz = dataset_<source_id>_<início>_<fim>_<8 hex> e dataset_id = raiz",
          {"pattern": pattern, "dataset_id": root},
          {"root": root, "dataset_id": manifest.get("dataset_id")},
          ok=re.fullmatch(pattern, root) is not None and manifest.get("dataset_id") == root)
    filename = Path(args.package).name
    kind = manifest["export_kind"]
    api_name = f"{API_FILE_PREFIX.get(kind, kind)}_{root}.zip"
    a.info("zip:filename", "nome do arquivo: o nome padronizado é '<opção>_<dataset_id>.zip'; a CLI grava em --out",
           {"filename": filename, "api_name": api_name, "matches_api_name": filename == api_name})
    a.add("zip:files_for_kind", f"ZIP contém só os arquivos previstos para a opção {kind}",
          sorted(files_for_kind(kind)), sorted(n.split("/", 1)[1] for n in zf.namelist()))
    interp, r = manifest["interpretation"], manifest["request"]
    a.add("manifest:interpretation", "parâmetros de interpretação coerentes com o pedido do manifesto",
          {"source_id": manifest["source"]["source_id"], "window_seconds": r["window_seconds"],
           "step_seconds": r["step_seconds"], "grid_origin": r["grid_origin"]},
          {k: interp.get(k) for k in ("source_id", "window_seconds", "step_seconds", "grid_origin")})
    target, code = resolve_commit(args.expect_commit), manifest.get("code") or {}
    origin = f"--expect-commit {args.expect_commit}" if args.expect_commit else "HEAD do auditor"
    a.add("manifest:code", f"commit do pacote = {origin} e dirty=false"
          + (" (dirty permitido por --allow-dirty)" if args.allow_dirty else ""),
          {"commit": target, "dirty": "any" if args.allow_dirty else False},
          {"commit": code.get("commit"), "dirty": code.get("dirty")},
          ok=target is not None and code.get("commit") == target
          and (args.allow_dirty or code.get("dirty") is False))
    audit_source_hashes(a, manifest, target, code)
    file_sha = (manifest.get("source") or {}).get("file_sha256")
    if file_sha is None:
        a.info("manifest:source_file", "source.file_sha256 nulo (exportação sem --sqlite, lida de DATABASE_URL): "
               "não há hash de origem a recomputar; o banco auditado fica identificado por source_sha256 do "
               "relatório", None)
    else:
        a.add("manifest:source_file", "source.file_sha256 = SHA-256 do --sqlite auditado", args.source_sha256,
              file_sha)


def commit_source_hashes(commit: str) -> dict:
    """SHA-256 dos RELEVANT_SOURCES no commit (git show <commit>:<arquivo>); arquivo ausente fica de fora,
    como em provenance.source_hashes()."""
    hashes = {}
    for rel in RELEVANT_SOURCES:
        proc = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=ROOT, capture_output=True)
        if proc.returncode == 0:
            hashes[rel] = hashlib.sha256(proc.stdout).hexdigest()
    return hashes


def audit_source_hashes(a, manifest, target, code):
    """source_hashes = SHA-256 dos fontes do código (provenance.RELEVANT_SOURCES) na exportação."""
    recorded = manifest.get("source_hashes")
    if code.get("dirty") is not False or target is None:
        # árvore modificada (ou commit indeterminado): os fontes usados não estão em nenhum commit
        working = {rel: sha256_path(ROOT / rel) for rel in RELEVANT_SOURCES if (ROOT / rel).is_file()}
        listed = recorded if isinstance(recorded, dict) else {}
        a.info("manifest:source_hashes", "não recomputável: pacote gerado com dirty≠false (ou commit não "
               "resolvido); comparado só com a árvore de trabalho do auditor, a título de evidência",
               {"dirty": code.get("dirty"), "matches_auditor_working_tree": recorded == working,
                "differs": sorted(k for k in set(working) | set(listed) if working.get(k) != listed.get(k))})
        return
    a.add("manifest:source_hashes", f"SHA-256 dos {len(RELEVANT_SOURCES)} fontes do código = git show "
          f"{target[:12]}:<arquivo>", commit_source_hashes(target), recorded)


def audit_raw(a, zf, root, manifest, req, requested, expected):
    requested_set = set(requested)
    for file_name, tname in RAW_TABLES.items():
        header, rows = _csv_rows(zf, root, file_name)
        src = table(tname)
        key = RAW_KEY[tname]
        a.add(f"columns:{tname}", f"{file_name}: cabeçalho = colunas do modelo + source_id, record_role",
              [c.name for c in src.columns] + list(EXTRA_COLUMNS), header)
        got_ids = sorted(i for i in (_int(r.get(key)) for r in rows) if i is not None)
        exp_ids = sorted(expected[tname])
        a.add(f"ids:{tname}", f"{file_name}: conjunto de IDs = regra de recorte na origem",
              {"n": len(exp_ids), "missing": [], "extra": [], "invalid_or_duplicated": 0},
              {"n": len(got_ids), "missing": sorted(set(exp_ids) - set(got_ids))[:20],
               "extra": sorted(set(got_ids) - set(exp_ids))[:20],
               "invalid_or_duplicated": len(rows) - len(set(got_ids))})
        bad_values, bad_roles, bad_src, order_keys, compared = [], [], [], [], 0
        for r in rows:
            db = expected[tname].get(_int(r.get(key)))
            if db is None:
                continue  # já reportado em ids:<tabela>
            order_keys.append(_sort_key(db[k] for k in RAW_ORDER[tname]))
            for c in src.columns:
                compared += 1
                problem = _raw_value_problem(r.get(c.name), db[c.name], c)
                if problem:
                    bad_values.append({"id": r.get(key), "col": c.name, "csv": r.get(c.name),
                                       "expected_text": encode(db[c.name], c), "problem": problem})
            if r.get("record_role") != expected_role(tname, db, req, requested_set):
                bad_roles.append({"id": r.get(key), "csv": r.get("record_role")})
            if r.get("source_id") != req.source_id:
                bad_src.append(r.get(key))
        a.compared[f"values:{tname}"] = compared
        a.add(f"values:{tname}", f"{file_name}: valores = origem (texto exato do codec e valor decodificado)",
              [], bad_values[:20])
        a.add(f"role:{tname}", f"{file_name}: record_role = regra (período, contexto, reverso, fecho)",
              [], bad_roles[:20])
        a.add(f"source_id:{tname}", f"{file_name}: source_id constante", [], bad_src[:20])
        a.add(f"count:{tname}", f"{file_name}: linhas = manifest.tables",
              ((manifest.get("tables") or {}).get(tname) or {}).get("rows"), len(rows))
        a.add(f"order:{tname}", f"{file_name}: ordem {RAW_ORDER[tname]} (schema.json conventions.ordering)",
              None, _first_unsorted(order_keys))
        a.info(f"roles_dist:{tname}", f"{file_name}: distribuição de record_role",
               dict(Counter(r.get("record_role") for r in rows)))


def _raw_value_problem(text, db_value, column) -> Optional[str]:
    try:
        if decode(text, column) != db_value:
            return "decoded value differs"
    except ValueError as exc:  # valor ilegível é falha de auditoria, não erro de execução
        return str(exc)
    if text != encode(db_value, column):
        return "text differs from codec encoding"   # ex.: " -71", "+5", "1e1", "+00:00"
    return None


# ─── features.csv / coverage.csv ─────────────────────────────────────────────────
def audit_features(a, zf, root, manifest, session, req, grid, requested, interpretation, expected):
    from ai.reference_policy import ReferencePolicy
    _, rows = _csv_rows(zf, root, "features.csv")
    order_keys = _features_layout(a, rows, req, grid, requested)
    policy = ReferencePolicy.from_session(session, interpretation.reference_margin_minutes)
    windows = _compare_windows(a, rows, session, req, interpretation, expected, policy)
    _classify_cases(a, rows, windows, policy, ReferencePolicy.from_session(session, 0.0), expected)
    _, cov = _csv_rows(zf, root, "coverage.csv")
    _reaggregate_coverage(a, rows, cov, req, grid, requested, expected, order_keys)
    _manifest_features(a, manifest, rows, cov, grid, requested)


def _features_layout(a, rows, req, grid, requested) -> list:
    """Grade, chaves, janela e identidade de versão; devolve as chaves de ordenação válidas."""
    from dataset_export.features import FEATURE_SCHEMA_VERSION, extractor_revision
    a.add("features:grid", "linhas = nós pedidos × instantes da grade (recalculada)",
          len(grid) * len(requested), len(rows))
    # chaves (node_id, window_end_utc): conjunto esperado, sem duplicatas e sem ausências
    expected_keys = {(str(n), iso_z(end)) for n in requested for end in grid}
    got_keys = Counter((r["node_id"], r["window_end_utc"]) for r in rows)
    a.add("features:keys", "chaves (node_id, window_end_utc) = pedidos × grade, únicas",
          {"missing": [], "duplicated": [], "unexpected": []},
          {"missing": sorted(expected_keys - set(got_keys))[:20],
           "duplicated": sorted(k for k, n in got_keys.items() if n > 1)[:20],
           "unexpected": sorted(set(got_keys) - expected_keys)[:20]})
    window = timedelta(seconds=req.window_seconds)
    bad_window, order_keys = [], []
    for r in rows:
        try:
            end = parse_z(r["window_end_utc"])
            ok = r["window_start_utc"] == iso_z(end - window) and r["available_at_utc"] == r["window_end_utc"]
            order_keys.append((_int(r["node_id"]) if _int(r["node_id"]) is not None else -1, end))
        except (TypeError, ValueError):
            ok = False
        if not ok:
            bad_window.append((r["node_id"], r["window_start_utc"], r["window_end_utc"], r["available_at_utc"]))
    a.add("features:window", "window_start = iso_z(fim − janela), available_at = window_end (texto exato, com Z)",
          [], bad_window[:20])
    from artifact_versions import version_matches
    a.add("features:schema_version", "feature_schema_version esperado", [FEATURE_SCHEMA_VERSION],
          sorted({str(r["feature_schema_version"]) for r in rows}),
          ok=all(version_matches(r["feature_schema_version"], FEATURE_SCHEMA_VERSION) for r in rows))
    a.add("features:extractor_revision", "extractor_revision = código atual", [extractor_revision()],
          sorted({str(r["extractor_revision"]) for r in rows}))
    a.add("features:extraction_policy", "extraction_policy = all_origins_v1", [EXTRACTION_POLICY],
          sorted({str(r["extraction_policy"]) for r in rows}))
    return order_keys


def _compare_windows(a, rows, session, req, interpretation, expected, policy) -> list:
    """Cada linha × extrator aplicado à origem na mesma janela. Devolve o contexto de cada janela
    (linha, nó, limites, fontes, simulações, linhas sintéticas) para a classificação dos casos."""
    from ai.feature_engineering import FeatureExtractor
    from ai.sequence_builder import FEATURE_NAMES
    from dataset_export.features import _status, make_origin_classifier, origin_status
    fe = FeatureExtractor()
    classify = make_origin_classifier(interpretation.data_origin_tracked_since)
    window = timedelta(seconds=req.window_seconds)
    bad = {k: [] for k in ("identity", "values", "status", "counts", "origin", "simulation",
                           "eligibility", "defaults_match")}
    windows, n_valid = [], 0
    for r in rows:
        node = _int(r["node_id"])
        try:
            end = parse_z(r["window_end_utc"])
        except (TypeError, ValueError):
            continue  # já reportado em features:window / features:keys
        start = end - window
        where = {"node": node, "end": r["window_end_utc"]}
        node_row = expected["node"].get(node) or {}
        ident = {"source_id": req.source_id, "node_name": node_row.get("name"),
                 "network_id": None if node_row.get("network_id") is None else str(node_row["network_id"]),
                 "window_seconds": str(req.window_seconds), "step_seconds": str(req.step_seconds),
                 "grid_origin_utc": iso_z(req.grid_origin)}
        if {k: r[k] for k in ident} != ident:
            bad["identity"].append({**where, "csv": {k: r[k] for k in ident}, "src": ident})
        prov = {}
        try:
            f = fe.extract_for_window(session, node, start, end, provenance=prov, origin_classifier=classify)
            status, reason, _ = _status(f)
        except Exception as exc:  # mesmo tratamento do gerador
            f, status, reason = None, "error", f"{type(exc).__name__}: {exc}"
        if (r["feature_status"], r["invalid_reason"]) != (status, reason):
            bad["status"].append({**where, "csv": (r["feature_status"], r["invalid_reason"]),
                                  "src": (status, reason)})
        if status == "valid":
            n_valid += 1
            for k in FEATURE_NAMES:
                if not _close(r[k], f[k]):
                    bad["values"].append({**where, "feature": k, "csv": r[k], "src": f[k]})
        elif any(r[k] is not None for k in FEATURE_NAMES):
            bad["values"].append({**where, "feature": "*", "csv": "non-null", "src": status})
        for k in COUNT_COLUMNS:
            if _int(r[k]) != prov.get(k, 0):
                bad["counts"].append({**where, "k": k, "csv": r[k], "src": prov.get(k, 0)})
        for col, k in (("first_observation_utc", "first_observation"),
                       ("last_observation_utc", "last_observation")):
            src_v = iso_z(prov[k]) if prov.get(k) is not None else None
            if r[col] != src_v:
                bad["counts"].append({**where, "k": col, "csv": r[col], "src": src_v})
        # defaults e indisponibilidade = extração (o gerador grava sorted(defaulted) e dict ordenado)
        src_defaulted = sorted(prov.get("defaulted_features", []))
        src_unavailable = dict(sorted(((f or {}).get("_unavailable") or {}).items()))
        if _loads(r["defaulted_features"]) != src_defaulted or _loads(r["unavailable_features"]) != src_unavailable:
            bad["defaults_match"].append({**where, "csv": (r["defaulted_features"], r["unavailable_features"]),
                                          "src": (src_defaulted, src_unavailable)})
        counts = prov.get("origin_counts", {})
        if (r["origin_status"] != origin_status(counts)
                or _int(r["real_record_count"]) != counts.get("real", 0)
                or _int(r["synthetic_record_count"]) != counts.get("synthetic", 0)
                or _int(r["unknown_origin_record_count"]) != counts.get("unknown", 0)):
            bad["origin"].append({**where, "csv": r["origin_status"], "src": counts})
        sources = sorted(set(prov.get("source_node_ids", ())) | {node})
        runs = policy.overlapping_runs(sources, start, end)
        if (_loads(r["simulation_run_ids"]) != runs or r["simulation_overlap"] != ("1" if runs else "0")
                or _loads(r["source_node_ids"]) != sources):
            bad["simulation"].append({**where, "csv": (r["simulation_overlap"], r["simulation_run_ids"],
                                                       r["source_node_ids"]),
                                      "src": (bool(runs), runs, sources)})
        policy_reason = policy.check_window(node, start, end, prov) if prov else None
        if status != "valid":
            elig, excl = False, "features_not_valid"
        elif policy_reason:
            elig, excl = False, policy_reason
        else:
            elig, excl = True, None
        if (r["reference_eligible_by_policy"], r["reference_exclusion_reason"]) != ("1" if elig else "0", excl):
            bad["eligibility"].append({**where, "csv": (r["reference_eligible_by_policy"],
                                                        r["reference_exclusion_reason"]),
                                       "src": (elig, excl)})
        windows.append({"row": r, "node": node, "start": start, "end": end, "sources": sources, "runs": runs,
                        "synthetic": counts.get("synthetic", 0)})
    a.compared.update({"features:windows": len(rows), "features:valid_windows": n_valid,
                       "features:values": n_valid * len(FEATURE_NAMES)})
    texts = {
        "identity": "source_id, node_name, network_id, janela, passo e origem da grade = origem/pedido",
        "values": "11 features = extrator direto na origem (|Δ| ≤ 1e-9)",
        "status": "feature_status/invalid_reason = extrator",
        "counts": "contagens de registros e 1ª/última observação = proveniência do extrator",
        "origin": "origin_status e contagens por origem = classificador",
        "simulation": "simulation_overlap/run_ids/source_node_ids = política",
        "eligibility": "reference_eligible/exclusion_reason = regra",
        "defaults_match": "defaulted_features/unavailable_features = extração",
    }
    for key, text in texts.items():
        a.add(f"features:{key}", text, [], bad[key][:20])
    return windows


def _classify_cases(a, rows, windows, policy, policy_no_margin, expected):
    """Evidência de cada caso de origem/simulação, com base nos valores já conferidos."""
    run_node = {rid: row["node_id"] for rid, row in expected["simulation_run"].items()}
    cases = {k: [] for k in CASE_NAMES}
    for w in windows:
        r, runs, synthetic = w["row"], w["runs"], w["synthetic"]
        example = (r["node_id"], r["window_end_utc"], r["reference_exclusion_reason"], runs)
        own_runs = policy.overlapping_runs([w["node"]], w["start"], w["end"])
        if r["origin_status"] in cases:
            cases[r["origin_status"]].append(example)
        if synthetic and not own_runs:
            cases["reverse_synthetic"].append(example)
        if runs and not synthetic and not own_runs:
            cases["reverse_simulation"].append(example)
        if (runs and not synthetic and r["reference_exclusion_reason"] == "simulation_overlap"
                and not policy_no_margin.overlapping_runs(w["sources"], w["start"], w["end"])):
            cases["margin_only"].append(example)
        # legado = intervalo Node.sim_* sem SimulationRun do mesmo nó na janela (o legado espelha a última execução)
        legacy_nodes = {int(x[len(LEGACY_PREFIX):]) for x in runs
                        if isinstance(x, str) and x.startswith(LEGACY_PREFIX)}
        if legacy_nodes - {run_node.get(x) for x in runs if isinstance(x, int)}:
            cases["legacy"].append(example)
    combos = Counter((r["origin_status"], r["simulation_overlap"], r["reference_exclusion_reason"]) for r in rows)
    a.info("origin:cases", "janelas por (origin_status, simulation_overlap, reference_exclusion_reason)",
           [{"origin_status": k[0], "simulation_overlap": k[1], "exclusion": k[2], "windows": v}
            for k, v in sorted(combos.items(), key=lambda kv: (-kv[1], str(kv[0])))])
    reverse_only = [r for r in rows if r["simulation_overlap"] == "1" and r["synthetic_record_count"] == "0"]
    a.info("origin:reverse_or_margin",
           "janelas com sobreposição de simulação sem linha sintética própria (reverso ou margem)",
           {"windows": len(reverse_only),
            "examples": [(r["node_id"], r["window_end_utc"], r["reference_exclusion_reason"])
                         for r in reverse_only[:5]]})
    a.info("origin:case_examples", ORIGIN_CASES_TEXT, {k: {"windows": len(v), "examples": v[:3]} for k, v in cases.items()})
    defaults = Counter(d for r in rows for d in (_loads(r["defaulted_features"]) or []))
    unavailable = Counter(k for r in rows for k in (_loads(r["unavailable_features"]) or {}))
    a.info("features:defaults", "features preenchidas por default (contagem de janelas)", dict(defaults))
    a.info("features:unavailable", "features indisponíveis (contagem de janelas)", dict(unavailable))


def _reaggregate_coverage(a, rows, cov, req, grid, requested, expected, order_keys):
    """coverage.csv recalculado de features.csv (agregação independente) e ordem dos dois arquivos."""
    by_node = {}
    for r in rows:
        c = by_node.setdefault(r["node_id"], Counter())
        c["written"] += 1
        c[f"status_{r['feature_status']}"] += 1
        c[f"origin_{r['origin_status']}"] += 1
        c["reference_eligible" if r["reference_eligible_by_policy"] == "1" else "reference_excluded"] += 1
        # o gerador (CoverageAccumulator.add) só conta defaults em linhas válidas
        c["windows_with_defaults"] += 1 if (r["feature_status"] == "valid" and _loads(r["defaulted_features"])) else 0
        c["simulation_overlap_windows"] += 1 if r["simulation_overlap"] == "1" else 0
    bad_cov, bad_cov_ident = [], []
    first_instant, last_instant = (iso_z(grid[0]), iso_z(grid[-1])) if grid else (None, None)
    for r in cov:
        c = by_node.get(r["node_id"], Counter())
        for k in ("status_valid", "status_insufficient_data", "status_invalid_numeric", "status_error",
                  "origin_real", "origin_synthetic", "origin_mixed", "origin_unknown", "origin_no_data",
                  "reference_eligible", "reference_excluded", "windows_with_defaults", "simulation_overlap_windows"):
            if _int(r[k]) != c[k]:
                bad_cov.append({"node": r["node_id"], "k": k, "coverage": r[k], "features": c[k]})
        if _int(r["written_windows"]) != c["written"] or _int(r["expected_windows"]) != len(grid):
            bad_cov.append({"node": r["node_id"], "k": "written/expected",
                            "coverage": (r["written_windows"], r["expected_windows"]),
                            "features": (c["written"], len(grid))})
        node_row = expected["node"].get(_int(r["node_id"])) or {}
        ident = {"source_id": req.source_id, "node_name": node_row.get("name"),
                 "requested_start_utc": iso_z(req.start), "requested_end_utc": iso_z(req.end),
                 "first_instant_utc": first_instant, "last_instant_utc": last_instant}
        if {k: r[k] for k in ident} != ident:
            bad_cov_ident.append({"node": r["node_id"], "coverage": {k: r[k] for k in ident}, "expected": ident})
    a.add("coverage:sums", "coverage.csv = agregação independente de features.csv", [], bad_cov[:20])
    a.add("coverage:identity", "source_id, node_name, período pedido e 1º/último instante da grade",
          [], bad_cov_ident[:20])
    a.add("coverage:nodes", "uma linha por nó pedido", sorted(map(str, requested)),
          sorted(str(r["node_id"]) for r in cov))
    cov_keys = [_int(r["node_id"]) if _int(r["node_id"]) is not None else -1 for r in cov]
    a.add("order:features", "features.csv por (node_id, window_end_utc); coverage.csv por node_id",
          {"features.csv": None, "coverage.csv": None},
          {"features.csv": _first_unsorted(order_keys), "coverage.csv": _first_unsorted(cov_keys)})


def _manifest_features(a, manifest, rows, cov, grid, requested):
    from ai.sequence_builder import FEATURE_NAMES
    from dataset_export.features import FEATURE_SCHEMA_VERSION, extractor_revision
    status_counts = {s: 0 for s in FEATURE_STATUSES}
    for r in rows:
        status_counts[r["feature_status"]] = status_counts.get(r["feature_status"], 0) + 1
    expected_rows = len(grid) * len(requested)
    a.add("manifest:features", "bloco features do manifesto = recontagem de features.csv/coverage.csv e pedido",
          {"feature_schema_version": FEATURE_SCHEMA_VERSION, "extractor_revision": extractor_revision(),
           "extraction_policy": EXTRACTION_POLICY, "feature_names": list(FEATURE_NAMES),
           "rows": len(rows), "expected_rows": expected_rows, "complete": len(rows) == expected_rows,
           "status_counts": status_counts, "coverage_rows": len(cov)},
          manifest.get("features"))


# ─── documentação embutida ───────────────────────────────────────────────────────
def _expected_column(file_name, name):
    """(tipo, nullable) esperados de uma coluna; nullable None = não comparado."""
    if file_name in RAW_TABLES:
        if name in EXTRA_COLUMNS:
            return "string", False
        src = table(RAW_TABLES[file_name])
        if name not in src.c:
            return None
        column = src.c[name]
        return CODEC_TO_SCHEMA[codec_kind(column)], bool(column.nullable)
    if file_name == "features.csv":
        from ai.sequence_builder import FEATURE_NAMES
        if name in FEATURE_NAMES:
            return "float", None
        if name in ("feature_status", "origin_status"):
            return "string", False
        if name.endswith("_utc"):
            return "datetime_utc", None
        if name.endswith("_count") or name in ("node_id", "network_id", "window_seconds", "step_seconds"):
            return "integer", None
        if name in ("simulation_overlap", "reference_eligible_by_policy"):
            return "boolean", None
        if name in ("defaulted_features", "source_node_ids", "simulation_run_ids"):
            return "json_array", None
        if name == "unavailable_features":
            return "json_object", None
        return "string", None
    if file_name == "coverage.csv":
        if name.endswith("_utc"):
            return "datetime_utc", None
        return ("string" if name in ("source_id", "node_name") else "integer"), None
    if file_name == "events_template.csv":
        types = {"node_id": "integer", "start_at_utc": "datetime_utc", "end_at_utc": "datetime_utc"}
        return types.get(name, "string"), None
    return None


def _value_ok(type_name, value) -> bool:
    try:
        if type_name == "integer":
            return INTEGER_TEXT.fullmatch(value) is not None
        if type_name == "float":
            number = float(value)
            return math.isfinite(number) and repr(number) == value
        if type_name == "boolean":
            return value in ("0", "1")
        if type_name == "datetime_utc":
            parse_z(value)
            return True
        if type_name == "json_array":
            return isinstance(json.loads(value), list)
        if type_name == "json_object":
            return isinstance(json.loads(value), dict)
        return True
    except (TypeError, ValueError):
        return False


def _column_problems(file_name, column, header, rows) -> list:
    cname, ctype, nullable, desc = (column.get("name"), column.get("type"), column.get("nullable"),
                                    column.get("description"))
    where = {"file": file_name, "column": cname}
    problems = []
    if ctype not in SCHEMA_TYPES:
        problems.append({**where, "type": ctype, "problem": "tipo ausente ou desconhecido"})
    if not isinstance(nullable, bool):
        problems.append({**where, "nullable": nullable, "problem": "nullable deve ser booleano"})
    if not isinstance(desc, str) or not desc.strip():
        problems.append({**where, "problem": "descrição vazia"})
    spec = _expected_column(file_name, cname)
    if spec is not None:
        exp_type, exp_nullable = spec
        if ctype != exp_type:
            problems.append({**where, "type": ctype, "expected_type": exp_type})
        if exp_nullable is not None and nullable != exp_nullable:
            problems.append({**where, "nullable": nullable, "expected_nullable": exp_nullable})
    if (file_name, cname) in ENUMS and column.get("enum") != ENUMS[(file_name, cname)]:
        problems.append({**where, "enum": column.get("enum"), "expected_enum": ENUMS[(file_name, cname)]})
    if cname not in header:
        return problems  # já reportado em schema_columns_vs_header
    values = [r[cname] for r in rows if r[cname] is not None]
    if "enum" in column:
        outside = sorted(set(values) - set(column["enum"] or []))
        if outside:
            problems.append({**where, "values_outside_enum": outside[:5]})
    bad = [v for v in values if not _value_ok(ctype, v)]
    if bad:
        problems.append({**where, "type": ctype, "bad": bad[:3], "n_bad": len(bad)})
    if nullable is False and len(values) != len(rows):
        problems.append({**where, "nullable_false_but_null": len(rows) - len(values)})
    return problems


def audit_docs(a, zf, root, manifest, verified):
    schema = json.loads(zf.read(f"{root}/schema.json"))
    csvs = [n for n in manifest["files"] if n.endswith(".csv")]
    a.add("schema:files", "todo CSV do pacote descrito no schema", [], [n for n in csvs if n not in schema["files"]])
    problems = []
    for name in csvs:
        if name not in schema["files"]:
            continue
        header, rows = _csv_rows(zf, root, name)
        cols = schema["files"][name].get("columns") or []
        names = [c.get("name") for c in cols]
        if names != header:
            problems.append({"file": name, "schema_columns_vs_header": (names, header)})
        for c in cols:
            problems.extend(_column_problems(name, c, header, rows))
    a.add("schema:complete", "colunas = cabeçalho; tipo conhecido e = modelo/regra; nullable booleano (= modelo "
          "nos brutos); descrição não vazia; enum = lista esperada e ⊇ valores; valores no formato do tipo", [], problems[:30])
    if "features.csv" in schema["files"]:
        from ai.sequence_builder import FEATURE_NAMES
        a.add("schema:model_input_columns", "model_input_columns = 11 features canônicas", list(FEATURE_NAMES),
              schema["files"]["features.csv"].get("model_input_columns"))
    header, rows = _csv_rows(zf, root, "events_template.csv")
    a.add("events_template:header", "events_template.csv: cabeçalho = EVENTS_HEADER e zero linhas",
          {"header": EVENTS_HEADER, "rows": 0}, {"header": header, "rows": len(rows)})
    readme = zf.read(f"{root}/README.md").decode()
    _readme_blocks(a, readme, verified)
    _readme_files_and_identity(a, readme, manifest)


def _readme_files_and_identity(a, readme, manifest):
    """README pertence a este pacote: cita só CSVs da opção e do pacote, e o cabeçalho traz
    dataset_id, formato, opção, fonte, rede e período deste manifesto (package.py: _README_HEAD)."""
    kind, r = manifest["export_kind"], manifest["request"]
    a.info("readme:mentions", "arquivos citados no README",
           sorted({m for m in re.findall(r"`([a-z_]+\.(?:csv|json|md|py))`", readme)}))
    cited = {m for m in re.findall(r"`([a-z_]+\.csv)`", readme)}
    option_csvs = {n for n in files_for_kind(kind) if n.endswith(".csv")}
    a.add("readme:no_absent_files", f"README não cita CSV ausente do pacote nem fora da opção {kind}",
          [], sorted(cited - (option_csvs & set(manifest["files"]))))
    required = {
        "title": f"# Pacote de dados — {manifest.get('dataset_id')}",
        "option": f"(formato `{manifest.get('format')}`, opção **{KIND_LABELS.get(kind)}**)",
        "source": f"da fonte `{manifest['source']['source_id']}`,",
        "network_period": f"rede {r['network_id']}, período UTC ({r['start']}, {r['end']}].",
    }
    first_line = readme.splitlines()[:1]
    missing = {k: text for k, text in required.items()
               if (first_line != [text] if k == "title" else text not in readme)}
    a.add("readme:identity", "README: 1ª linha = título com dataset_id; formato, opção, fonte, rede e período "
          "deste pacote; cita todos os CSVs da opção",
          {"missing": {}, "csv_not_cited": []},
          {"missing": missing, "csv_not_cited": sorted(option_csvs - cited)})


def _readme_blocks(a, readme, verified):
    """Verifica a sintaxe dos exemplos; hashes não autorizam executar código do ZIP."""
    blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.S)
    if not verified:
        a.compared["readme:python_blocks"] = 0
        a.add("readme:python_blocks", "blocos não analisados: package:verify falhou",
              {"blocks_min": 1, "failed": []}, {"blocks": len(blocks), "skipped": "package:verify falhou"},
              ok=False)
        return
    failed = []
    for i, block in enumerate(blocks):
        try:
            ast.parse(block)
        except SyntaxError as exc:
            failed.append({"block": i, "line": exc.lineno, "error": "invalid_python_syntax"})
    a.compared["readme:python_blocks"] = len(blocks)
    a.add("readme:python_blocks", "sintaxe dos blocos Python do README; execução desabilitada",
          {"blocks_min": 1, "failed": []}, {"blocks": len(blocks), "failed": failed, "executed": False},
          ok=bool(blocks) and not failed)


def audit_pandas_nulls(a, zf, root, manifest):
    """Diferenças de interpretação de nulos entre pandas.read_csv (padrão) e o codec."""
    import pandas as pd
    out = {}
    for name in sorted(n for n in manifest["files"] if n.endswith(".csv")):
        header, rows = _csv_rows(zf, root, name)
        with zf.open(f"{root}/{name}") as fh, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_csv(fh)
        if len(df) != len(rows) or list(df.columns) != header:
            out[name] = {"pandas_shape": [len(df), list(df.columns)[:5]], "codec_rows": len(rows)}
            continue
        per_column = {}
        for col in header:
            null = empty = text = 0
            for is_na, r in zip(df[col].isna().tolist(), rows):
                value = r[col]
                if value is None:
                    null += 1
                elif is_na:
                    if value == "":
                        empty += 1
                    else:
                        text += 1
            if null or empty or text:
                per_column[col] = {"codec_null": null, "empty_text_as_nan": empty, "text_as_nan": text}
        if per_column:
            out[name] = per_column
    a.info("pandas:nulls", "células por coluna: codec_null = nulo (vazio sem aspas, pandas NaN); "
           "empty_text_as_nan = texto vazio (\"\") que o pandas lê como NaN; text_as_nan = texto "
           "(ex.: 'NA') que o pandas lê como NaN", out)


def audit_import(a, package: Path, kind: str):
    """Reimporta o pacote com scripts/import_dataset.py num SQLite temporário e confere as FKs."""
    if kind not in ("raw", "full"):
        a.info("import:foreign_key_check", "não se aplica: pacote sem dados brutos", None)
        return
    with tempfile.TemporaryDirectory(prefix="dataset_audit_import_") as tmp:
        target = Path(tmp) / "importado.db"
        started = time.monotonic()
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "import_dataset.py"),
                               "--package", str(Path(package).resolve()), "--out", str(target)],
                              cwd=ROOT, capture_output=True, text=True, timeout=3600)
        violations = None
        if proc.returncode == 0:
            with contextlib.closing(sqlite3.connect(target)) as connection:
                violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        seconds = round(time.monotonic() - started, 2)
    a.add("import:foreign_key_check", "import_dataset.py importa o pacote e PRAGMA foreign_key_check vazio",
          {"importer_rc": 0, "violations": 0},
          {"importer_rc": proc.returncode, "violations": None if violations is None else len(violations),
           "stderr": proc.stderr.strip()[-300:], "seconds": seconds},
          ok=proc.returncode == 0 and violations == [])


# ─── comparação de dois pacotes do mesmo pedido ──────────────────────────────────
def compare_packages(path_a, path_b):
    out = {"csv_equal": {}, "variable_metadata": [], "unexpected_differences": []}
    try:
        ma, ra = verify_package(path_a)
        mb, rb = verify_package(path_b)
    except PackageError as exc:  # pacote adulterado: comparação falha, não erro de execução
        return {**out, "ok": False, "error": f"verify_package: {exc}"}
    with zipfile.ZipFile(path_a) as za, zipfile.ZipFile(path_b) as zb:
        for name in sorted(set(ma["files"]) | set(mb["files"])):
            if name.endswith(".csv"):
                out["csv_equal"][name] = (name in ma["files"] and name in mb["files"]
                                          and za.read(f"{ra}/{name}") == zb.read(f"{rb}/{name}"))
    sci = {k: (ma.get(k) == mb.get(k)) for k in SCIENTIFIC_KEYS}
    # hashes dos CSVs devem coincidir; README e schema podem diferir só se o conteúdo diferir
    files_eq = {n: ma["files"][n]["sha256"] == mb["files"].get(n, {}).get("sha256") for n in ma["files"]}
    out["files_sha256_equal"] = files_eq
    out["scientific_params"] = sci
    out["scientific_params_equal"] = all(sci.values())
    for k in VARIABLE_KEYS:
        if ma.get(k) != mb.get(k):
            out["variable_metadata"].append(k)
    for k in sorted(set(ma) | set(mb)):
        if k not in SCIENTIFIC_KEYS and k not in VARIABLE_KEYS and k != "files" and ma.get(k) != mb.get(k):
            out["unexpected_differences"].append(k)
    out["readme_equal"] = files_eq.get("README.md", False)
    out["ok"] = all(out["csv_equal"].values()) and out["scientific_params_equal"] and not out["unexpected_differences"]
    return out


# ─── execução ────────────────────────────────────────────────────────────────────
def _section(a, name, fn, *args):
    """Exceção numa seção vira falha registrada (rc=1); banco ilegível continua erro (rc=2)."""
    try:
        fn(*args)
    except SQLAlchemyError:
        raise
    except Exception as exc:
        frames = traceback.extract_tb(exc.__traceback__)[-4:]
        a.add(f"{name}:error", f"seção {name} interrompida", None,
              {"type": type(exc).__name__, "message": str(exc)[:500],
               "traceback": [f"{Path(f.filename).name}:{f.lineno} in {f.name}: {f.line}" for f in frames]},
              ok=False)


def run_audit(a, args, manifest, root, verified):
    """Todas as seções; levanta só para banco ilegível."""
    _section(a, "request", audit_request, a, args, manifest)
    try:
        req = request_from_manifest(manifest)
        interpretation = InterpretationParams.from_dict(manifest["interpretation"])
    except Exception as exc:
        a.add("manifest:request", "pedido e interpretação do manifesto legíveis", None,
              f"{type(exc).__name__}: {exc}"[:500], ok=False)
        return
    grid = req.grid()
    engine = open_source(args.sqlite)
    try:
        with Session(engine) as session, zipfile.ZipFile(args.package) as zf:
            expected, requested = expected_slice(session, req, grid, interpretation.reference_margin_minutes)
            _section(a, "manifest", audit_manifest, a, zf, root, manifest, req, grid, expected, requested, args)
            if manifest["export_kind"] in ("raw", "full"):
                _section(a, "raw", audit_raw, a, zf, root, manifest, req, requested, expected)
            if manifest["export_kind"] in ("features", "full"):
                _section(a, "features", audit_features, a, zf, root, manifest, session, req, grid, requested,
                         interpretation, expected)
            _section(a, "docs", audit_docs, a, zf, root, manifest, verified)
            _section(a, "pandas", audit_pandas_nulls, a, zf, root, manifest)
    finally:
        engine.dispose()
    if args.import_check:
        _section(a, "import", audit_import, a, args.package, manifest["export_kind"])
    else:
        a.info("import:foreign_key_check", "não executada (--no-import-check)", None)


def _parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--package", required=True, type=Path)
    p.add_argument("--sqlite", required=True, type=Path,
                   help="banco de origem (cópia congelada), aberto somente leitura")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--compare", type=Path, help="segundo pacote do mesmo pedido, para comparação")
    p.add_argument("--compare-out", type=Path)
    # pedido registrado pelo auditor (independente do manifesto): o pacote deve corresponder a ele
    p.add_argument("--expect-source-id")
    p.add_argument("--expect-network-id", type=int)
    p.add_argument("--expect-start")
    p.add_argument("--expect-end")
    p.add_argument("--expect-window", type=int)
    p.add_argument("--expect-step", type=int)
    p.add_argument("--expect-kind", choices=("raw", "features", "full"))
    p.add_argument("--expect-nodes", help="node_ids separados por vírgula, ou 'all' (manifesto com node_ids nulo)")
    p.add_argument("--expect-grid-origin", help="origem da grade (ISO 8601)")
    p.add_argument("--expect-tracked-since", help="DATA_ORIGIN_TRACKED_SINCE esperado (ISO 8601) ou 'none'")
    p.add_argument("--expect-margin", type=float, help="margem de referência esperada, em minutos")
    p.add_argument("--expect-campaign-id", help="campaign_id esperado, ou 'none' (sem campanha)")
    p.add_argument("--expect-commit", help="commit esperado do pacote (SHA, curto ou não, ou ref; padrão: HEAD)")
    p.add_argument("--allow-dirty", action="store_true",
                   help="aceita pacote gerado com árvore git modificada (o commit ainda deve conferir)")
    p.add_argument("--import-check", action=argparse.BooleanOptionalAction, default=True,
                   help="reimporta o pacote e roda PRAGMA foreign_key_check (padrão: sim)")
    return p


def _validate_args(args):
    """Erros de uso (rc=2) antes de qualquer leitura."""
    for value in (args.expect_start, args.expect_end, args.expect_grid_origin,
                  None if (args.expect_tracked_since or "").lower() == "none" else args.expect_tracked_since):
        if value is not None:
            parse_arg(value)
    if args.expect_nodes not in (None, "all"):
        [int(n) for n in args.expect_nodes.split(",")]
    if args.expect_commit is not None and resolve_commit(args.expect_commit) is None:
        raise ValueError(f"--expect-commit não resolve para um commit: {args.expect_commit!r}")
    if not args.package.is_file():
        raise FileNotFoundError(f"pacote inexistente: {args.package}")
    if not args.sqlite.is_file():
        raise FileNotFoundError(f"banco de origem inexistente: {args.sqlite}")
    if args.compare is not None and not args.compare.is_file():
        raise FileNotFoundError(f"pacote de comparação inexistente: {args.compare}")
    zipfile.ZipFile(args.package).close()  # ZIP ilegível → erro de execução


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    a = Audit()
    try:
        _validate_args(args)
        args.source_sha256 = sha256_path(args.sqlite)
        verified = True
        try:
            manifest, root = verify_package(args.package)
            a.add("package:verify", "verify_package (estrutura, hashes, contagens)", True, True)
        except PackageError as exc:
            verified = False
            a.add("package:verify", "verify_package (estrutura, hashes, contagens)", True, str(exc), ok=False)
            try:
                manifest, root = read_manifest_unverified(args.package)
            except (PackageError, KeyError, ValueError) as inner:
                manifest, root = None, None
                a.add("manifest:readable", "manifest.json legível na raiz do pacote", True,
                      f"{type(inner).__name__}: {inner}", ok=False)
        if manifest is not None:
            run_audit(a, args, manifest, root, verified)
        a.info("audit:compared", "itens efetivamente comparados por verificação (0 = verificação sem objeto)",
               a.compared)
        report = {
            "package": str(args.package), "package_sha256": sha256_path(args.package),
            "source_sqlite": str(args.sqlite), "source_sha256": args.source_sha256,
            "dataset_id": (manifest or {}).get("dataset_id"), "export_kind": (manifest or {}).get("export_kind"),
            "request": (manifest or {}).get("request"), "interpretation": (manifest or {}).get("interpretation"),
            "audited_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "code_revision": resolve_commit(None),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "summary": dict(Counter(c["status"] for c in a.checks)),
            "checks": a.checks,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        rc = 1 if report["summary"].get("fail") else 0
        if args.compare:
            cmp = compare_packages(args.package, args.compare)
            (args.compare_out or args.out.with_name("COMPARE.json")).write_text(
                json.dumps(cmp, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            rc = rc or (0 if cmp["ok"] else 1)
        print(f"auditoria: {report['summary']} → {args.out}")
        return rc
    except Exception as exc:  # uso inválido, pacote ou banco ilegível
        print(f"erro: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
