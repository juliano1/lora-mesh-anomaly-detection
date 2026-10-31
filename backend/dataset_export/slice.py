"""
Recorte da origem, fecho referencial e materialização.

Recorte (intervalo (data_start, end]):
- telemetria dos nós solicitados;
- vizinhos relatados pelos nós solicitados e enlaces reversos SOBRE eles
  (qualquer nó que relata o endereço de um solicitado);
- nós: solicitados, relatores reversos e donos dos endereços ouvidos;
- simulações desses nós que intersectam o período ± margem de referência;
- fecho referencial: node.network_id, node.current_sim_run_id e
  simulation_run.node_id, até não surgirem referências novas.
Linhas são copiadas sem alteração (mesmos IDs e valores) para um SQLite
temporário; `finalize_sqlite` cria índices compostos e roda
`PRAGMA foreign_key_check` (deve retornar vazio). O mesmo `write_rows` serve
ao importador.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator
import math

from sqlalchemy import create_engine, or_, select, Float
from sqlmodel import SQLModel

EXPORT_TABLES = ("network", "node", "simulation_run", "telemetry", "neighbor")
BATCH = 5000
COMPOSITE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS ix_spec064_neighbor_node_ts ON neighbor (node_id, timestamp)",
    "CREATE INDEX IF NOT EXISTS ix_spec064_telemetry_node_ts ON telemetry (node_id, timestamp)",
    "CREATE INDEX IF NOT EXISTS ix_spec064_neighbor_node_addr_ts "
    "ON neighbor (node_id, neighbor_address, timestamp)",
)


class ExportError(Exception):
    """Pedido inválido ou recorte impossível de montar de forma íntegra."""


def table(name: str):
    import models  # noqa: F401 — registra as tabelas
    return SQLModel.metadata.tables[name]


@dataclass
class SliceSummary:
    requested_node_ids: list
    context_node_ids: list
    network_ids: list
    simulation_run_ids: list
    data_start: datetime
    end: datetime
    rows: dict = field(default_factory=dict)


def create_target(path):
    """SQLite novo com o esquema atual das tabelas exportadas."""
    path = Path(path)
    if path.exists():
        raise ExportError(f"{path} já existe")
    engine = create_engine(f"sqlite:///{path}")
    SQLModel.metadata.create_all(engine, tables=[table(n) for n in EXPORT_TABLES])
    return engine


def write_rows(engine, table_name: str, rows: Iterable[dict]) -> int:
    """Insere linhas (dict coluna→valor) em lotes, sem alterar valores nem IDs."""
    target = table(table_name)
    columns = [c.name for c in target.columns]
    written, batch = 0, []
    with engine.begin() as connection:
        for row in rows:
            for column in target.columns:
                value = row[column.name]
                if isinstance(column.type, Float) and value is not None and not math.isfinite(value):
                    raise ExportError(f"{table_name}.{column.name}: valor não finito na origem")
            batch.append({c: row[c] for c in columns})
            if len(batch) >= BATCH:
                connection.execute(target.insert(), batch)
                written += len(batch)
                batch = []
        if batch:
            connection.execute(target.insert(), batch)
            written += len(batch)
    return written


def finalize_sqlite(engine) -> list:
    """Índices compostos + `PRAGMA foreign_key_check`. Retorna as violações."""
    with engine.begin() as connection:
        for ddl in COMPOSITE_INDEXES:
            connection.exec_driver_sql(ddl)
        violations = connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
        connection.exec_driver_sql("ANALYZE")
    return [tuple(v) for v in violations]


def _stream(session, statement) -> Iterator[dict]:
    result = session.execute(statement.execution_options(yield_per=BATCH))
    try:
        for row in result.mappings():
            yield dict(row)
    finally:
        result.close()


def _by_key(session, name: str, key: str, values) -> dict:
    values = sorted(v for v in values if v is not None)
    if not values:
        return {}
    t = table(name)
    return {r[key]: r for r in _stream(session, select(t).where(t.c[key].in_(values)))}


def materialize_slice(session, request, target_engine, margin_minutes: float) -> SliceSummary:
    node_t, nb_t, tel_t, sim_t = (table(n) for n in ("node", "neighbor", "telemetry", "simulation_run"))
    lo, hi = request.data_start(), request.end

    in_network = {r["node_id"]: r for r in _stream(
        session, select(node_t).where(node_t.c.network_id == request.network_id))}
    if not in_network:
        raise ExportError(f"rede {request.network_id} sem nós na origem")
    if request.node_ids is None:
        requested = sorted(in_network)
    else:
        outside = sorted(set(request.node_ids) - set(in_network))
        if outside:
            raise ExportError(f"nós fora da rede {request.network_id}: {outside}")
        requested = sorted(set(request.node_ids))
    requested_set = set(requested)
    addresses = sorted({in_network[n]["lora_address"] for n in requested
                        if in_network[n]["lora_address"]})

    counts = {"telemetry": write_rows(target_engine, "telemetry", _stream(session, select(tel_t).where(
        tel_t.c.node_id.in_(requested), tel_t.c.timestamp > lo, tel_t.c.timestamp <= hi)))}

    reporters, heard = set(), set()

    def _neighbors():
        statement = select(nb_t).where(
            or_(nb_t.c.node_id.in_(requested), nb_t.c.neighbor_address.in_(addresses)),
            nb_t.c.timestamp > lo, nb_t.c.timestamp <= hi,
        )
        for row in _stream(session, statement):
            if row["node_id"] in requested_set:
                heard.add(row["neighbor_address"])
            else:
                reporters.add(row["node_id"])
            yield row

    counts["neighbor"] = write_rows(target_engine, "neighbor", _neighbors())

    nodes = {r["node_id"]: r for r in _stream(session, select(node_t).where(or_(
        node_t.c.node_id.in_(sorted(requested_set | reporters)),
        node_t.c.lora_address.in_(sorted(heard)),
    )))}
    margin = timedelta(minutes=margin_minutes)
    sims = {r["id"]: r for r in _stream(session, select(sim_t).where(
        sim_t.c.node_id.in_(sorted(nodes)),
        sim_t.c.started_at < hi + margin,
        or_(sim_t.c.ended_at.is_(None), sim_t.c.ended_at > lo - margin),
    ))}
    networks: dict = {}

    while True:  # fecho referencial
        need_net = {n["network_id"] for n in nodes.values()} - set(networks) - {None}
        need_sim = {n["current_sim_run_id"] for n in nodes.values()} - set(sims) - {None}
        need_node = {s["node_id"] for s in sims.values()} - set(nodes)
        before = (len(networks), len(sims), len(nodes))
        networks.update(_by_key(session, "network", "id", need_net))
        sims.update(_by_key(session, "simulation_run", "id", need_sim))
        nodes.update(_by_key(session, "node", "node_id", need_node))
        if (len(networks), len(sims), len(nodes)) == before:
            break  # nada novo: referências ausentes na origem aparecem no FK check

    counts["node"] = write_rows(target_engine, "node", nodes.values())
    counts["network"] = write_rows(target_engine, "network", networks.values())
    counts["simulation_run"] = write_rows(target_engine, "simulation_run", sims.values())
    return SliceSummary(
        requested_node_ids=requested,
        context_node_ids=sorted(set(nodes) - requested_set),
        network_ids=sorted(networks),
        simulation_run_ids=sorted(sims),
        data_start=lo,
        end=hi,
        rows=counts,
    )
