"""
Referência saudável para treino/calibração.

O estado atual `Node.sim_mode` não determina a origem do histórico. Esta política
reúne TODOS os intervalos de `SimulationRun` (inclusive encerrados) e, como
complemento legado das versões antigas, o par `Node.sim_started_at/sim_ended_at` (que só
guarda a última simulação). Uma janela de referência é rejeitada quando:

- contém linhas com `data_origin != 'real'` (vizinhos, telemetria ou enlace reverso)
  → motivo `synthetic_rows`;
- intersecta um intervalo de simulação do próprio nó ou de qualquer nó cujos
  dados alimentaram as features (enlaces reversos), expandido pela margem
  contextual → motivo `simulation_overlap`.

Intervalos são semiabertos e compatíveis com a convenção `(início, fim]` das
janelas: uma janela que termina exatamente no início da simulação não é afetada.
Regras de treino e de inferência são distintas: a inferência experimental
avalia intervenções e não usa esta política.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Iterable, Optional

SIMULATION_OVERLAP = "simulation_overlap"
SYNTHETIC_ROWS = "synthetic_rows"


class ReferencePolicy:
    def __init__(self, intervals: Optional[dict] = None, margin_minutes: float = 0.0,
                 source: str = "explicit"):
        # (início, fim, id da simulação ou "legacy_node:<id>"; None se informado sem id)
        self._intervals: dict[int, list[tuple]] = defaultdict(list)
        for node_id, spans in (intervals or {}).items():
            for span in spans:
                start, end, *rest = span
                self._intervals[int(node_id)].append((start, end, rest[0] if rest else None))
        self.margin = timedelta(minutes=float(margin_minutes))
        self.source = source

    @classmethod
    def empty(cls) -> "ReferencePolicy":
        return cls({}, source="empty")

    @classmethod
    def from_session(cls, session, margin_minutes: float = 0.0) -> "ReferencePolicy":
        from sqlmodel import select
        from models import Node, SimulationRun

        intervals: dict[int, list] = defaultdict(list)
        for run in session.exec(select(SimulationRun)).all():
            intervals[run.node_id].append((run.started_at, run.ended_at, run.id))
        # Legado das versões antigas: só a última simulação por nó. Fim ausente fora de
        # sim_mode é desconhecido → não inventamos duração e o intervalo é ignorado.
        legacy = session.exec(select(Node).where(Node.sim_started_at != None)).all()  # noqa: E711
        for node in legacy:
            if node.sim_ended_at is not None and node.sim_ended_at >= node.sim_started_at:
                intervals[node.node_id].append((node.sim_started_at, node.sim_ended_at, f"legacy_node:{node.node_id}"))
            elif node.sim_mode:
                intervals[node.node_id].append((node.sim_started_at, None, f"legacy_node:{node.node_id}"))
        return cls(intervals, margin_minutes=margin_minutes, source="simulation_run+node_legacy")

    def overlapping_runs(self, node_ids: Iterable[int], start: datetime, end: datetime) -> list:
        """IDs das simulações (ou `legacy_node:<id>`) que intersectam a janela ± margem.

        Inteiros primeiro, em ordem numérica; depois marcadores do legado.
        """
        lo = start - self.margin
        hi = end + self.margin
        found = []
        for node_id in node_ids:
            for run_start, run_end, run_id in self._intervals.get(int(node_id), ()):
                if run_start < hi and (run_end is None or run_end > lo) and run_id not in found:
                    found.append(run_id)
        return sorted(found, key=lambda r: (0, r, "") if isinstance(r, int) else (1, 0, str(r)))

    def node_contaminated(self, node_ids: Iterable[int], start: datetime, end: datetime) -> bool:
        return bool(self.overlapping_runs(node_ids, start, end))

    def check_window(self, node_id: int, start: datetime, end: datetime,
                     provenance: dict) -> Optional[str]:
        """Motivo de exclusão da janela de referência, ou None se aceitável."""
        if provenance.get("synthetic_rows", 0) > 0:
            return SYNTHETIC_ROWS
        sources = set(provenance.get("source_node_ids", ())) | {node_id}
        # A leitura de bateria anterior à janela também alimenta as features.
        context_start = provenance.get("context_start") or start
        if self.node_contaminated(sources, min(context_start, start), end):
            return SIMULATION_OVERLAP
        return None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "margin_minutes": self.margin.total_seconds() / 60.0,
            "n_intervals": sum(len(v) for v in self._intervals.values()),
        }
