"""Configuração compartilhada de testes."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from datetime import datetime as _dt, timedelta as _td
import pytest as _pytest


@_pytest.fixture(autouse=True)
def _isolate_online_settings(monkeypatch, tmp_path):
    """Um .env local (ex.: modelos congelados) não altera a suíte; testes
    que precisam de outro valor fazem o próprio monkeypatch."""
    from config import settings
    monkeypatch.setattr(settings, "ML_ONLINE_FREEZE_MODELS", False)
    monkeypatch.setattr(settings, "ML_ONLINE_BUNDLE_DIR", str(tmp_path / "online_models"))

# ─── Banco de origem para exportação ───────────────────────────────────────────
EXPORT_T0 = _dt(2026, 9, 1, 0, 0, 0)


def build_export_source(path):
    """Origem SQLite com contexto, enlace reverso entre redes, simulações e textos difíceis.

    Rede 1: nós 1 (0x0001), 2 (0x0002), 4 (0x0004, sem dados). Rede 2: nó 3 (0x0003),
    que ouve o nó 1 (enlace reverso de contexto). Dados de T0−1h a T0+3h, 1/min.
    Simulações: 10 (nó 2, anterior; só entra pelo fecho via node.current_sim_run_id),
    11 (nó 1, dentro do período, linhas sintéticas), 12 (nó 3, aberta), 13 (nó 1, distante).
    """
    from sqlmodel import Session, SQLModel, create_engine
    import models  # noqa: F401
    from models import Network, Node, Neighbor, Telemetry, SimulationRun

    T0 = EXPORT_T0
    eng = create_engine(f"sqlite:///{path}")
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(Network(id=1, name='Rede A, "lab"'))
        s.add(Network(id=2, name="Rede B"))
        s.add(Node(node_id=1, network_id=1, name='Nó, Ação "1"', lora_address="0x0001", status="online"))
        s.add(Node(node_id=2, network_id=1, name="N2", lora_address="0x0002", status="online"))
        s.add(Node(node_id=3, network_id=2, name="N3", lora_address="0x0003", status="online"))
        s.add(Node(node_id=4, network_id=1, name="N4 sem dados", lora_address="0x0004", status="offline"))
        s.commit()
        s.add(SimulationRun(id=10, node_id=2, anomaly="A2", duration_planned=3600,
                            started_at=T0 - _td(days=2), ended_at=T0 - _td(days=2) + _td(hours=1)))
        s.add(SimulationRun(id=11, node_id=1, anomaly="A1", duration_planned=1800,
                            started_at=T0 + _td(minutes=30), ended_at=T0 + _td(hours=1)))
        s.add(SimulationRun(id=12, node_id=3, anomaly="A3", duration_planned=7200,
                            started_at=T0 - _td(days=1), ended_at=None))
        s.add(SimulationRun(id=13, node_id=1, anomaly="A1", duration_planned=3600,
                            started_at=T0 - _td(days=10), ended_at=T0 - _td(days=9)))
        s.commit()
        node2 = s.get(Node, 2)
        node2.current_sim_run_id = 10
        s.add(node2)
        seq = {1: 0, 2: 0, 3: 0}
        payload = '{"msg": "a,b", "q": "x\\"y"}\nsegunda linha'
        for minute in range(-60, 181):
            ts = T0 + _td(minutes=minute, seconds=20, microseconds=123456)
            for node, peers in ((1, ("0x0002", "0x0003")), (2, ("0x0001",)), (3, ("0x0001",))):
                origin = "synthetic" if node == 1 and 30 <= minute < 60 else "real"
                seq[node] += 1
                for k, peer in enumerate(peers):
                    s.add(Neighbor(node_id=node, neighbor_address=peer,
                                   rssi=-70 - node - k - (minute % 5),
                                   snr=7.25 + 0.5 * (minute % 3), link_quality=8 + (minute % 2),
                                   is_parent=(k == 0), timestamp=ts, data_origin=origin,
                                   distance_m=12.5 if k == 0 else None))
                s.add(Telemetry(node_id=node, timestamp=ts, rssi=-75, snr=8.125,
                                sequence=seq[node], uptime=seq[node],
                                battery_level=90 - minute // 60, payload_raw=payload,
                                data_origin=origin,
                                temperature=None if minute % 7 else 21.5))
        s.commit()
    eng.dispose()


def export_request(**over):
    from dataset_export.request import ExportRequest
    base = dict(source_id="lab-teste", network_id=1, node_ids=(1, 2, 4),
                start=EXPORT_T0, end=EXPORT_T0 + _td(hours=2),
                window_seconds=900, step_seconds=420)
    base.update(over)
    return ExportRequest(**base)


@_pytest.fixture(scope="session")
def export_source(tmp_path_factory):
    path = tmp_path_factory.mktemp("export_src") / "origem.db"
    build_export_source(path)
    return path
