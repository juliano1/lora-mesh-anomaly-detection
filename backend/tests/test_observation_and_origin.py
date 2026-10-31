"""Ausência de observação, bateria por envio condicional, origem sintética e A6 do enlace."""
import json
import threading
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

T = datetime(2026, 9, 1, 12, 0, 0)


def _n(addr, minute, rssi=-80.0, parent=True):
    return SimpleNamespace(neighbor_address=addr, rssi=rssi, snr=8.0, link_quality=9,
                           is_parent=parent, timestamp=T - timedelta(minutes=minute))


def _t(minute, seq, bat=None):
    return SimpleNamespace(timestamp=T - timedelta(minutes=minute), sequence=seq, uptime=seq,
                           battery_level=bat)


def _window(telemetries, baseline=None):
    from ai.feature_engineering import compute_window_features
    nbs = [_n("0x0002", m) for m in (4, 3, 2)]
    return compute_window_features(nbs, telemetries, {"0x0002": [_n("0x0001", 3, -82.0)]},
                                   battery_baseline=baseline, window_end=T)


# ─── Bateria: envio condicional do firmware (2 p.p. ou 5 min) ────────────────

def test_single_reading_uses_the_reading_before_the_window():
    f = _window([_t(4, 1, bat=78), _t(3, 2), _t(2, 3)], baseline=_t(7, 0, bat=80))
    assert f["battery_delta"] == -2.0 and "battery_delta" not in f["_unavailable"]
    assert f["_battery_first_at"] == T - timedelta(minutes=7)


def test_no_reading_in_window_with_fresh_baseline_is_an_observed_zero():
    # Sem envio na janela e última leitura com ≤ 60 min: nenhuma mudança informada.
    f = _window([_t(4, 1), _t(3, 2), _t(2, 3)], baseline=_t(59, 0, bat=80))
    assert f["battery_delta"] == 0.0 and "battery_delta" not in f["_unavailable"]


def test_stale_baseline_or_single_reading_without_baseline_is_unavailable():
    from ai.feature_engineering import INSUFFICIENT_BATTERY_READINGS
    stale = _window([_t(4, 1), _t(3, 2), _t(2, 3)], baseline=_t(61, 0, bat=80))
    lonely = _window([_t(4, 1, bat=80), _t(3, 2), _t(2, 3)])
    for f in (stale, lonely):
        assert f["_unavailable"]["battery_delta"] == INSUFFICIENT_BATTERY_READINGS


def test_window_with_unobserved_feature_never_reaches_the_models():
    from ai.sequence_builder import feature_dict_to_vector
    f = _window([_t(4, 1, bat=80)])             # 1 contador: perda e intervalo não observados
    vec, reason = feature_dict_to_vector(f)
    assert vec is None and reason.startswith("missing_observation:")
    assert f["packet_loss_rate"] == 0.0         # o 0 de apresentação não vira medida


def test_database_extractor_reads_the_battery_before_the_window():
    from models import Neighbor, Node, Telemetry
    from ai.feature_engineering import FeatureExtractor
    s = _db()
    s.add(Node(node_id=1, network_id=1, lora_address="0x0001"))
    s.add(Node(node_id=2, network_id=1, lora_address="0x0002"))
    for m in (4, 3, 2):
        s.add(Neighbor(node_id=1, neighbor_address="0x0002", rssi=-80, snr=8.0, link_quality=9,
                       is_parent=True, timestamp=T - timedelta(minutes=m)))
        s.add(Neighbor(node_id=2, neighbor_address="0x0001", rssi=-82, snr=8.0, link_quality=9,
                       timestamp=T - timedelta(minutes=m, seconds=10)))
        s.add(Telemetry(node_id=1, rssi=-80, snr=8.0, timestamp=T - timedelta(minutes=m), sequence=10 - m, uptime=1,
                        battery_level=70 if m == 3 else None, payload_raw="{}"))
    s.add(Telemetry(node_id=1, rssi=-80, snr=8.0, timestamp=T - timedelta(minutes=8), sequence=5, uptime=1,
                    battery_level=73, payload_raw="{}", data_origin="synthetic"))
    s.commit()
    prov: dict = {}
    f = FeatureExtractor().extract_for_window(s, 1, T - timedelta(minutes=5), T, provenance=prov)
    assert f["battery_delta"] == -3.0
    # A leitura anterior conta na proveniência e na política de referência.
    assert prov["synthetic_rows"] == 1 and prov["context_start"] == T - timedelta(minutes=8)


# ─── Origem sintética no contexto da decisão ─────────────────────────────────

def _db():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    import models  # noqa: F401
    SQLModel.metadata.create_all(eng)
    return Session(eng)


def test_synthetic_rows_are_counted_without_a_simulation_run():
    from models import Neighbor, Node, Telemetry
    from ai.anomaly_detector import _data_origin_evidence
    s = _db()
    s.add(Node(node_id=1, network_id=1, lora_address="0x0001"))
    s.add(Telemetry(node_id=1, rssi=-80, snr=8.0, timestamp=T - timedelta(minutes=50), sequence=1, uptime=1,
                    payload_raw="{}", data_origin="synthetic"))
    s.add(Neighbor(node_id=2, neighbor_address="0x0001", rssi=-80, snr=8.0,
                   timestamp=T - timedelta(minutes=5), data_origin="synthetic"))     # reverso
    s.add(Telemetry(node_id=1, rssi=-80, snr=8.0, timestamp=T - timedelta(minutes=70), sequence=2, uptime=1,
                    payload_raw="{}", data_origin="synthetic"))                     # fora
    s.commit()
    origin = _data_origin_evidence(s, 1, T - timedelta(minutes=60), T)
    assert origin["synthetic_rows"] == 2


@pytest.mark.parametrize("sim_run_id, evidence, expected", [
    (None, {"data_origin": {"synthetic_rows": 3}}, (True, "window_rows")),
    (7, {"data_origin": {"synthetic_rows": 0}}, (True, "sim_run")),
    (None, {"data_origin": {"synthetic_rows": 0}}, (False, "window_rows")),
    (None, {}, (None, "not_recorded")),              # registro antigo: não verificado
])
def test_is_synthetic_uses_window_rows_and_run(sim_run_id, evidence, expected):
    from main import _synthetic_fields
    a = SimpleNamespace(sim_run_id=sim_run_id, evidence=json.dumps(evidence))
    got = _synthetic_fields(a)
    assert (got["is_synthetic"], got["synthetic_basis"]) == expected


def test_online_emission_records_synthetic_context():
    from ai.anomaly_detector import AnomalyDetector
    from ai.online.combiner import ComboOutcome
    from ai.online.pipeline import PipelineOutcome
    from ai.sequence_builder import FEATURE_NAMES
    from models import Node, Telemetry
    s = _db()
    s.add(Node(node_id=1, network_id=7, lora_address="0x0001"))
    s.add(Telemetry(node_id=1, rssi=-80, snr=8.0, timestamp=T - timedelta(minutes=40), sequence=1, uptime=1,
                    payload_raw="{}", data_origin="synthetic"))
    s.commit()
    d = object.__new__(AnomalyDetector)
    d._models_lock = threading.Lock()
    d._online_combo, d._online_primary = "single", "lstm_ae"
    pipeline = MagicMock(fusion_alpha=0.5, tau=None, tau_status=None, bundle_id=None,
                         model_signature=None, _secondary_runner=None)
    pipeline.detect_detailed.return_value = PipelineOutcome(
        "lstm_ae", None, "single", None, None, ComboOutcome("valid"),
        [{"score_norm": 0.9, "evidence": {}}])
    d._get_pipeline_for_network = MagicMock(return_value=pipeline)
    saved = []
    d._save_and_notify = lambda session, **kw: saved.append(kw["anomaly"])
    fe = MagicMock()
    fe.extract_features.return_value = {name: 0.0 for name in FEATURE_NAMES} | {"rssi_mean": -105.0}
    d._run_online_pipeline(s, 7, fe, [1], None, reference_time=T)
    origin = saved[0]["evidence"]["data_origin"]
    # Contexto = sequência do LSTM (60 min), maior que a janela do IF (15 min).
    assert origin["synthetic_rows"] == 1
    assert origin["context_start"] == (T - timedelta(minutes=60)).isoformat()


# ─── A6: assimetria do enlace, sem nó de origem ──────────────────────────────

def test_a6_names_the_link_and_does_not_pick_a_culprit():
    from ai.anomaly_ml import classify_anomaly_type
    from ai.anomaly_types import get_anomaly_def
    from ai.feature_engineering import compute_window_features
    nbs = [_n("0x0002", m, -80.0) for m in (4, 3, 2)] + [_n("0x0003", m, -70.0, False) for m in (4, 3)]
    reverse = {"0x0002": [_n("0x0001", 3, -100.0)], "0x0003": [_n("0x0001", 3, -71.0)]}
    tel = [_t(4, 1, bat=80), _t(3, 2), _t(2, 3, bat=80)]
    f = compute_window_features(nbs, tel, reverse, window_end=T)
    assert f["link_asymmetry"] > 10.0
    f["link_asymmetry"] = 16.0                       # garante o corte de A6 (> 15 dB)
    a6 = [a for a in classify_anomaly_type(f, -0.4) if a["anomaly_type"] == "A6"][0]
    assert a6["neighbor_address"] == "0x0002"        # enlace com a maior diferença
    assert a6["evidence"]["origin"] == "not_determined"
    assert [link["neighbor"] for link in a6["evidence"]["asymmetric_links"]] == ["0x0002", "0x0003"]
    assert "qualquer um dos dois nós" in a6["description"]
    assert a6["category"] == get_anomaly_def("A6").category.value == "radio"
