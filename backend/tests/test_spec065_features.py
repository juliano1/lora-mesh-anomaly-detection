"""Núcleo puro, desvios por enlace e disponibilidade."""
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

T = datetime(2026, 9, 1, 12, 0, 0)


def _n(addr, minute, rssi, snr=8.0, lq=9, parent=False):
    return SimpleNamespace(neighbor_address=addr, rssi=rssi, snr=snr, link_quality=lq,
                           is_parent=parent, timestamp=T - timedelta(minutes=minute))


def _t(minute, seq, bat=80, uptime=None):
    return SimpleNamespace(timestamp=T - timedelta(minutes=minute), sequence=seq,
                           uptime=seq if uptime is None else uptime, battery_level=bat)


def test_constant_links_with_different_means_have_zero_temporal_std():
    from ai.feature_engineering import compute_window_features
    nbs = [_n("0x0002", m, -70.0) for m in (10, 8, 6)] + [_n("0x0003", m, -90.0) for m in (9, 7, 5)]
    f = compute_window_features(nbs, [], {})
    assert f["rssi_std"] == 0.0 and f["snr_std"] == 0.0
    assert f["rssi_mean"] == -80.0
    assert "rssi_std" not in f["_unavailable"]


def test_shifting_constant_mean_of_one_link_does_not_change_its_std():
    from ai.feature_engineering import compute_window_features
    a = [_n("0x0002", 10, -70.0), _n("0x0002", 8, -74.0), _n("0x0003", 9, -90.0), _n("0x0003", 7, -90.0)]
    b = [_n("0x0002", 10, -70.0), _n("0x0002", 8, -74.0), _n("0x0003", 9, -60.0), _n("0x0003", 7, -60.0)]
    assert compute_window_features(a, [], {})["rssi_std"] == compute_window_features(b, [], {})["rssi_std"]


def test_variable_link_has_positive_std_mean_without_weighting():
    from ai.feature_engineering import compute_window_features
    nbs = [_n("0x0002", 10, -70.0), _n("0x0002", 8, -74.0),          # std 2.0
           _n("0x0003", 9, -90.0), _n("0x0003", 7, -90.0), _n("0x0003", 5, -90.0)]  # std 0.0
    f = compute_window_features(nbs, [], {})
    assert f["rssi_std"] == pytest.approx(1.0)          # (2 + 0) / 2, sem ponderação
    assert f["_link_stats"]["eligible_links"] == 2


def test_no_eligible_link_gives_none_and_reason():
    from ai.feature_engineering import compute_window_features, NO_ELIGIBLE_LINK
    f = compute_window_features([_n("0x0002", 10, -70.0), _n("0x0003", 9, -90.0)], [], {})
    assert f["rssi_std"] is None and f["snr_std"] is None
    assert f["_unavailable"]["rssi_std"] == NO_ELIGIBLE_LINK
    assert f["_unavailable"]["snr_std"] == NO_ELIGIBLE_LINK


def test_non_finite_pairs_are_not_eligible():
    from ai.feature_engineering import compute_window_features
    nbs = [_n("0x0002", 10, -70.0), _n("0x0002", 8, float("nan")), _n("0x0003", 9, -90.0)]
    f = compute_window_features(nbs, [], {})
    assert f["rssi_std"] is None


def test_packet_loss_needs_two_distinct_counters():
    from ai.feature_engineering import packet_loss_estimate
    assert packet_loss_estimate([_t(5, 7), _t(4, 7)]) == (0.0, False)     # duplicata
    assert packet_loss_estimate([_t(5, 7)]) == (0.0, False)
    rate, ok = packet_loss_estimate([_t(5, 1), _t(4, 2), _t(3, 4)])
    assert ok and rate == pytest.approx(1 / 3)


def test_unavailable_covers_defaults():
    from ai.feature_engineering import compute_window_features
    nbs = [_n("0x0002", 10, -70.0, lq=None), _n("0x0002", 8, -71.0, lq=None)]
    f = compute_window_features(nbs, [_t(5, 1, bat=-1)], {})
    assert f["pdr_mean"] == 5.0 and f["_unavailable"]["pdr_mean"] == "no_link_quality"
    assert f["battery_delta"] == 0.0
    assert f["_unavailable"]["battery_delta"] == "insufficient_battery_readings"
    assert f["_unavailable"]["packet_loss_rate"] == "insufficient_distinct_counters"
    assert f["_unavailable"]["beacon_interval_std"] == "insufficient_telemetry"
    assert f["_unavailable"]["route_change_rate"] == "insufficient_parent_records"
    assert f["_unavailable"]["link_asymmetry"] == "no_reverse_link"


def test_battery_interval_is_exposed_outside_vector():
    from ai.feature_engineering import compute_window_features
    nbs = [_n("0x0002", 10, -70.0), _n("0x0002", 8, -71.0)]
    f = compute_window_features(nbs, [_t(12, 1, bat=75), _t(2, 2, bat=50)], {})
    assert f["battery_delta"] == -25.0
    assert f["_battery_first_at"] == T - timedelta(minutes=12)
    assert f["_battery_last_at"] == T - timedelta(minutes=2)


@pytest.fixture
def db():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    import models  # noqa: F401
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        from models import Node
        s.add(Node(node_id=1, network_id=1, lora_address="0x0001"))
        s.add(Node(node_id=2, network_id=1, lora_address="0x0002"))
        s.commit()
        yield s


def _row(node_id, addr, minute, rssi, origin="real"):
    from models import Neighbor
    return Neighbor(node_id=node_id, neighbor_address=addr, rssi=rssi, snr=8.0, link_quality=9,
                    timestamp=T - timedelta(minutes=minute), data_origin=origin)


def test_extractor_keeps_provenance_counts_and_sources(db):
    from ai.feature_engineering import FeatureExtractor
    db.add(_row(1, "0x0002", 10, -80)); db.add(_row(1, "0x0002", 5, -81, origin="synthetic"))
    db.add(_row(2, "0x0001", 7, -79))
    db.commit()
    prov = {}
    f = FeatureExtractor().extract_for_window(db, 1, T - timedelta(minutes=15), T, provenance=prov)
    assert prov["synthetic_rows"] == 1 and prov["source_node_ids"] == {1, 2}
    assert prov["neighbor_record_count"] == 2
    assert "battery_delta" in prov["defaulted_features"]
    assert "rssi_std" not in prov["defaulted_features"]     # None não é default numérico
    assert f["link_asymmetry"] == pytest.approx(abs(-80.5 - -79.0))


NINE = ["rssi_mean", "snr_mean", "pdr_mean", "battery_delta", "packet_loss_rate",
        "rssi_snr_divergence", "beacon_interval_std", "route_change_rate", "link_asymmetry"]
GOLDEN_WINDOWS = [(T - timedelta(minutes=30), T - timedelta(minutes=15)),
                  (T - timedelta(minutes=15), T)]
# Saída do extrator anterior (bateria medida só dentro da janela) para a fixture abaixo.
GOLDEN_V1 = [
    {"rssi_mean": -83.62068965517241, "snr_mean": 3.9655172413793105, "pdr_mean": 5.862068965517241,
     "battery_delta": -13.0, "packet_loss_rate": 0.15384615384615385,
     "rssi_snr_divergence": 0.3478456784988633, "beacon_interval_std": 21.077189895341775,
     "route_change_rate": 0.5333333333333333, "link_asymmetry": 1.0},
    {"rssi_mean": -84.48275862068965, "snr_mean": 3.5689655172413794, "pdr_mean": 5.655172413793103,
     "battery_delta": -14.0, "packet_loss_rate": 0.1428571428571429,
     "rssi_snr_divergence": 0.3446277307712562, "beacon_interval_std": 21.379344912529216,
     "route_change_rate": 0.42857142857142855, "link_asymmetry": 0.8571428571428612},
]


def build_golden_db():
    """Fixture determinística: 2 enlaces com médias distintas, enlace reverso,
    parent alternando, lacunas e descontinuidade de contador, bateria em queda."""
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    import models  # noqa: F401
    SQLModel.metadata.create_all(eng)
    s = Session(eng)
    from models import Neighbor, Node, Telemetry
    for nid in (1, 2, 3):
        s.add(Node(node_id=nid, network_id=1, lora_address=f"0x000{nid}"))
    for m in range(1, 30):
        ts = T - timedelta(minutes=m, seconds=7 * (m % 3))
        s.add(Neighbor(node_id=1, neighbor_address="0x0002", rssi=-70 - (m % 5), snr=7.5 + 0.5 * (m % 3),
                       link_quality=8 + (m % 2), is_parent=(m % 4 != 0), timestamp=ts))
        s.add(Neighbor(node_id=1, neighbor_address="0x0003", rssi=-95 - (m % 3), snr=1.0 - (m % 4),
                       link_quality=3, is_parent=(m % 4 == 0), timestamp=ts + timedelta(seconds=20)))
        if m % 2:
            s.add(Neighbor(node_id=2, neighbor_address="0x0001", rssi=-72 - (m % 2), snr=8.0,
                           link_quality=8, timestamp=ts + timedelta(seconds=5)))
        if m % 6:
            seq = 1 if m == 12 else 40 - m
            s.add(Telemetry(node_id=1, timestamp=ts + timedelta(seconds=40), rssi=-80, snr=8.0,
                            sequence=seq, uptime=seq, battery_level=90 - (29 - m), payload_raw="{}"))
    s.commit()
    return s


def test_nine_features_match_baseline_golden():
    from ai.feature_engineering import FeatureExtractor
    assert GOLDEN_V1 is not None, "GOLDEN_V1 ausente: gere com o extrator anterior"
    s = build_golden_db()
    fe = FeatureExtractor()
    for (ws, we), expected in zip(GOLDEN_WINDOWS, GOLDEN_V1):
        f = fe.extract_for_window(s, 1, ws, we)
        same = [k for k in NINE if k != "battery_delta"]
        assert {k: f[k] for k in same} == pytest.approx({k: expected[k] for k in same}, abs=1e-12)
    # Features v4: a bateria parte da última leitura anterior à janela (até 60 min antes).
    # Na 1ª janela não há leitura anterior e o valor é o do extrator anterior; na 2ª, a leitura
    # de 11:44:33 (77) é a referência, e não a primeira da janela (76).
    (ws1, we1), (ws2, we2) = GOLDEN_WINDOWS
    assert fe.extract_for_window(s, 1, ws1, we1)["battery_delta"] == GOLDEN_V1[0]["battery_delta"]
    second = fe.extract_for_window(s, 1, ws2, we2)
    assert second["battery_delta"] == -15.0 and GOLDEN_V1[1]["battery_delta"] == -14.0
    assert second["_battery_first_at"] == datetime(2026, 9, 1, 11, 44, 33) <= ws2


def test_extractor_none_std_rejects_sequence_not_forward_fill(db):
    from ai.feature_engineering import FeatureExtractor
    from ai.sequence_builder import SequenceBuilder
    # sub-janela 0 (11:00–11:05] válida; sub-janela 1 (11:05–11:10] com 2 vizinhos distintos, 1 obs cada
    for m in (59, 57):
        db.add(_row(1, "0x0002", m, -80))
    db.add(_row(1, "0x0002", 54, -80)); db.add(_row(1, "0x0003", 53, -85))
    db.commit()
    from ai.sequence_builder import feature_dict_to_vector
    fe = FeatureExtractor()
    ext = SequenceBuilder(window_size=2, window_minutes=5).extract_sequence(
        db, 1, fe, T - timedelta(minutes=60))
    # A sequência é recusada (sem telemetria, a sub-janela 0 também não tem as 11
    # features observadas); a sub-janela 1, com desvio None, nunca vira forward-fill.
    assert ext.windows is None and ext.reason.split(":")[0] in ("invalid_feature", "missing_observation")
    second = fe.extract_for_window(db, 1, T - timedelta(minutes=55), T - timedelta(minutes=50))
    assert feature_dict_to_vector(second) == (None, "invalid_feature:missing:rssi_std")


def _ctx_db():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    import models  # noqa: F401
    SQLModel.metadata.create_all(eng)
    s = Session(eng)
    from models import Network, Node
    # has_context exige os 5 campos (ai/propagation.py: environment_type, building_type,
    # obstacle_density, typical_link_distance_class, availability_criticality)
    s.add(Network(id=1, name="ctx", environment_type="indoor", building_type="residential",
                  obstacle_density="medium", typical_link_distance_class="short",
                  availability_criticality="medium"))
    s.add(Network(id=2, name="sem ctx"))
    s.add(Node(node_id=1, network_id=1, lora_address="0x0001"))
    s.add(Node(node_id=3, network_id=2, lora_address="0x0003"))
    s.add(Node(node_id=4, network_id=None, lora_address="0x0004"))
    s.commit()
    return s


def test_absolute_stays_absolute():
    from ai.feature_engineering import FeatureExtractor
    fe = FeatureExtractor(); fe.anomaly_classification_mode = "absolute"
    out = fe._compute_context_residuals(_ctx_db(), 1)
    assert out == {"_classification_mode": "absolute"}


def test_contextual_with_context_falls_back_for_rssi_derived_distance():
    from ai.feature_engineering import FeatureExtractor
    out = FeatureExtractor()._compute_context_residuals(_ctx_db(), 1)
    assert out["_classification_mode"] == "contextual_fallback"
    assert out["_context_reason"] == "distance_not_independent"
    assert "rssi_residual_mean" not in out and "pdr_residual_mean" not in out


def test_contextual_without_context_reason():
    from ai.feature_engineering import FeatureExtractor
    s = _ctx_db()
    assert FeatureExtractor()._compute_context_residuals(s, 3)["_context_reason"] == "network_without_context"
    assert FeatureExtractor()._compute_context_residuals(s, 4)["_context_reason"] == "node_without_network"
