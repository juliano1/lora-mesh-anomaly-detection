"""Corte explícito e proveniência na extração."""
from datetime import datetime, timedelta

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

T = datetime(2026, 9, 1, 12, 0, 0)


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


def _nb(node_id, addr, ts, rssi, origin="real"):
    from models import Neighbor
    return Neighbor(node_id=node_id, neighbor_address=addr, rssi=rssi, snr=8.0,
                    link_quality=9, timestamp=ts, data_origin=origin)


def _tel(node_id, ts, seq, bat, origin="real"):
    from models import Telemetry
    return Telemetry(node_id=node_id, timestamp=ts, rssi=-80, snr=8.0, sequence=seq,
                     uptime=seq, battery_level=bat, payload_raw="{}", data_origin=origin)


def test_extract_features_ignores_rows_after_reference_time(db):
    from ai.feature_engineering import FeatureExtractor
    for m in (10, 8):
        db.add(_nb(1, "0x0002", T - timedelta(minutes=m), -80))
    db.add(_tel(1, T - timedelta(minutes=10), 1, 90))
    db.add(_tel(1, T - timedelta(minutes=8), 2, 89))
    # registros FUTUROS ao corte
    for m in (2, 4):
        db.add(_nb(1, "0x0002", T + timedelta(minutes=m), -50))
    db.add(_tel(1, T + timedelta(minutes=3), 3, 10))
    db.commit()
    f = FeatureExtractor().extract_features(db, 1, window_minutes=15, reference_time=T)
    assert f["rssi_mean"] == -80.0
    assert f["battery_delta"] == -1.0


def test_extract_features_equals_extract_for_window(db):
    from ai.feature_engineering import FeatureExtractor
    for m in (12, 9, 6, 3):
        db.add(_nb(1, "0x0002", T - timedelta(minutes=m), -70 - m))
        db.add(_tel(1, T - timedelta(minutes=m), 20 - m, 80))
    db.commit()
    fe = FeatureExtractor()
    a = fe.extract_features(db, 1, window_minutes=15, reference_time=T)
    b = fe.extract_for_window(db, 1, T - timedelta(minutes=15), T)
    assert a == b


def test_window_provenance_counts_synthetic_and_reverse_sources(db):
    from ai.feature_engineering import FeatureExtractor
    s, e = T - timedelta(minutes=15), T
    db.add(_nb(1, "0x0002", T - timedelta(minutes=10), -80))
    db.add(_nb(1, "0x0002", T - timedelta(minutes=5), -81, origin="synthetic"))
    db.add(_nb(2, "0x0001", T - timedelta(minutes=7), -79))       # enlace reverso
    db.commit()
    prov = {}
    f = FeatureExtractor().extract_for_window(db, 1, s, e, provenance=prov)
    assert f is not None
    assert prov["synthetic_rows"] == 1
    assert prov["source_node_ids"] == {1, 2}
    assert "battery_delta" in prov["defaulted_features"]       # sem telemetria
    assert "link_asymmetry" not in prov["defaulted_features"]  # observado


def test_sliding_windows_respect_end_time_and_policy(db):
    from ai.feature_engineering import FeatureExtractor
    from ai.reference_policy import ReferencePolicy
    from collections import Counter
    for m in range(0, 120, 2):
        db.add(_nb(1, "0x0002", T - timedelta(minutes=m), -80))
    for m in (1, 3):                    # futuro
        db.add(_nb(1, "0x0002", T + timedelta(minutes=m), -50))
    db.commit()
    fe = FeatureExtractor()
    policy = ReferencePolicy({1: [(T - timedelta(minutes=60), T - timedelta(minutes=45))]})
    stats = Counter()
    wins = fe.extract_sliding_windows(db, 1, history_hours=2, window_minutes=15, overlap_pct=0.5,
                                      end_time=T, reference_policy=policy, exclusion_stats=stats)
    assert wins and all(w["rssi_mean"] == -80.0 for w in wins)
    assert stats["simulation_overlap"] > 0
