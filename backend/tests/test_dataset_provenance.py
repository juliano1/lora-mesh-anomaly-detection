"""Proveniência do extrator compartilhado."""
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


def _nb(node, addr, ts, origin="real", parent=False, rssi=-80):
    from models import Neighbor
    return Neighbor(node_id=node, neighbor_address=addr, rssi=rssi, snr=8.0, link_quality=9,
                    is_parent=parent, timestamp=ts, data_origin=origin)


def _tel(node, ts, seq, origin="real"):
    from models import Telemetry
    return Telemetry(node_id=node, timestamp=ts, rssi=-80, snr=8.0, sequence=seq, uptime=seq,
                     battery_level=90, payload_raw="{}", data_origin=origin)


def test_counts_origin_and_bounds(db):
    from ai.feature_engineering import FeatureExtractor
    db.add_all([_nb(1, "0x0002", T - timedelta(minutes=10)),
                _nb(1, "0x0003", T - timedelta(minutes=8)),
                _nb(1, "0x0002", T - timedelta(minutes=2), origin="synthetic"),
                _nb(2, "0x0001", T - timedelta(minutes=5)),
                _tel(1, T - timedelta(minutes=12), 1), _tel(1, T - timedelta(minutes=1), 2)])
    db.commit()
    prov = {}
    assert FeatureExtractor().extract_for_window(db, 1, T - timedelta(minutes=15), T, provenance=prov)
    assert prov["neighbor_record_count"] == 3 and prov["distinct_neighbor_count"] == 2
    assert prov["telemetry_record_count"] == 2 and prov["reverse_record_count"] == 1
    assert prov["origin_counts"] == {"real": 5, "synthetic": 1}
    assert prov["synthetic_rows"] == 1 and prov["source_node_ids"] == {1, 2}
    assert prov["first_observation"] == T - timedelta(minutes=12)
    assert prov["last_observation"] == T - timedelta(minutes=1)


def test_sparse_window_still_counts_everything(db):
    from ai.feature_engineering import FeatureExtractor
    db.add_all([_nb(1, "0x0002", T - timedelta(minutes=3)),
                _nb(2, "0x0001", T - timedelta(minutes=4), origin="synthetic"),
                _tel(1, T - timedelta(minutes=2), 1, origin="synthetic")])
    db.commit()
    prov = {}
    assert FeatureExtractor().extract_for_window(db, 1, T - timedelta(minutes=15), T, provenance=prov) is None
    assert prov["neighbor_record_count"] == 1 and prov["telemetry_record_count"] == 1
    assert prov["reverse_record_count"] == 1 and prov["synthetic_rows"] == 2
    assert prov["origin_counts"] == {"real": 1, "synthetic": 2}
    assert prov["defaulted_features"] == []


def test_empty_window(db):
    from ai.feature_engineering import FeatureExtractor
    prov = {}
    assert FeatureExtractor().extract_for_window(db, 1, T - timedelta(minutes=15), T, provenance=prov) is None
    assert prov["origin_counts"] == {} and prov["first_observation"] is None
    assert prov["neighbor_record_count"] == 0


def test_custom_origin_classifier(db):
    from ai.feature_engineering import FeatureExtractor
    db.add_all([_nb(1, "0x0002", T - timedelta(minutes=10)), _nb(1, "0x0002", T - timedelta(minutes=5))])
    db.commit()
    prov = {}
    FeatureExtractor().extract_for_window(
        db, 1, T - timedelta(minutes=15), T, provenance=prov,
        origin_classifier=lambda row: "unknown" if row.timestamp < T - timedelta(minutes=6) else "real")
    assert prov["origin_counts"] == {"unknown": 1, "real": 1}


def test_timestamp_ties_ordered_by_id(db):
    """route_change_rate depende da ordem; empates seguem o id (A, B, B → 1 troca)."""
    from ai.feature_engineering import FeatureExtractor
    ts = T - timedelta(minutes=5)
    db.add(_nb(1, "0x000a", ts, parent=True)); db.commit()
    db.add(_nb(1, "0x000b", ts, parent=True)); db.commit()
    db.add(_nb(1, "0x000b", T - timedelta(minutes=1), parent=True)); db.commit()
    f = FeatureExtractor().extract_for_window(db, 1, T - timedelta(minutes=15), T)
    assert f["route_change_rate"] == pytest.approx(1 / 3)
