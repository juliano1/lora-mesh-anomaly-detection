"""Linhas de features e cobertura da exportação."""
import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel

from tests.conftest import EXPORT_T0, export_request


def _params(tracked=None, margin=0.0, req=None):
    from dataset_export.interpretation import InterpretationParams
    req = req or export_request()
    return InterpretationParams(source_id=req.source_id, data_origin_tracked_since=tracked,
                                reference_margin_minutes=margin, window_seconds=req.window_seconds,
                                step_seconds=req.step_seconds, grid_origin=req.grid_origin)


def _nodes(session, ids):
    from models import Node
    return {i: {"name": session.get(Node, i).name, "network_id": session.get(Node, i).network_id} for i in ids}


def test_one_row_per_node_and_instant_and_values_match_extractor(export_source):
    from ai.feature_engineering import FeatureExtractor
    from ai.sequence_builder import FEATURE_NAMES
    from dataset_export.features import iter_feature_rows
    req = export_request()
    eng = create_engine(f"sqlite:///{export_source}")
    with Session(eng) as s:
        rows = list(iter_feature_rows(s, req, _params(), _nodes(s, [1, 2, 4])))
        grid = req.grid()
        assert len(rows) == 3 * len(grid)
        assert [(r["node_id"], r["window_end_utc"]) for r in rows] == [(n, t) for n in (1, 2, 4) for t in grid]
        fe = FeatureExtractor()
        for r in rows:
            f = fe.extract_for_window(s, r["node_id"], r["window_start_utc"], r["window_end_utc"])
            if f is None:
                assert r["feature_status"] == "insufficient_data"
                assert all(r[k] is None for k in FEATURE_NAMES)
            else:
                assert r["feature_status"] == "valid"
                assert [r[k] for k in FEATURE_NAMES] == [float(f[k]) for k in FEATURE_NAMES]
            assert r["available_at_utc"] == r["window_end_utc"]
            assert r["window_end_utc"] - r["window_start_utc"] == timedelta(seconds=900)
    eng.dispose()


def test_node_without_data_is_insufficient_not_zero(export_source):
    from dataset_export.features import iter_feature_rows
    eng = create_engine(f"sqlite:///{export_source}")
    with Session(eng) as s:
        rows = [r for r in iter_feature_rows(s, export_request(node_ids=(4,)), _params(), _nodes(s, [4]))]
    assert rows and all(r["feature_status"] == "insufficient_data" and r["origin_status"] == "no_data"
                        and r["rssi_mean"] is None and r["reference_eligible_by_policy"] is False
                        and r["reference_exclusion_reason"] == "features_not_valid" for r in rows)
    eng.dispose()


def test_simulation_and_reverse_link_provenance(export_source):
    from dataset_export.features import iter_feature_rows
    eng = create_engine(f"sqlite:///{export_source}")
    with Session(eng) as s:
        rows = list(iter_feature_rows(s, export_request(), _params(tracked=datetime(2026, 1, 1)), _nodes(s, [1, 2])))
    by = {(r["node_id"], r["window_end_utc"]): r for r in rows}
    in_sim = [r for (n, t), r in by.items() if n == 1 and EXPORT_T0 + timedelta(minutes=45) <= t <= EXPORT_T0 + timedelta(minutes=60)]
    assert in_sim and all(11 in json.loads(r["simulation_run_ids"]) for r in in_sim)
    assert all(r["origin_status"] in ("synthetic", "mixed") and r["reference_eligible_by_policy"] is False for r in in_sim)
    node1 = [r for (n, _), r in by.items() if n == 1]
    assert all(12 in json.loads(r["simulation_run_ids"]) for r in node1)      # nó 3 (reverso) com simulação aberta
    assert all(json.loads(r["source_node_ids"]) == [1, 2, 3] for r in node1 if r["feature_status"] == "valid")
    late2 = [r for (n, t), r in by.items() if n == 2 and t > EXPORT_T0 + timedelta(minutes=80)]
    assert late2 and all(r["reference_eligible_by_policy"] is True and r["origin_status"] == "real" for r in late2)
    eng.dispose()


def test_origin_unknown_without_or_before_tracking(export_source):
    from dataset_export.features import iter_feature_rows
    eng = create_engine(f"sqlite:///{export_source}")
    with Session(eng) as s:
        untracked = list(iter_feature_rows(s, export_request(node_ids=(2,)), _params(tracked=None), _nodes(s, [2])))
        tracked = list(iter_feature_rows(s, export_request(node_ids=(2,)),
                                         _params(tracked=EXPORT_T0 + timedelta(minutes=60)), _nodes(s, [2])))
    assert {r["origin_status"] for r in untracked} <= {"unknown", "mixed"}
    early = [r for r in tracked if r["window_end_utc"] <= EXPORT_T0 + timedelta(minutes=30)]
    late = [r for r in tracked if r["window_start_utc"] >= EXPORT_T0 + timedelta(minutes=60)
            and r["window_end_utc"] > EXPORT_T0 + timedelta(minutes=80)]
    assert all(r["unknown_origin_record_count"] > 0 for r in early)
    assert all(r["unknown_origin_record_count"] == 0 and r["origin_status"] == "real" for r in late)
    eng.dispose()


@pytest.fixture
def tiny():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    import models  # noqa: F401
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        from models import Network, Node
        s.add(Network(id=1, name="r"))
        s.add(Node(node_id=1, network_id=1, name="n1", lora_address="0x0001"))
        s.commit()
        yield s


def _nb(ts, rssi=-80):
    from models import Neighbor
    return Neighbor(node_id=1, neighbor_address="0x0009", rssi=rssi, snr=5.0, link_quality=9,
                    timestamp=ts, data_origin="real")


def test_minimum_records_and_boundaries(tiny):
    """1 registro → insuficiente; fronteiras (início, fim]; futuro não altera."""
    from dataset_export.features import iter_feature_rows
    from dataset_export.request import ExportRequest, EPOCH
    t = EPOCH + timedelta(seconds=420 * 4257670)
    req = ExportRequest(source_id="x", network_id=1, node_ids=(1,), start=t - timedelta(seconds=1),
                        end=t, window_seconds=900, step_seconds=420)
    assert req.grid() == [t]
    nodes = {1: {"name": "n1", "network_id": 1}}
    tiny.add(_nb(t - timedelta(seconds=900), rssi=-10))     # exatamente no início: fora
    tiny.add(_nb(t - timedelta(minutes=5)))
    tiny.commit()
    (row,) = iter_feature_rows(tiny, req, _params(req=req), nodes)
    assert row["feature_status"] == "insufficient_data" and row["neighbor_record_count"] == 1
    tiny.add(_nb(t, rssi=-90))                              # exatamente no fim: dentro
    tiny.add(_nb(t + timedelta(seconds=1), rssi=+500))      # futuro extremo: fora
    tiny.commit()
    (row,) = iter_feature_rows(tiny, req, _params(req=req), nodes)
    # (início, fim]: o registro do fim entra, o do futuro não.
    assert row["neighbor_record_count"] == 2
    # Só vizinhos, sem telemetria: perda, bateria e intervalo não foram observados (v4).
    assert row["feature_status"] == "insufficient_data" and row["rssi_mean"] is None
    assert row["invalid_reason"].startswith("missing_observation:")
    assert json.loads(row["defaulted_features"]) == sorted(json.loads(row["defaulted_features"]))


def test_coverage_accumulator_counts():
    from dataset_export.features import CoverageAccumulator, FEATURE_STATUSES
    req = export_request(node_ids=(1, 4))
    acc = CoverageAccumulator(req, {1: {"name": "a", "network_id": 1}, 4: {"name": "b", "network_id": 1}})
    t = req.grid()[0]
    acc.add({"node_id": 1, "feature_status": "valid", "origin_status": "real",
             "reference_eligible_by_policy": True, "defaulted_features": '["pdr_mean"]', "simulation_overlap": False})
    acc.add({"node_id": 4, "feature_status": "insufficient_data", "origin_status": "no_data",
             "reference_eligible_by_policy": False, "defaulted_features": "[]", "simulation_overlap": True})
    rows = {r["node_id"]: r for r in acc.rows()}
    assert rows[1]["expected_windows"] == len(req.grid()) and rows[1]["written_windows"] == 1
    assert rows[1]["status_valid"] == 1 and rows[1]["windows_with_defaults"] == 1 and rows[1]["reference_eligible"] == 1
    assert rows[4]["status_insufficient_data"] == 1 and rows[4]["origin_no_data"] == 1
    assert rows[4]["simulation_overlap_windows"] == 1 and rows[4]["reference_excluded"] == 1
    assert sum(rows[1][f"status_{s}"] for s in FEATURE_STATUSES) == rows[1]["written_windows"]


def test_encode_value_generic():
    from dataset_export.codec import encode_value
    assert encode_value(None) is None and encode_value(True) == "1" and encode_value(3) == "3"
    assert encode_value(0.1) == "0.1" and encode_value(datetime(2026, 1, 1)) == "2026-01-01T00:00:00Z"
    with pytest.raises(ValueError):
        encode_value(float("inf"))
