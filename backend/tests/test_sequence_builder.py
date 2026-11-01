"""Política de lacunas, referência e split global."""
from collections import Counter
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

from ai.sequence_builder import (
    FEATURE_NAMES, GAP_EXCEEDS_LIMIT, MISSING_INITIAL_CONTEXT, SequenceBuilder,
    feature_dict_to_vector,
)

T = datetime(2026, 9, 1, 12, 0, 0)


def _f(v=1.0):
    return {k: v for k in FEATURE_NAMES}


def _fe(seq):
    fe = MagicMock()
    fe.extract_for_window = MagicMock(side_effect=seq)
    return fe


@pytest.mark.parametrize("missing", [1, 2])
def test_initial_missing_windows_reject_sequence(missing):
    sb = SequenceBuilder(window_size=4, window_minutes=5, max_impute_gap=2)
    seq = [None] * missing + [_f(3.0)] * (4 - missing)
    ext = sb.build_inference_extraction(None, 1, _fe(seq), reference_time=T)
    assert not ext.is_valid and ext.reason == MISSING_INITIAL_CONTEXT
    assert ext.windows is None


def test_short_gap_forward_fill_keeps_mask():
    sb = SequenceBuilder(window_size=4, window_minutes=5, max_impute_gap=2)
    ext = sb.build_inference_extraction(None, 1, _fe([_f(5.0), None, None, _f(8.0)]), reference_time=T)
    assert ext.is_valid
    assert ext.imputed_mask == [False, True, True, False]
    np.testing.assert_allclose(ext.windows[1], ext.windows[0])
    assert ext.end == T and ext.start == T - timedelta(minutes=20)


def test_long_gap_rejected():
    sb = SequenceBuilder(window_size=5, window_minutes=5, max_impute_gap=1)
    ext = sb.build_inference_extraction(None, 1, _fe([_f(), None, None, _f(), _f()]), reference_time=T)
    assert ext.reason == GAP_EXCEEDS_LIMIT


def test_missing_or_non_finite_feature_is_invalid_not_zero():
    vec, reason = feature_dict_to_vector({k: 1.0 for k in FEATURE_NAMES[:-1]})
    assert vec is None and reason.startswith("invalid_feature:missing:")
    bad = _f(); bad["rssi_mean"] = float("nan")
    vec, reason = feature_dict_to_vector(bad)
    assert vec is None and reason == "invalid_feature:non_finite:rssi_mean"
    sb = SequenceBuilder(window_size=2, window_minutes=5)
    ext = sb.build_inference_extraction(None, 1, _fe([_f(), bad]), reference_time=T)
    assert not ext.is_valid and ext.reason.startswith("invalid_feature")


def test_inference_windows_are_anchored_on_reference_time():
    sb = SequenceBuilder(window_size=3, window_minutes=5)
    fe = _fe([_f(), _f(), _f()])
    sb.build_inference_extraction(None, 1, fe, reference_time=T)
    calls = [c.args[2:4] for c in fe.extract_for_window.call_args_list]
    assert calls == [
        (T - timedelta(minutes=15), T - timedelta(minutes=10)),
        (T - timedelta(minutes=10), T - timedelta(minutes=5)),
        (T - timedelta(minutes=5), T),
    ]


def test_reference_dataset_requires_explicit_end_time():
    sb = SequenceBuilder(window_size=2, window_minutes=5)
    with pytest.raises(ValueError):
        sb.build_reference_dataset(None, [1], _fe([]), end_time=None, history_hours=1,
                                   reference_policy=None)


def test_reference_dataset_excludes_contaminated_and_counts():
    from ai.reference_policy import ReferencePolicy
    sb = SequenceBuilder(window_size=2, window_minutes=5)
    fe = MagicMock()

    def _extract(session, node_id, s, e, provenance=None):
        if provenance is not None:
            provenance.update({"synthetic_rows": 0, "source_node_ids": {node_id}, "defaulted_features": []})
        return _f()
    fe.extract_for_window.side_effect = _extract
    policy = ReferencePolicy({1: [(T - timedelta(minutes=30), T - timedelta(minutes=20))]})
    ds = sb.build_reference_dataset(None, [1], fe, end_time=T, history_hours=1, reference_policy=policy)
    assert ds.exclusions["simulation_overlap"] > 0
    for m in ds.meta:
        assert m.end <= T - timedelta(minutes=30) or m.start >= T - timedelta(minutes=20)
    assert ds.X.shape[1:] == (2, len(FEATURE_NAMES)) and ds.X.dtype == np.float32
    assert ds.summary()["exclusions"]["simulation_overlap"] == ds.exclusions["simulation_overlap"]


def test_reference_dataset_rejects_synthetic_rows():
    from ai.reference_policy import ReferencePolicy
    sb = SequenceBuilder(window_size=2, window_minutes=5)
    fe = MagicMock()

    def _extract(session, node_id, s, e, provenance=None):
        provenance.update({"synthetic_rows": 1, "source_node_ids": {node_id}, "defaulted_features": []})
        return _f()
    fe.extract_for_window.side_effect = _extract
    ds = sb.build_reference_dataset(None, [1], fe, end_time=T, history_hours=1,
                                    reference_policy=ReferencePolicy.empty())
    assert ds.X is None and ds.exclusions["synthetic_rows"] > 0


def test_three_way_split_is_global_ordered_and_purged_across_nodes():
    from ai.sequence_builder import SequenceExtraction
    sb = SequenceBuilder(window_size=4, window_minutes=5)
    meta = []
    # nó 1 cobre 0–10h; nó 2 só 6–10h (coberturas diferentes)
    for node, h0 in ((1, 0), (2, 6)):
        start = T + timedelta(hours=h0)
        while start + timedelta(minutes=20) <= T + timedelta(hours=10):
            meta.append(SequenceExtraction(node_id=node, start=start, end=start + timedelta(minutes=20)))
            start += timedelta(minutes=5)
    tr, va, ca, val_cut, calib_cut = sb.split_train_val_calib(meta, 0.2, 0.2)
    assert tr and va and ca
    # Treino < validação (parada antecipada) < calibração (limiar e τ), sem sobreposição.
    assert max(meta[i].end for i in tr) <= val_cut <= min(meta[i].start for i in va)
    assert max(meta[i].end for i in va) <= calib_cut <= min(meta[i].start for i in ca)
    assert not (set(tr) & set(va) or set(va) & set(ca) or set(tr) & set(ca))
    # Purga: quem cruza um corte fica fora dos blocos.
    crossing = [i for i, m in enumerate(meta)
                if m.start < val_cut < m.end or m.start < calib_cut < m.end]
    assert crossing and not set(crossing) & (set(tr) | set(va) | set(ca))


# ─── Contexto sintético esparso não pode ser imputado ─────────────────────────

def _db_with_nodes():
    from sqlalchemy.pool import StaticPool
    from sqlmodel import Session, SQLModel, create_engine
    import models  # noqa: F401
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    s = Session(eng)
    from models import Node
    s.add(Node(node_id=1, network_id=1, lora_address="0x0001"))
    s.add(Node(node_id=2, network_id=1, lora_address="0x0002"))
    s.commit()
    return s


def _nb(node_id, addr, ts, origin="real", is_parent=False):
    from models import Neighbor
    return Neighbor(node_id=node_id, neighbor_address=addr, rssi=-80, snr=8.0,
                    link_quality=9, timestamp=ts, data_origin=origin, is_parent=is_parent)


def _tel(node_id, ts, origin="real", sequence=1):
    from models import Telemetry
    return Telemetry(node_id=node_id, timestamp=ts, rssi=-80, snr=8.0, sequence=sequence,
                     uptime=1, battery_level=90, payload_raw="{}", data_origin=origin)


def _three_windows(session, middle_rows):
    """Sub-janelas (T0,T0+5], (T0+5,T0+10], (T0+10,T0+15]: reais nas pontas.

    As pontas têm as 11 features observadas (parent, contadores, bateria e enlace
    reverso); feature não observada recusaria a sequência antes do teste de origem.
    """
    T0 = datetime(2026, 9, 1, 0, 0, 0)
    for k in (0, 2):
        for m in (1, 2, 3):
            ts = T0 + timedelta(minutes=5 * k + m)
            session.add(_nb(1, "0x0002", ts, is_parent=True))
            session.add(_nb(2, "0x0001", ts + timedelta(seconds=20)))
            session.add(_tel(1, ts, sequence=5 * k + m))
    for row in middle_rows(T0 + timedelta(minutes=5)):
        session.add(row)
    session.commit()
    return T0


def _reference(session, T0):
    from ai.feature_engineering import FeatureExtractor
    from ai.reference_policy import ReferencePolicy
    sb = SequenceBuilder(window_size=3, window_minutes=5, max_impute_gap=2)
    return sb.build_reference_dataset(session, [1], FeatureExtractor(),
                                      end_time=T0 + timedelta(minutes=15), history_hours=0.25,
                                      reference_policy=ReferencePolicy.from_session(session))


def test_sparse_synthetic_neighbor_rejects_reference_sequence():
    s = _db_with_nodes()
    T0 = _three_windows(s, lambda t: [_nb(1, "0x0002", t + timedelta(minutes=2), "synthetic")])
    ds = _reference(s, T0)
    assert ds.X is None and ds.exclusions["synthetic_rows"] == 1


def test_sparse_synthetic_telemetry_rejects_reference_sequence():
    s = _db_with_nodes()
    T0 = _three_windows(s, lambda t: [_tel(1, t + timedelta(minutes=2), "synthetic")])
    ds = _reference(s, T0)
    assert ds.X is None and ds.exclusions["synthetic_rows"] == 1


def test_sparse_synthetic_reverse_link_rejects_reference_sequence():
    s = _db_with_nodes()
    T0 = _three_windows(s, lambda t: [_nb(1, "0x0002", t + timedelta(minutes=1)),
                                      _nb(2, "0x0001", t + timedelta(minutes=2), "synthetic")])
    ds = _reference(s, T0)
    assert ds.X is None and ds.exclusions["synthetic_rows"] == 1


def test_clean_short_gap_still_allowed():
    s = _db_with_nodes()
    T0 = _three_windows(s, lambda t: [_nb(1, "0x0002", t + timedelta(minutes=2))])
    ds = _reference(s, T0)
    assert ds.X is not None and len(ds.X) == 1
    assert ds.meta[0].imputed_mask == [False, True, False]
