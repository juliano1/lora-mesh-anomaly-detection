"""Exclusão histórica de simulações (todas, não só a última)."""
from datetime import datetime, timedelta

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

T0 = datetime(2026, 9, 1, 0, 0, 0)


@pytest.fixture
def db():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    import models  # noqa: F401
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        yield s


def _run(node_id, start_h, end_h):
    from models import SimulationRun
    return SimulationRun(
        node_id=node_id, anomaly="A1", duration_planned=60,
        started_at=T0 + timedelta(hours=start_h),
        ended_at=None if end_h is None else T0 + timedelta(hours=end_h),
    )


def test_all_closed_runs_are_excluded_not_only_last(db):
    from models import Node
    from ai.reference_policy import ReferencePolicy
    db.add(Node(node_id=1, network_id=1, sim_mode=False,
                sim_started_at=T0 + timedelta(hours=10), sim_ended_at=T0 + timedelta(hours=11)))
    db.add(_run(1, 2, 3))
    db.add(_run(1, 10, 11))
    db.commit()
    p = ReferencePolicy.from_session(db)
    assert p.node_contaminated([1], T0 + timedelta(hours=2, minutes=30), T0 + timedelta(hours=2, minutes=40))
    assert p.node_contaminated([1], T0 + timedelta(hours=10, minutes=5), T0 + timedelta(hours=10, minutes=10))
    assert not p.node_contaminated([1], T0 + timedelta(hours=5), T0 + timedelta(hours=6))


def test_open_run_extends_indefinitely(db):
    from ai.reference_policy import ReferencePolicy
    db.add(_run(2, 4, None))
    db.commit()
    p = ReferencePolicy.from_session(db)
    assert p.node_contaminated([2], T0 + timedelta(days=30), T0 + timedelta(days=30, minutes=5))


def test_interval_boundaries_are_half_open(db):
    from ai.reference_policy import ReferencePolicy
    p = ReferencePolicy({3: [(T0, T0 + timedelta(hours=1))]})
    assert not p.node_contaminated([3], T0 - timedelta(minutes=5), T0)
    assert not p.node_contaminated([3], T0 + timedelta(hours=1), T0 + timedelta(hours=1, minutes=5))


def test_margin_extends_interval():
    from ai.reference_policy import ReferencePolicy
    p = ReferencePolicy({3: [(T0, T0 + timedelta(hours=1))]}, margin_minutes=15)
    assert p.node_contaminated([3], T0 + timedelta(hours=1, minutes=5), T0 + timedelta(hours=1, minutes=10))


def test_legacy_node_interval_used_when_no_simulation_run(db):
    from models import Node
    from ai.reference_policy import ReferencePolicy
    db.add(Node(node_id=4, network_id=1, sim_mode=False,
                sim_started_at=T0, sim_ended_at=T0 + timedelta(hours=1)))
    db.commit()
    p = ReferencePolicy.from_session(db)
    assert p.node_contaminated([4], T0 + timedelta(minutes=10), T0 + timedelta(minutes=20))


def test_check_window_flags_synthetic_rows_and_reverse_link_nodes():
    from ai.reference_policy import ReferencePolicy, SYNTHETIC_ROWS, SIMULATION_OVERLAP
    p = ReferencePolicy({9: [(T0, T0 + timedelta(hours=1))]})
    s, e = T0 + timedelta(minutes=10), T0 + timedelta(minutes=15)
    assert p.check_window(1, s, e, {"synthetic_rows": 2, "source_node_ids": {1}}) == SYNTHETIC_ROWS
    assert p.check_window(1, s, e, {"synthetic_rows": 0, "source_node_ids": {1, 9}}) == SIMULATION_OVERLAP
    assert p.check_window(1, s, e, {"synthetic_rows": 0, "source_node_ids": {1}}) is None


def test_empty_policy_and_summary():
    from ai.reference_policy import ReferencePolicy
    p = ReferencePolicy.empty()
    assert not p.node_contaminated([1], T0, T0 + timedelta(minutes=5))
    assert p.to_dict() == {"source": "empty", "margin_minutes": 0.0, "n_intervals": 0}
def test_overlapping_runs_lists_ids_with_margin_and_legacy(db):
    from models import Node
    from ai.reference_policy import ReferencePolicy
    db.add(Node(node_id=5, network_id=1, sim_mode=False,
                sim_started_at=T0, sim_ended_at=T0 + timedelta(hours=1)))
    db.add(_run(5, 2, 3))
    db.add(_run(6, 2, None))
    db.commit()
    p = ReferencePolicy.from_session(db, margin_minutes=30)
    runs = p.overlapping_runs([5, 6], T0 + timedelta(hours=1, minutes=10), T0 + timedelta(hours=1, minutes=20))
    assert runs == ["legacy_node:5"]
    later = p.overlapping_runs([5, 6], T0 + timedelta(hours=3, minutes=10), T0 + timedelta(hours=3, minutes=15))
    assert len(later) == 2 and all(isinstance(r, int) for r in later)
    assert p.overlapping_runs([5], T0 + timedelta(days=5), T0 + timedelta(days=5, minutes=5)) == []


def test_explicit_intervals_without_ids_still_supported():
    from ai.reference_policy import ReferencePolicy
    p = ReferencePolicy({3: [(T0, T0 + timedelta(hours=1))]})
    assert p.overlapping_runs([3], T0, T0 + timedelta(minutes=5)) == [None]
    assert p.node_contaminated([3], T0, T0 + timedelta(minutes=5))

