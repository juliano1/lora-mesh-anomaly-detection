"""Parâmetros de interpretação do pacote."""
from datetime import datetime
from types import SimpleNamespace

import pytest

from tests.conftest import export_request


def _settings(tracked=None, margin=0.0, explicit=()):
    return SimpleNamespace(DATA_ORIGIN_TRACKED_SINCE=tracked, ML_REFERENCE_MARGIN_MINUTES=margin,
                           model_fields_set=set(explicit))


def test_for_export_and_roundtrip():
    from dataset_export.interpretation import InterpretationParams
    p = InterpretationParams.for_export(export_request(), _settings("2026-05-05T03:00:00Z", 15.0))
    assert p.data_origin_tracked_since == datetime(2026, 5, 5, 3, 0)
    assert p.reference_margin_minutes == 15.0 and p.window_seconds == 900
    assert InterpretationParams.from_dict(p.to_dict()) == p
    assert p.to_dict()["data_origin_tracked_since"] == "2026-05-05T03:00:00Z"


def test_manifest_values_win_when_local_not_explicit():
    from dataset_export.interpretation import InterpretationParams, resolve_for_computation
    p = InterpretationParams.for_export(export_request(), _settings(None, 15.0))
    eff, applied = resolve_for_computation(p, _settings(None, 0.0))
    assert eff == p and applied == {}


def test_explicit_local_divergence_is_an_error():
    from dataset_export.interpretation import (InterpretationConflict, InterpretationParams,
                                               resolve_for_computation)
    p = InterpretationParams.for_export(export_request(), _settings(None, 15.0))
    with pytest.raises(InterpretationConflict, match="reference_margin_minutes"):
        resolve_for_computation(p, _settings(None, 5.0, explicit={"ML_REFERENCE_MARGIN_MINUTES"}))
    with pytest.raises(InterpretationConflict, match="data_origin_tracked_since"):
        resolve_for_computation(p, _settings("2026-01-01T00:00:00Z", 15.0))


def test_explicit_override_is_applied_and_recorded():
    from dataset_export.interpretation import InterpretationParams, resolve_for_computation
    p = InterpretationParams.for_export(export_request(), _settings(None, 15.0))
    eff, applied = resolve_for_computation(
        p, _settings(None, 5.0, explicit={"ML_REFERENCE_MARGIN_MINUTES"}),
        overrides={"reference_margin_minutes": 5.0})
    assert eff.reference_margin_minutes == 5.0
    assert applied == {"reference_margin_minutes": {"manifest": 15.0, "override": 5.0}}
    with pytest.raises(ValueError):
        resolve_for_computation(p, _settings(), overrides={"source_id": "outro"})
