"""features.csv v2 com motivos de ausência."""
import json
from datetime import datetime, timedelta
from types import SimpleNamespace


def test_schema_version_comes_from_extractor():
    from ai.feature_engineering import FEATURE_SCHEMA_VERSION
    from dataset_export import features
    assert features.FEATURE_SCHEMA_VERSION == FEATURE_SCHEMA_VERSION == "features-v4"
    assert "unavailable_features" in features.FEATURE_COLUMNS


def test_missing_std_is_insufficient_data():
    from dataset_export.features import _status
    from ai.sequence_builder import FEATURE_NAMES
    f = {n: 1.0 for n in FEATURE_NAMES}; f["rssi_std"] = None
    status, reason, vector = _status(f)
    assert status == "insufficient_data" and reason == "invalid_feature:missing:rssi_std" and vector is None


def test_units_dictionary_describes_v2():
    from dataset_export.package import FEATURE_UNITS
    assert "por enlace" in FEATURE_UNITS["rssi_std"][1] and "ddof=0" in FEATURE_UNITS["rssi_std"][1]
    assert FEATURE_UNITS["pdr_mean"][0] == "índice 0–10"


def test_invalid_rows_keep_reasons(monkeypatch, export_source):
    from sqlalchemy import create_engine
    from sqlmodel import Session
    from tests.conftest import export_request
    from tests.test_dataset_features import _params, _nodes
    from dataset_export import features as mod
    from ai.feature_engineering import FeatureExtractor

    real = FeatureExtractor.extract_for_window
    def fake(self, *a, **kw):
        f = real(self, *a, **kw)
        if f is not None:
            f["rssi_std"] = f["snr_std"] = None
            f["_unavailable"] = dict(f["_unavailable"], rssi_std="no_eligible_link", snr_std="no_eligible_link")
        return f
    monkeypatch.setattr(FeatureExtractor, "extract_for_window", fake)
    eng = create_engine(f"sqlite:///{export_source}")
    with Session(eng) as s:
        rows = list(mod.iter_feature_rows(s, export_request(node_ids=(1,)), _params(), _nodes(s, [1])))
    eng.dispose()
    row = next(r for r in rows if r["invalid_reason"] == "invalid_feature:missing:rssi_std")
    assert row["feature_status"] == "insufficient_data"
    assert json.loads(row["unavailable_features"])["rssi_std"] == "no_eligible_link"
    assert row["defaulted_features"] != "[]" or json.loads(row["unavailable_features"])
