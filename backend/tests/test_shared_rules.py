"""Fonte única das regras A1–A7 e snapshots históricos da evidência."""
from dataclasses import replace
from datetime import datetime, timedelta
import json
from types import MappingProxyType, SimpleNamespace

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

NEUTRAL = dict(rssi_mean=-80., rssi_std=1., snr_mean=8., snr_std=1., pdr_mean=10.,
               battery_delta=0., packet_loss_rate=0., rssi_snr_divergence=0.,
               beacon_interval_std=1., route_change_rate=0., link_asymmetry=1.)
NOW = datetime(2026, 10, 18, 12)


@pytest.fixture
def session():
    import models
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        for nid, net in [(1, 1), (2, 1), (99, 2)]:
            s.add(models.Node(node_id=nid, network_id=net, name=f"N{nid}", status="online"))
        s.commit()
        yield s
    engine.dispose()


def _change_a1(monkeypatch):
    from ai import anomaly_definitions as defs
    rule = defs.ANOMALY_DEFINITIONS["A1"]
    level = replace(rule.branches[0], comparisons=(replace(rule.branches[0].comparisons[0], threshold=-108.),))
    changed = replace(rule, branches=(level, *rule.branches[1:]))
    monkeypatch.setattr(defs, "ANOMALY_DEFINITIONS", MappingProxyType(dict(defs.ANOMALY_DEFINITIONS, A1=changed)))


def test_one_threshold_changes_evaluation_and_catalog(monkeypatch):
    from ai.anomaly_rules import evaluate_criteria
    from ai.anomaly_types import build_catalog
    assert evaluate_criteria(NEUTRAL | {"rssi_mean": -105.}).criteria["A1"].state == "hit"
    _change_a1(monkeypatch)
    assert evaluate_criteria(NEUTRAL | {"rssi_mean": -105.}).criteria["A1"].state == "miss"
    assert "RSSI médio < -108 dBm" in build_catalog()["A1"].criteria


def test_archived_rule_and_comparison_survive_definition_change(monkeypatch):
    from ai.anomaly_definitions import definition_fingerprint
    from ai.anomaly_rules import evaluate_criteria
    old = evaluate_criteria(NEUTRAL | {"rssi_mean": -105.})
    old_fingerprint = definition_fingerprint()
    _change_a1(monkeypatch)
    evidence = old.to_evidence()
    assert evidence["definitions_fingerprint"] == old_fingerprint != definition_fingerprint()
    assert evidence["definitions"]["A1"]["branches"]["level"][0]["threshold"] == -100.
    term = evidence["comparisons"]["A1"]["level"][0]
    assert term["value"] == -105. and term["state"] == "hit"


def test_units_and_input_order_remain_shared():
    from ai.anomaly_definitions import FEATURE_DEFINITIONS, FEATURE_ORDER
    from ai.feature_engineering import FeatureExtractor
    from ai.sequence_builder import FEATURE_NAMES
    from dataset_export.package import FEATURE_UNITS
    from dataset_export.provenance import source_hashes
    expected = ("rssi_mean", "rssi_std", "snr_mean", "snr_std", "pdr_mean", "battery_delta",
                "packet_loss_rate", "rssi_snr_divergence", "beacon_interval_std",
                "route_change_rate", "link_asymmetry")
    assert FEATURE_ORDER == expected
    assert list(expected) == FEATURE_NAMES == FeatureExtractor().feature_names
    assert FEATURE_UNITS["battery_delta"][0] == "pontos percentuais"
    assert FEATURE_UNITS["packet_loss_rate"][0] == "fração 0–1"
    assert all(FEATURE_UNITS[k] == (v.unit, v.description) for k, v in FEATURE_DEFINITIONS.items())
    assert "backend/ai/anomaly_definitions.py" in source_hashes()


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "não numérico"])
def test_unavailable_comparison_is_null_and_strict_json(value):
    from ai.anomaly_rules import evaluate_criteria
    result = evaluate_criteria(NEUTRAL | {"packet_loss_rate": value}).to_evidence()
    json.dumps(result, allow_nan=False)
    assert result["criteria"]["A2"]["state"] == "not_evaluable"
    term = result["comparisons"]["A2"]["loss"][0]
    assert term["value"] is None and "n/d" in term["text"]


def test_masked_default_never_becomes_observed_comparison():
    from ai.anomaly_rules import evaluate_criteria
    result = evaluate_criteria(NEUTRAL | {"packet_loss_rate": .9,
        "_unavailable": {"packet_loss_rate": "insufficient_distinct_counters"}}).to_evidence()
    term = result["comparisons"]["A2"]["loss"][0]
    assert term["value"] is None and term["state"] == "not_evaluable"
    assert result["unavailable_features"]["packet_loss_rate"] == "insufficient_distinct_counters"


def test_online_event_archives_point_window_and_keeps_lstm_context(session, monkeypatch):
    from ai.anomaly_detector import AnomalyDetector
    from config import settings
    monkeypatch.setattr(settings, "ML_WINDOW_MINUTES", 15)
    detector = AnomalyDetector(session.get_bind())
    features = NEUTRAL | {"rssi_mean": -105.}
    ml = SimpleNamespace(feature_extractor=SimpleNamespace(extract_features=lambda *a, **kw: features))
    pipeline = _published(**{"detect_detailed.return_value": _outcome([{"score_norm": 1., "evidence": {
        "context_start": "2026-10-18T11:00:00", "context_end": NOW.isoformat()}}])})
    monkeypatch.setattr(detector, "_get_pipeline_for_network", lambda *a: pipeline)
    saved = []
    monkeypatch.setattr(detector, "_save_and_notify", lambda session, **kw: saved.append(kw["anomaly"]))
    detector._run_online_pipeline(session, 1, ml.feature_extractor, [1], reference_time=NOW)
    evidence = saved[0]["evidence"]
    assert evidence["reference_time"] == NOW.isoformat()
    assert evidence["feature_window"] == {"start": "2026-10-18T11:45:00", "end": NOW.isoformat()}
    assert evidence["context_start"] == "2026-10-18T11:00:00"


def test_settings_api_exposes_generated_catalog(session, monkeypatch):
    import main
    from ai.anomaly_definitions import definition_fingerprint
    from ai.anomaly_rules import evaluate_criteria
    from ai.anomaly_types import get_catalog_info
    monkeypatch.setattr(main, "_get_available_algorithms", lambda: [])
    result = main.get_ml_settings(session)
    assert result["classification"]["catalog"] == get_catalog_info()
    assert result["classification"]["rules_version"] == "rules-v1"
    assert result["classification"]["definitions_version"] == "definitions-v3"
    fingerprint = result["classification"]["definitions_fingerprint"]
    assert fingerprint == evaluate_criteria(NEUTRAL).to_evidence()["definitions_fingerprint"]
    assert fingerprint == definition_fingerprint()


def test_evidence_exports_are_independent_deep_snapshots():
    from ai.anomaly_rules import evaluate_criteria
    result = evaluate_criteria(NEUTRAL | {
        "rssi_mean": -105., "_unavailable": {"packet_loss_rate": "masked"},
        "_battery_first_at": NOW-timedelta(minutes=15), "_battery_last_at": NOW,
    })
    original, exposed = result.to_evidence(), result.to_evidence()
    exposed["definitions"]["A1"]["branches"]["level"][0]["threshold"] = 0
    exposed["comparisons"]["A1"]["level"][0]["value"] = 0
    exposed["criteria"]["A2"]["reasons"].append("contamination")
    exposed["unavailable_features"]["packet_loss_rate"] = "contamination"
    exposed["battery_window"]["span_seconds"] = 0
    assert result.to_evidence() == original


def test_catalog_api_does_not_expose_mutable_feature_keys():
    from ai.anomaly_types import get_catalog_info
    original = get_catalog_info()
    exposed = get_catalog_info()
    exposed[0]["feature_keys"].append("contamination")
    assert get_catalog_info() == original


def test_fingerprint_and_model_order_ignore_metadata_dictionary_order(monkeypatch):
    from ai import anomaly_definitions as defs
    from ai.feature_engineering import FeatureExtractor
    from ai.sequence_builder import FEATURE_NAMES
    fingerprint = defs.definition_fingerprint()
    monkeypatch.setattr(defs, "FEATURE_DEFINITIONS",
                        MappingProxyType(dict(reversed(list(defs.FEATURE_DEFINITIONS.items())))))
    defs.validate_definitions()
    assert defs.definition_fingerprint() == fingerprint
    assert FeatureExtractor().feature_names == FEATURE_NAMES == list(defs.FEATURE_ORDER)


def test_fingerprint_covers_threshold_unit_and_vector_order(monkeypatch):
    from ai import anomaly_definitions as defs
    previous = defs.definition_fingerprint()
    _change_a1(monkeypatch)
    assert defs.definition_fingerprint() != previous
    previous = defs.definition_fingerprint()
    metadata = dict(defs.FEATURE_DEFINITIONS)
    metadata["rssi_mean"] = replace(metadata["rssi_mean"], unit="other unit")
    monkeypatch.setattr(defs, "FEATURE_DEFINITIONS", MappingProxyType(metadata))
    assert defs.definition_fingerprint() != previous
    previous = defs.definition_fingerprint()
    monkeypatch.setattr(defs, "FEATURE_ORDER", tuple(reversed(defs.FEATURE_ORDER)))
    assert defs.definition_fingerprint() != previous


@pytest.mark.parametrize("threshold", [None, True, "10", float("nan"), float("inf")])
def test_comparison_rejects_invalid_threshold_at_construction(threshold):
    from ai.anomaly_definitions import Comparison
    with pytest.raises(ValueError, match="número real finito"):
        Comparison("rssi_mean", "<", threshold)


@pytest.mark.parametrize("scale", [0., -1., True, "100", float("inf")])
def test_feature_rejects_invalid_display_scale(scale):
    from ai.anomaly_definitions import FEATURE_DEFINITIONS
    with pytest.raises(ValueError):
        replace(FEATURE_DEFINITIONS["packet_loss_rate"], display_scale=scale)


def test_definition_rejects_mutable_branches_duplicate_names_and_invalid_metadata():
    from ai.anomaly_definitions import ANOMALY_DEFINITIONS
    rule = ANOMALY_DEFINITIONS["A1"]
    invalid = [dict(branches=list(rule.branches)), dict(branches=(rule.branches[0],)*2),
               dict(branches=()), dict(category="typo"), dict(severity="typo"), dict(label=" ")]
    for changes in invalid:
        with pytest.raises(ValueError):
            replace(rule, **changes)
    with pytest.raises(ValueError):
        replace(rule.branches[0], comparisons=list(rule.branches[0].comparisons))
    with pytest.raises(ValueError):
        replace(rule.branches[0].comparisons[0], feature="unknown")
    with pytest.raises(ValueError):
        replace(rule.branches[0].comparisons[0], operator="<=")


@pytest.mark.parametrize("attribute,value", [
    ("FEATURE_ORDER", ("rssi_mean",)),
    ("CLASSIFICATION_ORDER", ("A1",)*7),
])
def test_registry_rejects_incomplete_or_duplicate_orders(monkeypatch, attribute, value):
    from ai import anomaly_definitions as defs
    monkeypatch.setattr(defs, attribute, value)
    with pytest.raises(ValueError):
        defs.validate_definitions()


def test_registry_rejects_key_code_mismatch_and_missing_fallback(monkeypatch):
    from ai import anomaly_definitions as defs
    invalid = dict(defs.ANOMALY_DEFINITIONS, A1=replace(defs.ANOMALY_DEFINITIONS["A1"], code="A8"))
    monkeypatch.setattr(defs, "ANOMALY_DEFINITIONS", MappingProxyType(invalid))
    with pytest.raises(ValueError, match="Chave do registro"):
        defs.validate_definitions()
    monkeypatch.setattr(defs, "ANOMALY_DEFINITIONS", MappingProxyType({k:v for k,v in invalid.items() if k != "AX"}))
    with pytest.raises(ValueError, match="AX ausente"):
        defs.validate_definitions()


def _outcome(anomalies=()):
    """O laço consome detect_detailed (componentes + combinação + emissões)."""
    from ai.online.combiner import ComboOutcome
    from ai.online.pipeline import PipelineOutcome
    return PipelineOutcome("isolation_forest", None, "single", None, None,
                           ComboOutcome("valid"), list(anomalies))


def _published(**over):
    """Dublê do pipeline publicado, com os atributos lidos pelo registro por ciclo."""
    from unittest.mock import MagicMock
    attrs = dict(fusion_alpha=0.5, tau=None, tau_status=None, bundle_id=None,
                 model_signature=None, _secondary_runner=None)
    attrs.update(over)
    return MagicMock(**attrs)
