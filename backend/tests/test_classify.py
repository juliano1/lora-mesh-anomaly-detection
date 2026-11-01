"""4.5 — tipificação pós-ML, contrato AX e textos."""
from datetime import datetime

NEUTRAL = {
    "rssi_mean": -80.0, "rssi_std": 1.0, "snr_mean": 8.0, "snr_std": 1.0, "pdr_mean": 9.0,
    "battery_delta": 0.0, "packet_loss_rate": 0.0, "rssi_snr_divergence": 0.0,
    "beacon_interval_std": 0.0, "route_change_rate": 0.0, "link_asymmetry": 0.0,
}


def _cls(**over):
    from ai.anomaly_ml import classify_anomaly_type
    f = dict(NEUTRAL); f.update(over)
    return classify_anomaly_type(f, -0.3)


def test_low_index_alone_is_not_a2():
    out = _cls(pdr_mean=0.0)
    assert "A2" not in {a["anomaly_type"] for a in out}


def test_loss_with_good_index_is_a2():
    out = _cls(packet_loss_rate=0.4)
    assert [a["anomaly_type"] for a in out] == ["A2"]
    assert "não é taxa de entrega" in out[0]["description"]


def test_ax_keeps_not_evaluable_reasons():
    out = _cls(_unavailable={"packet_loss_rate": "insufficient_distinct_counters"})
    assert out[0]["anomaly_type"] == "AX"
    rules = out[0]["evidence"]["rules"]
    assert rules["criteria"]["A2"]["state"] == "not_evaluable"
    assert rules["rules_version"] == "rules-v1"


def test_a5_first_and_texts_without_causal_claims():
    out = _cls(rssi_snr_divergence=6.0, snr_mean=2.0, pdr_mean=2.0, link_asymmetry=20.0,
               route_change_rate=0.5, battery_delta=-25.0,
               _battery_first_at=datetime(2026, 9, 1, 11, 0), _battery_last_at=datetime(2026, 9, 1, 11, 10))
    codes = [a["anomaly_type"] for a in out]
    assert codes[0] == "A5"
    text = " ".join(a["description"] + " " + a["suggested_action"] for a in out)
    for forbidden in ("Jamming detectado", "Possível jamming", "falha de TX", "mV", "PDR=", "flapping"):
        assert forbidden not in text
    a3 = next(a for a in out if a["anomaly_type"] == "A3")
    assert "p.p." in a3["description"] and "10 min" in a3["description"]
    a6 = next(a for a in out if a["anomaly_type"] == "A6")
    assert "dB" in a6["description"] and "dBm" not in a6["description"]


def test_classification_mode_and_evidence_rules_on_hits():
    out = _cls(rssi_mean=-105.0, _classification_mode="contextual_fallback",
               _context_reason="distance_not_independent")
    a1 = out[0]
    assert a1["anomaly_type"] == "A1"
    assert a1["classification_mode_used"] == "contextual_fallback"
    assert a1["evidence"]["rules"]["context_reason"] == "distance_not_independent"


def test_online_ax_without_vector_keeps_reasons():
    import threading
    from unittest.mock import MagicMock
    from ai.anomaly_detector import AnomalyDetector
    d = object.__new__(AnomalyDetector)
    d._models_lock = threading.Lock(); d._online_combo = "single"; d._online_primary = "isolation_forest"
    d.stats = {"total_detected": 0, "by_type": {}, "by_severity": {"low": 0, "medium": 0, "high": 0},
               "last_detection": None}
    fe = MagicMock()
    fe.extract_features.return_value = dict(NEUTRAL, rssi_std=None, snr_std=None,
                                            _unavailable={"rssi_std": "no_eligible_link",
                                                          "snr_std": "no_eligible_link"})
    pipeline = _published()
    pipeline.detect_detailed.return_value = _outcome([{"score_norm": 0.9, "evidence": {}}])
    d._get_pipeline_for_network = MagicMock(return_value=pipeline)
    saved = []
    d._save_and_notify = lambda session, **kw: saved.append(kw["anomaly"])
    d._run_online_pipeline(MagicMock(), 7, fe, [1], None, reference_time=datetime(2026, 9, 1, 12))
    assert saved[0]["anomaly_type"] == "AX"
    ev = saved[0]["evidence"]
    assert ev["features_reason"] == "invalid_feature:missing:rssi_std"
    assert ev["unavailable_features"]["rssi_std"] == "no_eligible_link"


def test_catalog_vocabulary():
    from ai.anomaly_types import ANOMALY_CATALOG
    text = " ".join(f"{d.label} {d.description} {d.criteria}" for d in ANOMALY_CATALOG.values())
    for forbidden in ("mV", "Jamming detectado", "falha no transmissor", "dBm\"", "PDR < 2"):
        assert forbidden not in text
    assert "p.p." in ANOMALY_CATALOG["A3"].criteria
    assert ANOMALY_CATALOG["A6"].criteria.endswith("15 dB")


# ─── Disponibilidade das features chega à descrição ─

def _unavailable_quality_features():
    return dict(NEUTRAL, pdr_mean=5.0, packet_loss_rate=0.5,
                _unavailable={"pdr_mean": "no_link_quality",
                              "battery_delta": "insufficient_battery_readings"})


def test_description_shows_nd_for_unavailable_quality_and_keeps_vector():
    out = _cls(**{k: v for k, v in _unavailable_quality_features().items() if k in NEUTRAL or k == "_unavailable"})
    a2 = next(a for a in out if a["anomaly_type"] == "A2")
    assert "5.0/10" not in a2["description"]
    assert "índice de qualidade de sinal=n/d (no_link_quality)" in a2["description"]
    assert a2["evidence"]["features"]["pdr_mean"] == 5.0            # vetor numérico preservado


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
