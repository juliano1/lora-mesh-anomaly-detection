"""Apresentação e evidência: arredondamento sem esconder a decisão, assinatura estável do
contrato e decisão de cada componente na evidência combinada."""
from datetime import datetime
import logging
from types import SimpleNamespace

import pytest
from sqlmodel import Session, SQLModel, create_engine

NEUTRAL = dict(rssi_mean=-80., rssi_std=1., snr_mean=8., snr_std=1., pdr_mean=10.,
               battery_delta=0., packet_loss_rate=0., rssi_snr_divergence=0.,
               beacon_interval_std=1., route_change_rate=0., link_asymmetry=1.)
NOW = datetime(2026, 10, 19, 12)
# Assinatura do contrato das definições v3 (A6 é "radio": descreve o enlace, sem nó de
# origem), com identificadores públicos descritivos. Mudanças só de apresentação
# não a alteram; o hash inclui os identificadores de versão do contrato.
FINGERPRINT_DEFINITIONS = "15c920e9a341c71347ae49f6042b421300c5106648d7200dc94a024572d698a0"  # pragma: allowlist secret


@pytest.fixture
def session():
    import models
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        for nid in (1, 2):
            s.add(models.Node(node_id=nid, network_id=1, name=f"N{nid}", status="online"))
        s.commit()
        yield s
    engine.dispose()


@pytest.mark.parametrize("feature,value,text", [
    ("packet_loss_rate", 0.29, "29%"),
    ("rssi_mean", -105.33333333333333, "-105.333 dBm"),
    ("pdr_mean", 1.2857142857142858, "1.286/10"),
    ("link_asymmetry", 15.0, "15 dB"),
    ("snr_mean", -0.0001, "0 dB"),
])
def test_presented_values_are_rounded(feature, value, text):
    from ai.anomaly_definitions import FEATURE_DEFINITIONS
    assert FEATURE_DEFINITIONS[feature].format_value(value) == text


def test_rounding_never_hides_the_decision_at_the_threshold():
    from ai.anomaly_rules import evaluate_criteria
    term = evaluate_criteria(NEUTRAL | {"packet_loss_rate": 0.3000004}) \
        .criteria["A2"].comparisons["loss"][0]
    assert term["state"] == "hit"
    assert not term["text"].startswith("packet_loss_rate: 30% > 30%")
    assert "30.00004" in term["text"]


def test_presentation_change_keeps_contract_fingerprint():
    from ai.anomaly_definitions import definition_fingerprint
    assert definition_fingerprint() == FINGERPRINT_DEFINITIONS


def test_fingerprint_is_cached_until_the_contract_object_changes(monkeypatch):
    from ai import anomaly_definitions as defs
    calls = []
    original = defs._compute_fingerprint
    monkeypatch.setattr(defs, "_compute_fingerprint", lambda: calls.append(1) or original())
    monkeypatch.setattr(defs, "_FINGERPRINT_CACHE", None)
    first = defs.definition_fingerprint()
    assert defs.definition_fingerprint() == first and len(calls) == 1
    monkeypatch.setattr(defs, "RULES_VERSION", "other-version")
    assert defs.definition_fingerprint() != first and len(calls) == 2


def _component(raw, *, anomaly=True, score=1.0, threshold=None, **evidence):
    from ai.online.registry import AlgoResult
    return AlgoResult(is_anomaly=anomaly, score_norm=score, raw_score=raw, threshold=threshold,
                      evidence={"raw_score": raw, **({"threshold": threshold} if threshold else {}),
                                **evidence})


LSTM_EVIDENCE = dict(context_start="2026-10-19T10:59:00", context_end="2026-10-19T11:59:00",
                     imputed_windows=0)


@pytest.mark.parametrize("combo", ["cascade", "fusion"])
def test_combined_evidence_keeps_each_component_decision(combo):
    from ai.online.combiner import Combiner
    if_result = _component(-0.61, score=0.9, margin=-0.1, offset=-0.5)
    lstm_result = _component(1.5, score=0.8, threshold=0.8, **LSTM_EVIDENCE)
    [event] = Combiner(combo).combine(if_result, lstm_result, "isolation_forest", "lstm_ae",
                                      alpha=0.5, tau=0.6)
    components = event["evidence"]["components"]
    assert components["primary"] == {"algorithm": "isolation_forest", "is_anomaly": True,
                                     "score_norm": 0.9, "raw_score": -0.61, "threshold": None}
    assert components["secondary"]["algorithm"] == "lstm_ae"
    assert components["secondary"]["threshold"] == 0.8
    assert event["evidence"]["secondary"]["context_end"] == "2026-10-19T11:59:00"

