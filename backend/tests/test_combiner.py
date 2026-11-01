"""Semântica dos modos de combinação com componentes indisponíveis."""
import pytest
from ai.online.registry import AlgoResult
from ai.online.combiner import Combiner


def _v(is_anomaly, score):
    return AlgoResult(is_anomaly=is_anomaly, score_norm=score, evidence={})


def _u(status="insufficient_history", reason="missing_initial_context"):
    return AlgoResult.unavailable(status, reason)


def test_single_unavailable_primary_is_unavailable_not_normal():
    c = Combiner("single")
    out = c.evaluate(_u(), None)
    assert out.status == "unavailable" and out.is_anomaly is None
    assert out.reason == "primary_insufficient_history"
    assert c.combine(_u(), None, "isolation_forest", None) == []


def test_cascade_is_and_of_valid_decisions():
    c = Combiner("cascade")
    assert c.evaluate(_v(True, 0.9), _v(True, 0.7)).is_anomaly is True
    assert c.evaluate(_v(True, 0.9), _v(False, 0.1)).is_anomaly is False


def test_cascade_without_valid_secondary_is_unavailable_no_if_fallback():
    c = Combiner("cascade")
    out = c.evaluate(_v(True, 0.9), _u())
    assert out.status == "unavailable" and out.reason == "secondary_insufficient_history"
    assert c.combine(_v(True, 0.9), _u(), "isolation_forest", "lstm_ae") == []


def test_shadow_emits_valid_component_when_other_unavailable_both_orders():
    c = Combiner("shadow")
    out1 = c.combine(_v(True, 0.9), _u("error", "boom"), "isolation_forest", "lstm_ae")
    assert [r["detector_primary"] for r in out1] == ["isolation_forest"]
    out2 = c.combine(_u("warming_up", "not_trained"), _v(True, 0.8), "isolation_forest", "lstm_ae")
    assert [r["detector_primary"] for r in out2] == ["lstm_ae"]
    assert c.evaluate(_v(True, 0.9), _u()).status == "not_applicable"


def test_fusion_missing_secondary_does_not_become_zero():
    c = Combiner("fusion")
    out = c.evaluate(_v(True, 0.9), None, alpha=1.0, tau=0.5)
    assert out.status == "unavailable" and out.reason == "secondary_missing"


@pytest.mark.parametrize("tau", [None, float("nan"), float("inf")])
def test_fusion_requires_finite_tau(tau):
    c = Combiner("fusion")
    out = c.evaluate(_v(True, 0.9), _v(True, 0.9), alpha=0.5, tau=tau)
    assert out.status == "unavailable" and out.reason == "tau_unavailable"
    assert c.combine(_v(True, 0.9), _v(True, 0.9), "isolation_forest", "lstm_ae", 0.5, tau) == []


def test_fusion_invalid_alpha_is_unavailable():
    c = Combiner("fusion")
    out = c.evaluate(_v(True, 0.9), _v(True, 0.9), alpha=float("nan"), tau=0.5)
    assert out.status == "unavailable" and out.reason == "invalid_alpha"


def test_fusion_valid_decision_uses_strict_threshold():
    c = Combiner("fusion")
    out = c.evaluate(_v(True, 0.8), _v(True, 0.6), alpha=0.5, tau=0.5)
    assert out.status == "valid" and out.is_anomaly is True and abs(out.score - 0.7) < 1e-9
