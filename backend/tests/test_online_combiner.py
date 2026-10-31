"""Testes do combiner genérico"""
import pytest
from ai.online.registry import AlgoResult
from ai.online.combiner import Combiner


def _r(is_anomaly, score):
    return AlgoResult(is_anomaly=is_anomaly, score_norm=score, evidence={})


# ─── single ───

def test_single_anomaly_emits_one_result():
    c = Combiner("single")
    out = c.combine(_r(True, 0.9), None, "isolation_forest", None, alpha=0.5, tau=0.5)
    assert len(out) == 1
    assert out[0]["detector_combo"] == "single"
    assert out[0]["detector_primary"] == "isolation_forest"
    assert out[0]["detector_secondary"] is None


def test_single_normal_emits_nothing():
    c = Combiner("single")
    out = c.combine(_r(False, 0.1), None, "isolation_forest", None, alpha=0.5, tau=0.5)
    assert len(out) == 0


# ─── cascade ───

def test_cascade_both_anomaly_emits_one():
    c = Combiner("cascade")
    out = c.combine(_r(True, 0.8), _r(True, 0.7), "isolation_forest", "lstm_ae", alpha=0.5, tau=0.5)
    assert len(out) == 1
    assert out[0]["detector_combo"] == "cascade"
    assert out[0]["detector_primary"] == "isolation_forest"
    assert out[0]["detector_secondary"] == "lstm_ae"


def test_cascade_only_primary_anomaly_emits_nothing():
    c = Combiner("cascade")
    out = c.combine(_r(True, 0.8), _r(False, 0.2), "isolation_forest", "lstm_ae", alpha=0.5, tau=0.5)
    assert len(out) == 0


def test_cascade_only_secondary_anomaly_emits_nothing():
    c = Combiner("cascade")
    out = c.combine(_r(False, 0.2), _r(True, 0.9), "isolation_forest", "lstm_ae", alpha=0.5, tau=0.5)
    assert len(out) == 0


# ─── shadow ───

def test_shadow_both_anomalies_emits_two_with_secondary_none():
    c = Combiner("shadow")
    out = c.combine(_r(True, 0.8), _r(True, 0.7), "isolation_forest", "lstm_ae", alpha=0.5, tau=0.5)
    assert len(out) == 2
    primaries = {r["detector_primary"] for r in out}
    assert primaries == {"isolation_forest", "lstm_ae"}
    assert all(r["detector_secondary"] is None for r in out)
    assert all(r["detector_combo"] == "shadow" for r in out)


def test_shadow_only_primary_anomaly_emits_one():
    c = Combiner("shadow")
    out = c.combine(_r(True, 0.8), _r(False, 0.1), "isolation_forest", "lstm_ae", alpha=0.5, tau=0.5)
    assert len(out) == 1
    assert out[0]["detector_primary"] == "isolation_forest"


def test_shadow_neither_anomaly_emits_nothing():
    c = Combiner("shadow")
    out = c.combine(_r(False, 0.2), _r(False, 0.1), "isolation_forest", "lstm_ae", alpha=0.5, tau=0.5)
    assert len(out) == 0


# ─── fusion ───

def test_fusion_above_tau_emits_one():
    c = Combiner("fusion")
    # alpha=0.5, scores 0.8+0.6 → 0.7 > tau=0.5
    out = c.combine(_r(True, 0.8), _r(True, 0.6), "isolation_forest", "lstm_ae", alpha=0.5, tau=0.5)
    assert len(out) == 1
    assert out[0]["detector_combo"] == "fusion"
    assert out[0]["detector_primary"] == "isolation_forest"
    assert out[0]["detector_secondary"] == "lstm_ae"


def test_fusion_below_tau_emits_nothing():
    c = Combiner("fusion")
    # alpha=0.5, scores 0.2+0.1 → 0.15 < tau=0.5
    out = c.combine(_r(False, 0.2), _r(False, 0.1), "isolation_forest", "lstm_ae", alpha=0.5, tau=0.5)
    assert len(out) == 0


def test_fusion_respects_alpha():
    c = Combiner("fusion")
    # alpha=1.0 → só primário importa. primary_score=0.6 < tau=0.7 → sem anomalia
    out = c.combine(_r(True, 0.6), _r(True, 0.9), "isolation_forest", "lstm_ae", alpha=1.0, tau=0.7)
    assert len(out) == 0
    # alpha=0.0 → só secundário importa. secondary_score=0.9 > tau=0.7 → anomalia
    out = c.combine(_r(False, 0.1), _r(True, 0.9), "isolation_forest", "lstm_ae", alpha=0.0, tau=0.7)
    assert len(out) == 1


# ─── extensibilidade ───

def test_third_algorithm_combines_without_code_change():
    """Novo algoritmo no combiner não exige mudança em combiner.py."""
    c = Combiner("cascade")
    out = c.combine(_r(True, 0.8), _r(True, 0.7), "algoritmo_a", "algoritmo_b", alpha=0.5, tau=0.5)
    assert len(out) == 1
    assert out[0]["detector_primary"] == "algoritmo_a"
    assert out[0]["detector_secondary"] == "algoritmo_b"
