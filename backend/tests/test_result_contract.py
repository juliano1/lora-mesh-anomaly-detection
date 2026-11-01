"""Contrato de resultado por algoritmo."""
import math
from datetime import datetime

import numpy as np
import pytest


def test_valid_result_keeps_decision_and_scores():
    from ai.online.registry import AlgoResult
    r = AlgoResult(is_anomaly=True, score_norm=0.8, evidence={}, raw_score=-0.6, threshold=None)
    assert r.status == "valid" and r.is_valid
    assert r.is_anomaly is True and r.score_norm == 0.8 and r.raw_score == -0.6


@pytest.mark.parametrize("status", ["warming_up", "insufficient_history", "disabled", "error"])
def test_unavailable_has_null_decision_and_score(status):
    from ai.online.registry import AlgoResult
    r = AlgoResult.unavailable(status, "motivo")
    assert r.status == status and not r.is_valid
    assert r.is_anomaly is None and r.score_norm is None and r.reason == "motivo"


def test_unavailable_rejects_valid_status():
    from ai.online.registry import AlgoResult
    with pytest.raises(ValueError):
        AlgoResult.unavailable("valid", "x")


def test_unknown_status_rejected():
    from ai.online.registry import AlgoResult
    with pytest.raises(ValueError):
        AlgoResult(is_anomaly=None, score_norm=None, evidence={}, status="normal")


@pytest.mark.parametrize("score", [float("nan"), float("inf"), None])
def test_valid_with_non_finite_or_missing_score_becomes_error(score):
    from ai.online.registry import AlgoResult
    r = AlgoResult(is_anomaly=False, score_norm=score, evidence={})
    assert r.status == "error"
    assert r.is_anomaly is None and r.score_norm is None
    assert r.reason == "non_finite_or_missing_score"


def test_valid_with_non_finite_raw_score_becomes_error():
    from ai.online.registry import AlgoResult
    r = AlgoResult(is_anomaly=True, score_norm=0.5, evidence={}, raw_score=float("nan"))
    assert r.status == "error" and r.is_anomaly is None


def test_detection_context_accepts_cutoff_and_policy():
    from ai.online.registry import DetectionContext
    t = datetime(2026, 10, 7, 12, 0, 0)
    ctx = DetectionContext(session=None, node_id=1, feature_extractor=None,
                           features_vec=None, network_node_ids=[1], network_id=1,
                           reference_time=t, reference_policy="P", features_reason="no_data")
    assert ctx.reference_time == t and ctx.reference_policy == "P"
    assert ctx.features_vec is None and ctx.features_reason == "no_data"


def test_detection_context_defaults_are_backward_compatible():
    from ai.online.registry import DetectionContext
    ctx = DetectionContext(session=None, node_id=1, feature_extractor=None,
                           features_vec=np.zeros(11), network_node_ids=[1], network_id=1)
    assert ctx.reference_time is None and ctx.reference_policy is None
    assert ctx.features_reason is None


# ─── Limiar não finito e scores fora de 'valid' ───────────────────────────────

@pytest.mark.parametrize("thr", [float("nan"), float("inf")])
def test_valid_with_non_finite_threshold_becomes_error(thr):
    from ai.online.registry import AlgoResult
    r = AlgoResult(is_anomaly=False, score_norm=0.3, evidence={}, raw_score=0.04, threshold=thr)
    assert r.status == "error" and r.reason == "non_finite_or_missing_score"
    assert r.is_anomaly is None and r.score_norm is None
    assert r.raw_score is None and r.threshold is None
    assert r.evidence["invalid_values"]["threshold"] == str(thr)


def test_non_valid_result_nulls_raw_score_but_keeps_serializable_diagnostic():
    import json
    from ai.online.registry import AlgoResult
    r = AlgoResult(is_anomaly=True, score_norm=0.5, evidence={}, raw_score=float("nan"))
    assert r.status == "error" and r.raw_score is None
    assert r.evidence["invalid_values"] == {"raw_score": "nan"}
    json.dumps(r.evidence, allow_nan=False)


def test_valid_result_without_threshold_is_fine():
    from ai.online.registry import AlgoResult
    r = AlgoResult(is_anomaly=False, score_norm=0.2, evidence={}, raw_score=-0.4, threshold=None)
    assert r.is_valid and r.threshold is None
