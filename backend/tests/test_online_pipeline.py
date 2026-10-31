"""Testes do OnlinePipeline"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_ctx(is_anomaly=True):
    from ai.online.registry import DetectionContext
    return DetectionContext(
        session=MagicMock(),
        node_id=1,
        feature_extractor=MagicMock(),
        features_vec=np.zeros(10),
        network_node_ids=[1, 2, 3],
        network_id=1,
    )


def _stub_runner(is_anomaly, score_norm):
    from ai.online.registry import AlgoResult
    runner = MagicMock()
    runner.ensure_trained.return_value = True
    runner.score_one.return_value = AlgoResult(
        is_anomaly=is_anomaly, score_norm=score_norm, evidence={}
    )
    return runner


def test_pipeline_single_anomaly_returns_dict():
    from ai.online.pipeline import OnlinePipeline

    pipeline = OnlinePipeline(
        primary_id="isolation_forest",
        secondary_id=None,
        combo="single",
        fusion_alpha=0.5,
    )
    pipeline._primary_runner = _stub_runner(True, 0.8)
    pipeline._secondary_runner = None

    ctx = _make_ctx()
    results = pipeline.detect(ctx)
    assert len(results) == 1
    assert results[0]["detector_combo"] == "single"
    assert results[0]["detector_primary"] == "isolation_forest"


def test_pipeline_single_normal_returns_empty():
    from ai.online.pipeline import OnlinePipeline

    pipeline = OnlinePipeline("isolation_forest", None, "single", 0.5)
    pipeline._primary_runner = _stub_runner(False, 0.1)
    pipeline._secondary_runner = None

    results = pipeline.detect(_make_ctx())
    assert len(results) == 0


def test_pipeline_returns_empty_when_untrained():
    from ai.online.pipeline import OnlinePipeline

    pipeline = OnlinePipeline("isolation_forest", None, "single", 0.5)
    runner = MagicMock()
    runner.ensure_trained.return_value = False
    pipeline._primary_runner = runner
    pipeline._secondary_runner = None

    results = pipeline.detect(_make_ctx())
    assert len(results) == 0


def test_pipeline_cascade():
    from ai.online.pipeline import OnlinePipeline

    pipeline = OnlinePipeline("isolation_forest", "lstm_ae", "cascade", 0.5)
    pipeline._primary_runner = _stub_runner(True, 0.8)
    pipeline._secondary_runner = _stub_runner(True, 0.7)

    results = pipeline.detect(_make_ctx())
    assert len(results) == 1
    assert results[0]["detector_combo"] == "cascade"
    assert results[0]["detector_secondary"] == "lstm_ae"


def test_default_registry_has_expected_algorithms():
    """IF e LSTM-AE devem estar registrados; LOF e os algoritmos retirados, não."""
    import ai.online.pipeline  # importar o módulo força o registro padrão
    from ai.online.registry import available, get
    ids = {s.id for s in available()}
    assert "isolation_forest" in ids
    assert not {"ocsvm", "ecod", "hs_trees"} & ids
    pytest.importorskip("torch")  # lstm_ae condicional a torch (perfil sem PyTorch)
    assert "lstm_ae" in ids
    assert "lof" not in ids


def test_lof_cannot_be_registered():
    """LOF (novelty=False) levanta ValueError ao tentar registrar."""
    from ai.online.registry import register, AlgorithmSpec
    spec = AlgorithmSpec(
        id="lof_online_attempt",
        label="LOF",
        input_kind="point",
        scaling="self",
        incremental=False,  # LOF novelty=False é transdutivo
        factory=lambda _: None,
    )
    with pytest.raises(ValueError, match="transductive"):
        register(spec)
