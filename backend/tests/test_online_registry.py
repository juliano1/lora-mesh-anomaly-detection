"""Testes do registry de algoritmos online"""
import pytest


@pytest.fixture(autouse=True)
def clean_registry():
    """Limpa o registry antes e depois de cada teste."""
    from ai.online import registry as reg
    original = dict(reg._registry)
    reg._registry.clear()
    yield
    reg._registry.clear()
    reg._registry.update(original)


def _make_spec(id="algo_x", incremental=True):
    from ai.online.registry import AlgorithmSpec
    return AlgorithmSpec(
        id=id,
        label="Test",
        input_kind="point",
        scaling="self",
        incremental=incremental,
        factory=lambda _ml: None,
    )


def test_register_incremental_succeeds():
    from ai.online.registry import register, get
    spec = _make_spec("my_algo", incremental=True)
    register(spec)
    assert get("my_algo") is spec


def test_register_transductive_raises():
    from ai.online.registry import register
    spec = _make_spec("lof_transductive", incremental=False)
    with pytest.raises(ValueError, match="transductive"):
        register(spec)


def test_get_unknown_raises():
    from ai.online.registry import get
    with pytest.raises(KeyError):
        get("nonexistent")


def test_available_lists_registered():
    from ai.online.registry import register, available
    s1 = _make_spec("a1")
    s2 = _make_spec("a2")
    register(s1)
    register(s2)
    ids = {s.id for s in available()}
    assert {"a1", "a2"}.issubset(ids)


def test_algo_result_dataclass():
    from ai.online.registry import AlgoResult
    r = AlgoResult(is_anomaly=True, score_norm=0.9, evidence={"k": 1})
    assert r.is_anomaly is True
    assert r.score_norm == 0.9


def test_detection_context_dataclass():
    import numpy as np
    from ai.online.registry import DetectionContext
    ctx = DetectionContext(
        session=None,
        node_id=3,
        feature_extractor=None,
        features_vec=np.zeros(10),
        network_node_ids=[1, 2, 3],
        network_id=1,
    )
    assert ctx.node_id == 3
    assert ctx.features_vec.shape == (10,)
