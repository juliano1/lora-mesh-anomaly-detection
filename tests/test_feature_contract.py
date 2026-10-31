"""Contrato científico: as 11 features têm os mesmos nomes e a mesma ordem em todo o código."""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

EXPECTED = (
    "rssi_mean", "rssi_std", "snr_mean", "snr_std", "pdr_mean", "battery_delta",
    "packet_loss_rate", "rssi_snr_divergence", "beacon_interval_std",
    "route_change_rate", "link_asymmetry",
)


def _literal(path: Path, name: str):
    """Lê a lista/tupla atribuída a `name` sem importar o módulo (evita torch/pandas)."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"{name} não encontrado em {path}")


def test_canonical_order_is_frozen():
    from ai.anomaly_definitions import FEATURE_ORDER
    assert FEATURE_ORDER == EXPECTED


@pytest.mark.parametrize("relative,name", [
    ("scripts/train_if.py", "FEATURE_NAMES"),
    ("scripts/train_lstm.py", "FEATURE_NAMES"),
    ("scripts/extract_features.py", "FEATURE_NAMES"),
    ("importacao_arquivos_csv/codigos/detector_001_isolation_forest_sklearn.py", "FEATURES"),
])
def test_copies_match_canonical_order(relative, name):
    assert _literal(ROOT / relative, name) == EXPECTED


def test_runtime_consumers_use_canonical_order():
    from ai.feature_engineering import FeatureExtractor
    from ai.sequence_builder import FEATURE_NAMES
    assert tuple(FEATURE_NAMES) == EXPECTED
    assert tuple(FeatureExtractor().feature_names) == EXPECTED
