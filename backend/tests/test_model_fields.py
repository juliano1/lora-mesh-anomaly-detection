"""Verifica que MLSettings e Anomaly têm os campos da seleção online e da procedência do detector."""
import pytest

def test_mlsettings_online_fields():
    from models import MLSettings
    ml = MLSettings()
    # LSTM-AE é o padrão experimental; o IF fica disponível para comparação.
    from ai.online.effective_config import DEFAULT_ONLINE_PRIMARY
    assert ml.ml_online_primary == DEFAULT_ONLINE_PRIMARY == "lstm_ae"
    assert ml.ml_online_secondary is None
    assert ml.ml_online_combo == "single"
    assert ml.ml_online_fusion_alpha == 0.5

def test_anomaly_procedence_fields():
    from models import Anomaly
    a = Anomaly(
        node_id=1,
        anomaly_type="A1",
        category="radio",
        severity="medium",
        description="test",
        detection_method="ml",
        payload_raw="",
    )
    assert a.detector_combo is None
    assert a.detector_primary is None
    assert a.detector_secondary is None

def test_detector_algorithm_max_length_is_40():
    from models import Anomaly
    import sqlmodel
    field_info = Anomaly.model_fields["detector_algorithm"]
    assert any(
        getattr(m, "max_length", None) == 40
        for m in getattr(field_info, "metadata", [])
    ), "detector_algorithm.max_length deve ser 40"
