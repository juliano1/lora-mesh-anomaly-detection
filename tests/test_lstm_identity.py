"""A persistência LSTM exige a identidade das features, usando dados sintéticos."""
import json

import numpy as np
import pytest

pytest.importorskip("torch")


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    import ai.lstm_autoencoder as mod
    monkeypatch.setattr(mod, "MODELS_DIR", tmp_path)
    model = mod.LSTMAutoencoderModel(window_size=2, hidden_size=4, epochs=1)
    data = np.random.default_rng(42).normal(size=(20, 2, mod.N_FEATURES)).astype(np.float32)
    assert model.train(data)
    model.save(version=7)
    return mod, tmp_path / "lstm_ae_v7.json"


def test_artifacts_without_identity_are_refused(artifact):
    mod, path = artifact
    meta = json.loads(path.read_text())
    meta.pop("feature_schema_version")
    meta.pop("feature_names")
    path.write_text(json.dumps(meta))
    with pytest.raises(mod.IncompatibleFeatureSchema):
        mod.LSTMAutoencoderModel.load(7)


def test_compatible_identity_loads_even_with_other_revision(artifact):
    mod, path = artifact
    meta = json.loads(path.read_text())
    meta["extractor_revision"] = "sha256:synthetic-previous-revision"
    path.write_text(json.dumps(meta))
    assert mod.LSTMAutoencoderModel.load(7).is_trained


def test_previous_version_prefix_preserves_model_compatibility(artifact):
    mod, path = artifact
    meta = json.loads(path.read_text())
    meta["feature_schema_version"] = "legacy065-" + meta["feature_schema_version"]
    path.write_text(json.dumps(meta))
    assert mod.LSTMAutoencoderModel.load(7).is_trained


@pytest.mark.parametrize("change", ["version", "order"])
def test_v1_or_reordered_names_refused(artifact, change):
    mod, path = artifact
    meta = json.loads(path.read_text())
    if change == "version":
        meta["feature_schema_version"] = "features-v1"
    else:
        meta["feature_names"].reverse()
    path.write_text(json.dumps(meta))
    with pytest.raises(mod.IncompatibleFeatureSchema):
        mod.LSTMAutoencoderModel.load(7)
