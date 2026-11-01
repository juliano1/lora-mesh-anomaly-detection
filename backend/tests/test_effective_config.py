"""Configuração efetiva resolvida."""
import pytest
from sqlalchemy import create_engine, text


def _ml(**over):
    base = dict(ml_if_contamination=0.03, ml_if_n_estimators=100, ml_lstm_window=12,
                ml_online_primary="isolation_forest", ml_online_secondary="lstm_ae",
                ml_online_combo="shadow", ml_online_fusion_alpha=0.5,
                ml_lstm_threshold_strategy="percentile")
    base.update(over)
    return type("ML", (), base)()


def test_defaults_preserve_current_effective_values():
    from ai.online.effective_config import resolve_effective_config
    cfg = resolve_effective_config(_ml())
    assert cfg.lstm_history_hours == 24.0            # efetivo hoje (ML_HISTORY_HOURS)
    assert cfg.lstm_history_source == "ML_HISTORY_HOURS"
    assert cfg.if_window_minutes == 15 and cfg.if_step_minutes == 7
    assert cfg.if_min_train_samples == 10
    assert cfg.lstm_window_size == 12 and cfg.lstm_window_minutes == 5
    assert (cfg.lstm_hidden_size, cfg.lstm_epochs, cfg.lstm_patience) == (32, 100, 5)
    assert cfg.lstm_max_impute_gap == 2 and cfg.random_seed == 42


def test_lstm_history_from_settings_is_explicit():
    from ai.online.effective_config import resolve_effective_config
    cfg = resolve_effective_config(_ml(ml_lstm_history_hours=168.0))
    assert cfg.lstm_history_hours == 168.0
    assert cfg.lstm_history_source == "mlsettings.ml_lstm_history_hours"


def test_window_change_changes_lstm_signature_only():
    from ai.online.effective_config import resolve_effective_config
    a, b = resolve_effective_config(_ml()), resolve_effective_config(_ml(ml_lstm_window=6))
    assert a.model_signature("lstm_ae") != b.model_signature("lstm_ae")
    assert a.model_signature("isolation_forest") == b.model_signature("isolation_forest")


def test_if_params_change_if_signature():
    from ai.online.effective_config import resolve_effective_config
    a = resolve_effective_config(_ml())
    b = resolve_effective_config(_ml(ml_if_contamination=0.10, ml_if_n_estimators=50))
    assert a.model_signature("isolation_forest") != b.model_signature("isolation_forest")
    assert a.pipeline_signature("isolation_forest", "lstm_ae", "shadow", 0.5) != \
        b.pipeline_signature("isolation_forest", "lstm_ae", "shadow", 0.5)


@pytest.mark.parametrize("over,msg", [
    (dict(ml_lstm_threshold_strategy="dynamic"), "estratégia"),
    (dict(ml_lstm_score_aggregation="median"), "score_aggregation"),
    (dict(ml_lstm_window=1), "window_size"),
    (dict(ml_if_contamination=0.9), "contamination"),
    (dict(ml_lstm_history_hours=0.0), "history_hours"),
])
def test_invalid_values_rejected(over, msg):
    from ai.online.effective_config import resolve_effective_config
    with pytest.raises(ValueError, match=msg):
        resolve_effective_config(_ml(**over))


def test_resolve_accepts_plain_objects_and_is_idempotent():
    from ai.online.effective_config import resolve_effective_config, EffectiveMLConfig
    cfg = resolve_effective_config(object())
    assert isinstance(cfg, EffectiveMLConfig)
    assert resolve_effective_config(cfg) is cfg
    assert cfg.to_dict()["primary"] == "lstm_ae"


def test_migration_adds_nullable_columns_idempotently():
    from migrations.add_lstm_config_fields import upgrade
    eng = create_engine("sqlite:///:memory:")
    with eng.connect() as c:
        c.execute(text("CREATE TABLE mlsettings (id INTEGER PRIMARY KEY, ml_lstm_window INTEGER)"))
        c.execute(text("INSERT INTO mlsettings (id, ml_lstm_window) VALUES (1, 12)"))
        c.commit()
    upgrade(eng)
    upgrade(eng)
    with eng.connect() as c:
        cols = {r[1] for r in c.execute(text("PRAGMA table_info(mlsettings)"))}
        row = c.execute(text("SELECT ml_lstm_history_hours, ml_lstm_window FROM mlsettings")).one()
    assert {"ml_lstm_history_hours", "ml_lstm_hidden", "ml_lstm_epochs", "ml_lstm_patience",
            "ml_lstm_score_aggregation", "ml_lstm_threshold_percentile"} <= cols
    assert row == (None, 12)            # histórico preservado, sem 168 silencioso
