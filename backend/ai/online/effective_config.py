"""
Configuração efetiva resolvida e serializável.

Fonte única para fábricas, builders, treino e inferência: defaults de config.py
+ MLSettings persistido (API). Valores sem suporte são rejeitados (ValueError),
nunca aceitos e ignorados. `model_signature` identifica o que invalida pesos,
scaler, sequência ou limiar de cada algoritmo; `pipeline_signature` acrescenta a
seleção/combinação.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional

SCORE_AGGREGATIONS = ("mean", "last", "max")
SUPPORTED_THRESHOLD_STRATEGIES = ("percentile",)
COMBOS = ("single", "cascade", "shadow", "fusion")
# Padrão experimental da próxima validação: LSTM-AE sozinho. O IF fica disponível para
# comparação; cascata e fusão são experimentais (sem vantagem demonstrada).
DEFAULT_ONLINE_PRIMARY = "lstm_ae"
EXPERIMENTAL_COMBOS = ("cascade", "fusion")


@dataclass(frozen=True)
class EffectiveMLConfig:
    primary: str
    secondary: Optional[str]
    combo: str
    fusion_alpha: float
    if_contamination: float
    if_n_estimators: int
    if_window_minutes: int
    if_overlap_pct: float
    if_step_minutes: int
    if_history_hours: float
    if_min_train_samples: int
    lstm_window_size: int
    lstm_window_minutes: int
    lstm_history_hours: float
    lstm_history_source: str
    lstm_hidden_size: int
    lstm_epochs: int
    lstm_patience: int
    lstm_score_aggregation: str
    lstm_threshold_percentile: float
    lstm_threshold_strategy: str
    lstm_max_impute_gap: int
    random_seed: int
    reference_margin_minutes: float

    def to_dict(self) -> dict:
        return asdict(self)

    def model_signature(self, algo_id: Optional[str]) -> tuple:
        if algo_id is None:
            return ()
        if algo_id == "lstm_ae":
            return (
                algo_id, self.lstm_window_size, self.lstm_window_minutes,
                self.lstm_history_hours, self.lstm_hidden_size, self.lstm_epochs,
                self.lstm_patience, self.lstm_score_aggregation,
                self.lstm_threshold_percentile, self.lstm_max_impute_gap,
                self.random_seed, self.reference_margin_minutes,
            )
        return (
            algo_id, self.if_contamination, self.if_n_estimators, self.if_window_minutes,
            self.if_overlap_pct, self.if_history_hours, self.if_min_train_samples,
            self.random_seed, self.reference_margin_minutes,
        )

    def models_signature(self, primary: str, secondary: Optional[str]) -> tuple:
        """Identidade dos modelos treinados, sem a regra de combinação."""
        return (primary, secondary, self.model_signature(primary), self.model_signature(secondary))

    def pipeline_signature(self, primary: str, secondary: Optional[str],
                           combo: str, alpha: float) -> tuple:
        return (primary, secondary, combo, float(alpha),
                self.model_signature(primary), self.model_signature(secondary))


def _pick(ml: Any, attr: str, default):
    value = getattr(ml, attr, None) if ml is not None else None
    return default if value is None else value


def resolve_effective_config(ml_settings: Any = None, app_settings: Any = None) -> EffectiveMLConfig:
    """Resolve defaults + banco/API em um objeto efetivo validado."""
    if isinstance(ml_settings, EffectiveMLConfig):
        return ml_settings
    if app_settings is None:
        from config import settings as app_settings
    ml = ml_settings

    stored_history = _pick(ml, "ml_lstm_history_hours", None)
    if stored_history is None:
        lstm_history = float(app_settings.ML_HISTORY_HOURS)
        history_source = "ML_HISTORY_HOURS"
    else:
        lstm_history = float(stored_history)
        history_source = "mlsettings.ml_lstm_history_hours"

    window_minutes = int(app_settings.ML_WINDOW_MINUTES)
    overlap = float(app_settings.ML_TRAIN_OVERLAP_PCT)
    cfg = EffectiveMLConfig(
        primary=str(_pick(ml, "ml_online_primary", DEFAULT_ONLINE_PRIMARY)),
        secondary=_pick(ml, "ml_online_secondary", None),
        combo=str(_pick(ml, "ml_online_combo", "single")),
        fusion_alpha=float(_pick(ml, "ml_online_fusion_alpha", 0.5)),
        if_contamination=float(_pick(ml, "ml_if_contamination", app_settings.ML_CONTAMINATION)),
        if_n_estimators=int(_pick(ml, "ml_if_n_estimators", app_settings.ML_N_ESTIMATORS)),
        if_window_minutes=window_minutes,
        if_overlap_pct=overlap,
        if_step_minutes=max(1, int(window_minutes * (1 - overlap))),
        if_history_hours=float(app_settings.ML_HISTORY_HOURS),
        if_min_train_samples=int(app_settings.ML_MIN_TRAIN_SAMPLES),
        lstm_window_size=int(_pick(ml, "ml_lstm_window", app_settings.ML_LSTM_WINDOW)),
        lstm_window_minutes=int(app_settings.ML_LSTM_WINDOW_MINUTES),
        lstm_history_hours=lstm_history,
        lstm_history_source=history_source,
        lstm_hidden_size=int(_pick(ml, "ml_lstm_hidden", app_settings.ML_LSTM_HIDDEN)),
        lstm_epochs=int(_pick(ml, "ml_lstm_epochs", app_settings.ML_LSTM_EPOCHS)),
        lstm_patience=int(_pick(ml, "ml_lstm_patience", app_settings.ML_LSTM_PATIENCE)),
        lstm_score_aggregation=str(_pick(ml, "ml_lstm_score_aggregation", "mean")),
        lstm_threshold_percentile=float(_pick(ml, "ml_lstm_threshold_percentile", 99.0)),
        lstm_threshold_strategy=str(_pick(ml, "ml_lstm_threshold_strategy", "percentile")),
        lstm_max_impute_gap=int(app_settings.ML_LSTM_MAX_IMPUTE_GAP),
        random_seed=int(app_settings.ML_RANDOM_SEED),
        reference_margin_minutes=float(app_settings.ML_REFERENCE_MARGIN_MINUTES),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: EffectiveMLConfig) -> None:
    errors = []
    if not 0.0 < cfg.if_contamination <= 0.5:
        errors.append(f"if contamination fora de (0, 0.5]: {cfg.if_contamination}")
    if cfg.if_n_estimators < 1:
        errors.append("if n_estimators < 1")
    if cfg.if_window_minutes < 1:
        errors.append("if window_minutes < 1")
    if not 0.0 <= cfg.if_overlap_pct < 1.0:
        errors.append("if overlap fora de [0, 1)")
    if cfg.if_history_hours <= 0:
        errors.append("if history_hours <= 0")
    if cfg.if_min_train_samples < 2:
        errors.append("if min_train_samples < 2")
    if cfg.lstm_window_size < 2:
        errors.append(f"lstm window_size < 2: {cfg.lstm_window_size}")
    if cfg.lstm_window_minutes < 1:
        errors.append("lstm window_minutes < 1")
    if not math.isfinite(cfg.lstm_history_hours) or cfg.lstm_history_hours <= 0:
        errors.append(f"lstm history_hours inválido: {cfg.lstm_history_hours}")
    if cfg.lstm_hidden_size < 1 or cfg.lstm_epochs < 1 or cfg.lstm_patience < 1:
        errors.append("lstm hidden_size/epochs/patience devem ser >= 1")
    if cfg.lstm_score_aggregation not in SCORE_AGGREGATIONS:
        errors.append(f"lstm score_aggregation não suportada: {cfg.lstm_score_aggregation!r}")
    if not 50.0 < cfg.lstm_threshold_percentile < 100.0:
        errors.append(f"lstm threshold_percentile fora de (50, 100): {cfg.lstm_threshold_percentile}")
    if cfg.lstm_threshold_strategy not in SUPPORTED_THRESHOLD_STRATEGIES:
        errors.append(
            f"estratégia de limiar {cfg.lstm_threshold_strategy!r} não implementada"
        )
    if cfg.lstm_max_impute_gap < 0:
        errors.append("lstm max_impute_gap < 0")
    if not 0.0 <= cfg.fusion_alpha <= 1.0:
        errors.append(f"fusion_alpha fora de [0, 1]: {cfg.fusion_alpha}")
    if cfg.combo not in COMBOS:
        errors.append(f"combo desconhecido: {cfg.combo!r}")
    if cfg.reference_margin_minutes < 0:
        errors.append("reference_margin_minutes < 0")
    if errors:
        raise ValueError("; ".join(errors))
