"""
Adaptadores (runners) para uso no OnlinePipeline.

PointRunner:    embrulha o modelo pontual (Isolation Forest, que normaliza as
                features internamente).
SequenceRunner: embrulha LSTMAutoencoderModel (input_kind='sequence').

    - parâmetros vêm de EffectiveMLConfig, não de settings soltos;
    - treino usa corte explícito `ctx.reference_time` e ReferencePolicy (referência
      saudável, sem simulações nem linhas sintéticas);
    - candidato (dados → modelo → normalizador) só fica pronto se todas as etapas
      forem válidas; falha preserva o estado anterior e expõe `last_error`;
    - score_one devolve um estado explícito (valid, warming_up, insufficient_history,
      error) — nunca score zero/normal por indisponibilidade;
    - após falha de treino, nova tentativa só depois de ML_ONLINE_TRAIN_RETRY_SECONDS
      (por runner) — a reconstrução do dataset não se repete a cada nó/ciclo;
    - o SequenceRunner divide a referência em treino | validação | calibração: a
      validação só faz a parada antecipada; a calibração (posterior) define o limiar do
      LSTM e é guardada para τ; o PointRunner pontua os vetores alinhados a ela;
    - export_state/import_state permitem congelar e recarregar os modelos treinados.
"""
from __future__ import annotations

import logging
import math
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np

from ai.online.effective_config import EffectiveMLConfig, resolve_effective_config
from ai.online.normalization import ScoreNormalizer
from ai.online.registry import AlgoResult, AlgorithmSpec, DetectionContext

logger = logging.getLogger(__name__)


def _cutoff(ctx: DetectionContext) -> datetime:
    return ctx.reference_time if ctx.reference_time is not None else datetime.utcnow()


def _retry_seconds() -> float:
    from config import settings
    return float(getattr(settings, "ML_ONLINE_TRAIN_RETRY_SECONDS", 300))


class _TrainingLifecycle:
    """Tentativa com espera após falha e estado exportável."""

    _ready: bool
    _retry_after: Optional[float] = None
    trained_at: Optional[datetime] = None

    def ensure_trained(self, ctx: DetectionContext) -> bool:
        if self._ready:
            return True
        now = time.monotonic()
        if self._retry_after is not None and now < self._retry_after:
            return False                      # last_error da tentativa anterior permanece
        ok = self._train(ctx)
        self._retry_after = None if ok else now + _retry_seconds()
        if ok:
            self.trained_at = datetime.utcnow()
        return ok

    def allow_retry(self) -> None:
        """Libera nova tentativa imediata (retreino manual)."""
        self._retry_after = None

    def retry_in_seconds(self) -> Optional[float]:
        if self._ready or self._retry_after is None:
            return None
        return max(0.0, self._retry_after - time.monotonic())

    def export_state(self) -> dict:
        if not self._ready:
            raise ValueError("runner não treinado não pode ser exportado")
        return {"model": self._model, "normalizer": self._normalizer,
                "training_summary": dict(self.training_summary),
                "trained_at": self.trained_at}

    def import_state(self, state: dict) -> None:
        self._model = state["model"]
        self._normalizer = state["normalizer"]
        self.training_summary = dict(state.get("training_summary") or {})
        self.trained_at = state.get("trained_at")
        self.last_error = None
        self._retry_after = None
        self._ready = True


@dataclass
class HoldoutSet:
    """Bloco cronológico de calibração do LSTM (limiar p99 e τ da fusão).

    ``cutoff`` é o início do bloco: nenhum dos dois modelos treina depois dele.
    """
    X: np.ndarray
    meta: list
    cutoff: datetime


def _policy(ctx: DetectionContext, cfg: EffectiveMLConfig):
    if ctx.reference_policy is not None:
        return ctx.reference_policy
    from ai.reference_policy import ReferencePolicy
    return ReferencePolicy.from_session(ctx.session, cfg.reference_margin_minutes)


class PointRunner(_TrainingLifecycle):
    """
    Runner para modelos pontuais (BaseAnomalyModel incremental).

    Passa as features brutas ao modelo: o IF tem scaler interno (scaling='self').
    """

    input_kind = "point"

    def __init__(self, model: Any, spec: AlgorithmSpec,
                 config: Optional[EffectiveMLConfig] = None):
        self._model = model
        self._spec = spec
        self._config = config
        # Modelos pontuais: score bruto menor = mais anômalo → invert=True
        self._normalizer = ScoreNormalizer(invert=True)
        self._ready = False
        self.last_error: Optional[str] = None
        self.training_summary: dict = {}

    def is_ready(self) -> bool:
        return self._ready

    def _train(self, ctx: DetectionContext) -> bool:
        """Treina escopado à rede com referência saudável até o corte. True se pronto."""
        session = ctx.session
        feature_extractor = ctx.feature_extractor
        if session is None or feature_extractor is None or not ctx.network_node_ids:
            self.last_error = "missing_context"
            return False
        try:
            from models import Node
            cfg = resolve_effective_config(self._config)
            end_time = _cutoff(ctx)
            policy = _policy(ctx, cfg)
            exclusions: Counter = Counter()
            batch = []
            for node_id in ctx.network_node_ids:
                node = session.get(Node, node_id)
                if node is not None and node.sim_mode:
                    exclusions["node_in_sim_mode"] += 1
                    continue  # Nó em simulação agora (intervalo aberto)
                windows = feature_extractor.extract_sliding_windows(
                    session, node_id,
                    history_hours=cfg.if_history_hours,
                    window_minutes=cfg.if_window_minutes,
                    overlap_pct=cfg.if_overlap_pct,
                    end_time=end_time,
                    reference_policy=policy,
                    exclusion_stats=exclusions,
                )
                batch.extend((node_id, f) for f in (windows or []))
            from ai.feature_engineering import FEATURE_SCHEMA_VERSION
            from ai.sequence_builder import vectorize_windows
            X, batch, invalid = vectorize_windows(batch)
            exclusions.update(invalid)
            self.training_summary = {
                "n_windows": len(batch),
                "end_time": end_time.isoformat(),
                "history_hours": cfg.if_history_hours,
                "window_minutes": cfg.if_window_minutes,
                "step_minutes": cfg.if_step_minutes,
                "min_train_samples": cfg.if_min_train_samples,
                "exclusions": dict(exclusions),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
            }
            if len(batch) < cfg.if_min_train_samples:
                self.last_error = f"insufficient_reference: {len(batch)} < {cfg.if_min_train_samples}"
                return False
            if not self._model.train(X):
                err = getattr(self._model, "last_train_error", None)
                self.last_error = err if isinstance(err, str) else "train_failed"
                return False
            normalizer = ScoreNormalizer(invert=True)
            normalizer.fit(self._model.score(X))
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.error(f"PointRunner.ensure_trained() falhou: {exc}", exc_info=True)
            return False
        # Publicação conjunta do candidato
        self._normalizer = normalizer
        self._ready = True
        self.last_error = None
        return True

    def score_vectors(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(score_norm, is_anomaly, raw) de vetores pontuais na MESMA janela da inferência
        online (calibração alinhada de τ). X shape (M, n_features)."""
        X = np.asarray(X, dtype=float)
        if len(X) == 0:
            return np.array([]), np.array([], dtype=bool), np.array([])
        raw = np.asarray(self._model.score(X), dtype=float)
        pred = self._model.predict(X)
        return (np.array([self._normalizer.transform(float(r)) for r in raw]),
                np.asarray(pred) == -1, raw)

    def score_one(self, ctx: DetectionContext) -> AlgoResult:
        """Pontua o vetor pontual do nó no corte do ciclo."""
        if not self._ready:
            return AlgoResult.unavailable("warming_up", self.last_error or "not_trained")
        if ctx.features_vec is None:
            return AlgoResult.unavailable(
                "insufficient_history", ctx.features_reason or "no_point_features"
            )
        x = np.asarray(ctx.features_vec, dtype=float)[np.newaxis, :]  # (1, n_features)
        if not np.all(np.isfinite(x)):
            return AlgoResult.unavailable("error", "non_finite_features")
        try:
            pred = self._model.predict(x)[0]
            raw_score = float(self._model.score(x)[0])
            evidence = {"raw_score": round(raw_score, 6)}
            from ai.anomaly_ml import IsolationForestModel
            if isinstance(self._model, IsolationForestModel):
                evidence["margin"] = float(self._model.margin(x)[0])
                evidence["offset"] = self._model.offset
            return AlgoResult(
                is_anomaly=int(pred) == -1,
                score_norm=self._normalizer.transform(raw_score),
                evidence=evidence,
                raw_score=raw_score,
            )
        except Exception as exc:
            logger.error(f"PointRunner.score_one() falhou: {exc}", exc_info=True)
            return AlgoResult.unavailable("error", f"{type(exc).__name__}: {exc}")


class SequenceRunner(_TrainingLifecycle):
    """
    Runner para o LSTM Autoencoder (input_kind='sequence').

    Usa SequenceBuilder para construir a janela temporal do nó.
    """

    input_kind = "sequence"

    def __init__(self, model: Any, config: Optional[EffectiveMLConfig] = None):
        self._model = model
        self._config = config
        # AE: erro de reconstrução maior = mais anômalo → invert=False
        self._normalizer = ScoreNormalizer(invert=False)
        self._ready = False
        self.last_error: Optional[str] = None
        self.training_summary: dict = {}
        self.calibration_set: Optional[HoldoutSet] = None

    def is_ready(self) -> bool:
        return self._ready

    def _builder(self, cfg: EffectiveMLConfig):
        from ai.sequence_builder import SequenceBuilder
        return SequenceBuilder(
            window_size=getattr(self._model, "window_size", cfg.lstm_window_size),
            window_minutes=cfg.lstm_window_minutes,
            max_impute_gap=cfg.lstm_max_impute_gap,
        )

    def _train(self, ctx: DetectionContext) -> bool:
        """Treina com referência saudável até o corte, em três blocos cronológicos."""
        if ctx.session is None or ctx.feature_extractor is None or not ctx.network_node_ids:
            self.last_error = "missing_context"
            return False
        try:
            cfg = resolve_effective_config(self._config)
            sb = self._builder(cfg)
            ds = sb.build_reference_dataset(
                ctx.session, ctx.network_node_ids, ctx.feature_extractor,
                end_time=_cutoff(ctx),
                history_hours=cfg.lstm_history_hours,
                reference_policy=_policy(ctx, cfg),
            )
            summary = dict(ds.summary())
            if ds.X is None or len(ds.X) < 10:
                self.training_summary = summary
                self.last_error = f"insufficient_reference: {0 if ds.X is None else len(ds.X)} < 10"
                return False
            train_idx, val_idx, calib_idx, val_cut, calib_cut = sb.split_train_val_calib(
                ds.meta, val_fraction=0.2, calib_fraction=0.2)
            summary.update(n_train=len(train_idx), n_val=len(val_idx), n_calib=len(calib_idx),
                           cutoff=val_cut.isoformat() if val_cut else None,
                           calib_cutoff=calib_cut.isoformat() if calib_cut else None)
            self.training_summary = summary
            if len(train_idx) < 10 or len(val_idx) < 1 or len(calib_idx) < 1:
                self.last_error = (f"insufficient_split: train={len(train_idx)} "
                                   f"val={len(val_idx)} calib={len(calib_idx)}")
                return False
            X_train, X_val, X_calib = ds.X[train_idx], ds.X[val_idx], ds.X[calib_idx]
            if not self._model.train(X_train, X_val, X_calib):
                err = getattr(self._model, "last_train_error", None)
                self.last_error = err if isinstance(err, str) else "train_failed"
                return False
            summary["threshold_source"] = getattr(self._model, "threshold_source", None)
            normalizer = ScoreNormalizer(invert=False)
            normalizer.fit(self._model.score_batch(X_train))
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.error(f"SequenceRunner.ensure_trained() falhou: {exc}", exc_info=True)
            return False
        self._normalizer = normalizer
        self.calibration_set = HoldoutSet(X=X_calib, meta=[ds.meta[i] for i in calib_idx],
                                          cutoff=calib_cut)
        self._ready = True
        self.last_error = None
        return True

    def score_val(self, X_val_sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(score_norm, is_anomaly, raw) das sequências de calibração (τ da fusão)."""
        if len(X_val_sequences) == 0:
            return np.array([]), np.array([], dtype=bool), np.array([])
        raw = np.asarray(self._model.score_batch(X_val_sequences), dtype=float)
        return (np.array([self._normalizer.transform(float(r)) for r in raw]),
                raw > float(self._model.threshold), raw)

    def score_one(self, ctx: DetectionContext) -> AlgoResult:
        """Pontua a sequência que termina no corte do ciclo."""
        if not self._ready:
            return AlgoResult.unavailable("warming_up", self.last_error or "not_trained")
        try:
            cfg = resolve_effective_config(self._config)
            ext = self._builder(cfg).build_inference_extraction(
                ctx.session, ctx.node_id, ctx.feature_extractor,
                reference_time=_cutoff(ctx),
            )
            if not ext.is_valid:
                return AlgoResult.unavailable("insufficient_history", ext.reason)
            raw_score = float(self._model.score(ext.windows))
            threshold = float(self._model.threshold)
            if not (math.isfinite(raw_score) and math.isfinite(threshold)):
                # NaN > τ é False: comparar antes da guarda viraria "normal válido".
                return AlgoResult.unavailable(
                    "error", "non_finite_score_or_threshold",
                    evidence={"invalid_values": {"raw_score": str(raw_score),
                                                 "threshold": str(threshold)}},
                )
            return AlgoResult(
                is_anomaly=raw_score > threshold,
                score_norm=self._normalizer.transform(raw_score),
                evidence={
                    "raw_score": round(raw_score, 6),
                    "threshold": round(threshold, 6),
                    "imputed_windows": int(sum(ext.imputed_mask)),
                    "context_start": ext.start.isoformat(),
                    "context_end": ext.end.isoformat(),
                },
                raw_score=raw_score,
                threshold=threshold,
            )
        except Exception as exc:
            logger.error(f"SequenceRunner.score_one() falhou: {exc}", exc_info=True)
            return AlgoResult.unavailable("error", f"{type(exc).__name__}: {exc}")
