"""
OnlinePipeline e registro padrão de algoritmos.

OnlinePipeline: instancia runners para (primary, secondary), garante treino lazy
e delega ao Combiner. Um pipeline por rede, reconstruído no hot-swap.

`train(ctx)` treina todos os componentes antes da publicação (o detector
treina fora do ciclo); τ da fusão é calibrado no bloco de calibração do componente
sequencial (posterior ao bloco da parada antecipada), com o pontual pontuando a MESMA
janela da inferência; trocar shadow/fusão/cascata (`set_combination`) reaproveita os
modelos; `export_bundle`/`from_bundle` congelam.

Registro: o Isolation Forest (pontual) está sempre registrado. O LSTM-AE (sequencial,
primário padrão) só é registrado quando ML_LSTM_ENABLED e o torch está disponível; sem
ele, um pipeline que pede o LSTM cai para o IF e guarda `fallback_reason`.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Optional

import numpy as np

from config import settings
from ai.online.combiner import COMBOS, Combiner, ComboOutcome
from ai.online.effective_config import EffectiveMLConfig, resolve_effective_config
from ai.online.registry import (
    AlgoResult, AlgorithmSpec, DetectionContext, register, get as registry_get,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineOutcome:
    """Resultado completo de um nó/instante: componentes, combinação e anomalias emitidas."""
    primary_id: str
    secondary_id: Optional[str]
    combo: str
    primary: AlgoResult
    secondary: Optional[AlgoResult]
    combination: ComboOutcome
    anomalies: list = field(default_factory=list)


def evaluate_runner(runner, ctx: DetectionContext) -> AlgoResult:
    """Avalia um runner isoladamente: falha de um componente não
    bloqueia o outro e nunca vira normalidade."""
    if runner is None:
        return AlgoResult.unavailable("disabled", "runner_absent")
    try:
        if not runner.ensure_trained(ctx):
            reason = getattr(runner, "last_error", None)
            return AlgoResult.unavailable(
                "warming_up", reason if isinstance(reason, str) else "not_trained"
            )
        result = runner.score_one(ctx)
    except Exception as exc:
        logger.error(f"OnlinePipeline: runner falhou: {exc}", exc_info=True)
        return AlgoResult.unavailable("error", f"{type(exc).__name__}: {exc}")
    if not isinstance(result, AlgoResult):
        return AlgoResult.unavailable("error", "runner_returned_non_result")
    return result


class OnlinePipeline:
    """
    Pipeline de detecção online para uma rede.

    Instancia runners via registro, garante treino/carregamento lazy
    e expõe detect(ctx) -> list[dict] chamando o combinador.
    """

    def __init__(
        self,
        primary_id: str,
        secondary_id: Optional[str],
        combo: str,
        fusion_alpha: float,
        tau: float = 0.5,
    ):
        # Seleção solicitada (antes de normalizações/fallbacks), base da assinatura
        self._requested = (primary_id, secondary_id, combo, float(fusion_alpha))

        self._primary_id = primary_id
        self._secondary_id = secondary_id
        self._combo = combo
        self._fusion_alpha = fusion_alpha
        # fusion: τ é calibrado no bloco de calibração (calibrate()); demais modos não usam τ.
        self._tau = None if combo == "fusion" else tau
        # Valores provisórios até build_runners(), que os substitui por
        # cfg.if_contamination e cfg.lstm_window_size.
        self._contamination = 0.03
        self._window = 12
        self._tau_calibrated = False
        self._calib_lock = threading.Lock()
        # Scores alinhados do holdout; τ é recalculado deles para qualquer α.
        self.calibration: Optional[dict] = None
        self.tau_status: str = "not_calibrated"
        self.tau_stats: dict = {}
        self.model_signature: Optional[tuple] = None
        self.bundle_id: Optional[str] = None

        self._primary_runner = None
        self._secondary_runner = None
        self._ready = False
        self._lock = threading.Lock()
        self._combiner = Combiner(combo)
        self._config: Optional[EffectiveMLConfig] = None
        self.signature: Optional[tuple] = None
        self.fallback_reason: Optional[str] = None

    def is_ready(self) -> bool:
        return getattr(self, "_ready", False)

    def build_runners(self, ml_settings) -> None:
        """Instancia runners de forma ATÔMICA: só publica se todos subirem.

        Em falha de qualquer runner, não publica meio-pipeline e marca não-pronto
        (is_ready() == False). O chamador decide preservar o anterior / não publicar.
        """
        with self._lock:
            self._ready = False
            try:
                cfg = resolve_effective_config(ml_settings)
            except ValueError as exc:
                logger.error(f"OnlinePipeline: configuração inválida — {exc}; não publicando.")
                return
            try:
                primary = registry_get(self._primary_id).factory(cfg)
            except KeyError:
                # Primário persistido não está disponível (ex.: lstm_ae num
                # ambiente sem torch). Cai para isolation_forest para não parar a detecção.
                logger.warning(
                    f"OnlinePipeline: primário {self._primary_id!r} indisponível — "
                    "fallback para 'isolation_forest'."
                )
                self.fallback_reason = f"primary_unavailable:{self._requested[0]}"
                self._primary_id = "isolation_forest"
                self._secondary_id = None
                self._combo = "single"
                self._combiner = Combiner("single")
                try:
                    primary = registry_get("isolation_forest").factory(cfg)
                except Exception as exc:
                    logger.error(f"OnlinePipeline: fallback isolation_forest falhou: {exc}")
                    return
            except Exception as exc:
                logger.error(f"OnlinePipeline: primário {self._primary_id!r} falhou: {exc}")
                return
            secondary = None
            if self._secondary_id:
                try:
                    secondary = registry_get(self._secondary_id).factory(cfg)
                except KeyError:
                    # Secundário persistido não está disponível (ex.: lstm_ae num
                    # ambiente sem torch). Degrada para single em vez de derrubar a
                    # detecção — o primário já subiu e continua publicando.
                    logger.warning(
                        f"OnlinePipeline: secundário {self._secondary_id!r} indisponível — "
                        "degradando para 'single'."
                    )
                    self.fallback_reason = f"secondary_unavailable:{self._requested[1]}"
                    self._secondary_id = None
                    self._combo = "single"
                    self._combiner = Combiner("single")
                    secondary = None
                except Exception as exc:
                    logger.error(
                        f"OnlinePipeline: secundário {self._secondary_id!r} falhou: {exc}. "
                        "Não publicando meio-pipeline."
                    )
                    return
            self._primary_runner = primary
            self._secondary_runner = secondary
            self._config = cfg
            self._contamination = cfg.if_contamination
            self._window = cfg.lstm_window_size
            self.signature = cfg.pipeline_signature(*self._requested)
            self.model_signature = cfg.models_signature(*self._requested[:2])
            self._ready = True

    def detect(self, ctx: DetectionContext) -> list[dict]:
        """Anomalias emitidas para um nó (vazia = nenhuma ou indisponível)."""
        return self.detect_detailed(ctx).anomalies

    def detect_detailed(self, ctx: DetectionContext) -> PipelineOutcome:
        """Avalia cada componente de forma independente e combina."""
        # Snapshot consistente sob o lock (o fallback de build_runners muta ids/combo).
        with self._lock:
            primary_runner = self._primary_runner
            secondary_runner = self._secondary_runner
            combo = self._combo
            primary_id = self._primary_id
            secondary_id = self._secondary_id
            combiner = self._combiner

        if primary_runner is None:
            primary_result = AlgoResult.unavailable("disabled", "pipeline_not_built")
        else:
            primary_result = evaluate_runner(primary_runner, ctx)
        secondary_result = None
        if combo != "single":
            secondary_result = evaluate_runner(secondary_runner, ctx)

        # Calibração tardia só para pipelines treinados de forma lazy
        # (o detector publica pipelines já calibrados por train()).
        if (combo == "fusion" and not self._tau_calibrated and primary_result.is_valid
                and secondary_result is not None and secondary_result.is_valid):
            with self._calib_lock:
                if not self._tau_calibrated:
                    self.calibrate(ctx)

        ranks = self.fusion_ranks(primary_result, secondary_result) if combo == "fusion" else None
        combination = combiner.evaluate(
            primary_result, secondary_result, alpha=self._fusion_alpha, tau=self._tau,
            fusion_scores=ranks,
        )
        anomalies = combiner.combine(
            primary=primary_result,
            secondary=secondary_result,
            primary_id=primary_id,
            secondary_id=secondary_id,
            alpha=self._fusion_alpha,
            tau=self._tau,
            fusion_scores=ranks,
        )
        return PipelineOutcome(
            primary_id=primary_id, secondary_id=secondary_id, combo=combo,
            primary=primary_result, secondary=secondary_result,
            combination=combination, anomalies=anomalies,
        )

    # ─── Treino explícito, calibração alinhada, combinação e congelamento ───

    @property
    def combo(self) -> str:
        return self._combo

    @property
    def fusion_alpha(self) -> float:
        return self._fusion_alpha

    @property
    def tau(self) -> Optional[float]:
        return self._tau

    def _roles(self):
        """(papel/runner sequencial, papel/runner pontual) ou None se o par não tem os dois."""
        pairs = [("primary", self._primary_runner), ("secondary", self._secondary_runner)]
        seq = [(r, x) for r, x in pairs if getattr(x, "input_kind", None) == "sequence"]
        pt = [(r, x) for r, x in pairs if getattr(x, "input_kind", None) == "point"]
        return (seq[0], pt[0]) if len(seq) == 1 and len(pt) == 1 else None

    def train(self, ctx: DetectionContext) -> bool:
        """Treina todos os componentes; True só com o pipeline completo pronto.

        Com um componente sequencial e um pontual, o sequencial treina primeiro e o
        pontual treina com referência até o início do bloco de calibração do LSTM —
        a calibração de τ fica fora do treino dos DOIS modelos. τ é calibrado em
        seguida para qualquer modo, deixando a fusão pronta para uma troca futura.
        """
        roles = self._roles()
        runners = [r for r in (self._primary_runner, self._secondary_runner) if r is not None]
        if not runners:
            return False
        if roles is None:
            ok = all(r.ensure_trained(ctx) for r in runners)
            if ok and self._secondary_runner is not None:
                self.calibrate(ctx)
            return ok
        (_, seq), (_, pt) = roles
        if not seq.ensure_trained(ctx):
            return False
        holdout = getattr(seq, "calibration_set", None)
        point_ctx = ctx
        if holdout is not None and holdout.cutoff is not None:
            point_ctx = replace(ctx, reference_time=holdout.cutoff)
        if not pt.ensure_trained(point_ctx):
            return False
        self.calibrate(ctx)
        return True

    @staticmethod
    def calibrate_tau(s_final, contamination: float):
        """τ = percentil (1-contamination) do score fundido na calibração. None se vazio."""
        s = np.asarray(s_final, dtype=float)
        if s.size == 0:
            return None
        return float(np.percentile(s, 100.0 * (1.0 - contamination)))

    def calibrate(self, ctx: DetectionContext) -> None:
        """Scores alinhados no bloco de calibração do componente sequencial.

        O bloco é posterior à validação que escolheu a época do LSTM e anterior a
        qualquer teste. Cada sequência de calibração termina em ``m.end``; o componente pontual pontua
        o vetor de features da janela online (if_window_minutes) que termina nesse
        mesmo instante, isto é, o que a inferência ao vivo veria naquele corte.
        Falhas deixam τ indisponível com motivo explícito — nunca τ silencioso.
        """
        MIN_CAL = 20
        self._tau_calibrated = True
        self.calibration = None
        try:
            roles = self._roles()
            if roles is None:
                return self._set_tau(None, "fusion_requires_sequence_and_point_components")
            (seq_role, seq), (pt_role, pt) = roles
            holdout = getattr(seq, "calibration_set", None)
            if holdout is None or len(holdout.meta) == 0:
                return self._set_tau(None, "no_calibration_set")
            from ai.sequence_builder import feature_dict_to_vector
            cfg = self._config or resolve_effective_config(None)
            s_seq, d_seq, r_seq = seq.score_val(holdout.X)
            vectors, keep = [], []
            for i, m in enumerate(holdout.meta):
                features = ctx.feature_extractor.extract_features(
                    ctx.session, m.node_id, window_minutes=cfg.if_window_minutes,
                    reference_time=m.end)
                vec, _ = feature_dict_to_vector(features)
                if vec is not None and np.all(np.isfinite(vec)):
                    vectors.append(vec)
                    keep.append(i)
            if len(keep) < MIN_CAL:
                return self._set_tau(None, f"insufficient_calibration:{len(keep)}<{MIN_CAL}")
            s_pt, d_pt, r_pt = pt.score_vectors(np.asarray(vectors, dtype=float))
            orientation = {seq_role: _orientation(seq), pt_role: _orientation(pt)}
            scores = {seq_role: (s_seq[keep], d_seq[keep], r_seq[keep] * orientation[seq_role]),
                      pt_role: (s_pt, d_pt, r_pt * orientation[pt_role])}
            if not all(np.all(np.isfinite(v[0])) and np.all(np.isfinite(v[2]))
                       for v in scores.values()):
                return self._set_tau(None, "non_finite_calibration_scores")
            self.calibration = {
                "alignment": f"point_window_{cfg.if_window_minutes}min_at_sequence_end",
                "cutoff": holdout.cutoff.isoformat() if holdout.cutoff else None,
                "n_calibration": len(holdout.meta), "n_used": len(keep),
                "computed_at": datetime.utcnow().isoformat(),
                "s_primary": scores["primary"][0].tolist(),
                "s_secondary": scores["secondary"][0].tolist(),
                "d_primary": scores["primary"][1].astype(bool).tolist(),
                "d_secondary": scores["secondary"][1].astype(bool).tolist(),
                # Brutos orientados (maior = mais anômalo) → posto na calibração
                "fusion_scale": "calibration_rank", "orientation": orientation,
                "r_primary": scores["primary"][2].tolist(),
                "r_secondary": scores["secondary"][2].tolist(),
            }
            self._apply_tau()
        except Exception as exc:
            logger.error(f"OnlinePipeline: calibração de τ falhou: {exc}", exc_info=True)
            self.calibration = None
            self._set_tau(None, f"error:{type(exc).__name__}")

    def _set_tau(self, tau: Optional[float], status: str, stats: Optional[dict] = None) -> None:
        self._tau = tau
        self.tau_status = status
        self.tau_stats = stats or {}
        log = logger.info if status == "ok" else logger.warning
        log(f"OnlinePipeline: τ={tau} ({status}) {self.tau_stats}")

    def _apply_tau(self) -> None:
        """Recalcula τ dos scores arquivados para o α/contamination atuais."""
        cal = self.calibration
        if not cal:
            return self._set_tau(None, self.tau_status if self.tau_status != "ok" else "not_calibrated")
        if "r_primary" not in cal or "r_secondary" not in cal:
            return self._set_tau(None, "calibration_without_raw_scores")
        # Fusão na escala de posto do bloco de calibração: score_norm é cortado em 1
        # acima do p95 do treino e, fora da amostra, empata em 1 numa fração maior que
        # a contaminação — um τ no percentil (1 − contaminação) de score_norm saturaria em 1.
        sp, ss = np.asarray(cal["s_primary"]), np.asarray(cal["s_secondary"])
        a = float(self._fusion_alpha)
        s_final = (a * self._rank("primary", cal["r_primary"])
                   + (1.0 - a) * self._rank("secondary", cal["r_secondary"]))
        tau = self.calibrate_tau(s_final, self._contamination)
        top = 1.0 - 1e-9
        stats = {
            "alpha": a, "contamination": self._contamination, "n": int(s_final.size),
            "scale": "calibration_rank",
            "tau": tau, "frac_fused_at_max": float(np.mean(s_final >= top)),
            "frac_primary_score_norm_at_1": float(np.mean(sp >= top)),
            "frac_secondary_score_norm_at_1": float(np.mean(ss >= top)),
            "frac_above_tau": None if tau is None else float(np.mean(s_final > tau)),
            "agreement": float(np.mean(np.asarray(cal["d_primary"]) == np.asarray(cal["d_secondary"]))),
        }
        if tau is None or not math.isfinite(tau):
            return self._set_tau(None, "non_finite_tau", stats)
        if tau >= top:
            # Comparação estrita s > τ com τ=1 nunca dispara: indisponível, não silencioso.
            return self._set_tau(None, "saturated", stats)
        self._set_tau(tau, "ok", stats)

    def _rank(self, role: str, oriented) -> np.ndarray:
        """CDF empírica do bloco de calibração: fração dos pontos com score ≤ ao valor."""
        cache = self.__dict__.setdefault("_rank_tables", {})
        entry = cache.get(role)
        if entry is None or entry[0] is not self.calibration:
            entry = (self.calibration,
                     np.sort(np.asarray(self.calibration[f"r_{role}"], dtype=float)))
            cache[role] = entry
        table = entry[1]
        return np.searchsorted(table, np.asarray(oriented, dtype=float), side="right") / table.size

    def fusion_ranks(self, primary: Optional[AlgoResult],
                     secondary: Optional[AlgoResult]) -> Optional[tuple]:
        """(posto do primário, posto do secundário) para a fusão; None sem calibração."""
        cal = self.calibration
        if not cal or "r_primary" not in cal or primary is None or secondary is None:
            return None
        if not (primary.is_valid and secondary.is_valid
                and primary.raw_score is not None and secondary.raw_score is not None):
            return None
        orientation = cal["orientation"]
        return (float(self._rank("primary", [primary.raw_score * orientation["primary"]])[0]),
                float(self._rank("secondary", [secondary.raw_score * orientation["secondary"]])[0]))

    def set_combination(self, combo: str, alpha: float) -> None:
        """Troca shadow/fusão/cascata reaproveitando os modelos treinados."""
        if combo not in COMBOS:
            raise ValueError(f"Combo desconhecido: {combo!r}")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"alpha fora de [0, 1]: {alpha!r}")
        with self._lock:
            if (combo == "single") != (self._secondary_runner is None):
                raise ValueError("troca entre single e combinação exige outros modelos")
            self._combo = combo
            self._combiner = Combiner(combo)
            self._fusion_alpha = float(alpha)
            self._requested = (self._requested[0], self._requested[1], combo, float(alpha))
            if self._config is not None:
                self.signature = self._config.pipeline_signature(*self._requested)
            if self.calibration:
                self._apply_tau()
        logger.info(f"OnlinePipeline: combinação {combo} α={alpha} sem retreino (τ={self._tau})")

    def status(self) -> dict:
        """Estado observável dos componentes e da calibração."""
        def runner_status(runner) -> Optional[dict]:
            if runner is None:
                return None
            trained_at = getattr(runner, "trained_at", None)
            retry = getattr(runner, "retry_in_seconds", lambda: None)()
            return {"ready": bool(runner.is_ready()), "last_error": runner.last_error,
                    "trained_at": trained_at.isoformat() if trained_at else None,
                    "retry_in_seconds": None if retry is None else round(retry, 1),
                    "training_summary": _jsonable(getattr(runner, "training_summary", {}))}
        cal = self.calibration or {}
        return {
            "primary_id": self._primary_id, "secondary_id": self._secondary_id,
            "combo": self._combo, "alpha": self._fusion_alpha, "tau": self._tau,
            "tau_status": self.tau_status, "tau_stats": self.tau_stats,
            "calibration": {k: v for k, v in cal.items() if not k.startswith(("s_", "d_", "r_"))},
            "fallback_reason": self.fallback_reason,
            "model_signature": _jsonable(self.model_signature),
            "primary": runner_status(self._primary_runner),
            "secondary": runner_status(self._secondary_runner),
        }

    def export_bundle(self) -> dict:
        """Estado congelável: modelos, normalizadores e scores de calibração."""
        with self._lock:
            if self.fallback_reason:
                # Os runners não são os pedidos (ex.: IF no lugar do LSTM-AE); um pacote com a
                # identidade pedida carregaria o modelo errado num ambiente com o algoritmo.
                raise ValueError(f"pipeline degradado ({self.fallback_reason}) não pode ser congelado")
            if not all(r.is_ready() for r in (self._primary_runner, self._secondary_runner)
                       if r is not None):
                raise ValueError("pipeline incompleto não pode ser congelado")
            return {
                "format": 1,
                "primary_id": self._requested[0], "secondary_id": self._requested[1],
                "model_signature": _jsonable(self.model_signature),
                "config": self._config.to_dict() if self._config else None,
                "primary": self._primary_runner.export_state(),
                "secondary": (self._secondary_runner.export_state()
                              if self._secondary_runner is not None else None),
                "calibration": self.calibration,
            }

    @classmethod
    def from_bundle(cls, bundle: dict, ml_settings, combo: str, alpha: float) -> "OnlinePipeline":
        """Recria o pipeline congelado; recusa pacote de outra configuração de modelos."""
        if bundle.get("format") != 1:
            raise ValueError(f"formato de pacote desconhecido: {bundle.get('format')!r}")
        pipeline = cls(bundle["primary_id"], bundle["secondary_id"], combo, alpha)
        pipeline.build_runners(ml_settings)
        if not pipeline.is_ready():
            raise ValueError("pipeline do pacote não subiu")
        if _jsonable(pipeline.model_signature) != bundle["model_signature"]:
            raise ValueError("pacote treinado com outra configuração de modelos")
        pipeline._primary_runner.import_state(bundle["primary"])
        if bundle.get("secondary") is not None:
            pipeline._secondary_runner.import_state(bundle["secondary"])
        pipeline.calibration = bundle.get("calibration")
        pipeline._tau_calibrated = True
        if pipeline.calibration:
            pipeline._apply_tau()
        else:
            pipeline._set_tau(None, "no_calibration_in_bundle")
        return pipeline


def _orientation(runner) -> float:
    """+1 se bruto maior = mais anômalo (LSTM-AE); −1 se menor = mais anômalo (IF)."""
    normalizer = getattr(runner, "_normalizer", None)
    return -1.0 if getattr(normalizer, "invert", False) else 1.0


def _jsonable(value):
    """Tuplas/datetimes → estrutura JSON estável (comparação e API)."""
    return json.loads(json.dumps(value, default=str))


# ─── Registro padrão dos algoritmos elegíveis ────────────────────────────────
# Executado na importação do módulo.

def _register_defaults() -> None:
    from ai.online.runners import PointRunner, SequenceRunner

    def _if_factory(ml_settings):
        from ai.anomaly_ml import IsolationForestModel
        cfg = resolve_effective_config(ml_settings)
        model = IsolationForestModel(
            contamination=cfg.if_contamination,
            n_estimators=cfg.if_n_estimators,
            random_state=cfg.random_seed,
        )
        return PointRunner(model=model, spec=registry_get("isolation_forest"), config=cfg)

    def _lstm_factory(ml_settings):
        from ai.lstm_autoencoder import LSTMAutoencoderModel
        cfg = resolve_effective_config(ml_settings)
        model = LSTMAutoencoderModel(
            window_size=cfg.lstm_window_size,
            hidden_size=cfg.lstm_hidden_size,
            epochs=cfg.lstm_epochs,
            patience=cfg.lstm_patience,
            threshold_percentile=cfg.lstm_threshold_percentile,
            score_aggregation=cfg.lstm_score_aggregation,
            random_seed=cfg.random_seed,
        )
        return SequenceRunner(model=model, config=cfg)

    _defaults = [
        AlgorithmSpec(
            id="isolation_forest", label="Isolation Forest",
            input_kind="point", scaling="self", incremental=True,
            factory=_if_factory,
        ),
    ]
    from ai.online.registry import _registry
    for spec in _defaults:
        if spec.id not in _registry:
            register(spec)

    # lstm_ae só é registrado quando habilitado E torch disponível. Com
    # ML_LSTM_ENABLED=false o LSTM não é registrado e torch nunca é importado.
    if settings.ML_LSTM_ENABLED:
        try:
            from ai.lstm_autoencoder import _torch_available
            if _torch_available:
                if "lstm_ae" not in _registry:
                    register(AlgorithmSpec(
                        id="lstm_ae", label="LSTM-AE",
                        input_kind="sequence", scaling="self", incremental=True,
                        factory=_lstm_factory,
                    ))
            else:
                logger.info("LSTM habilitado mas torch indisponível — lstm_ae não registrado.")
        except ImportError as exc:
            logger.warning(f"lstm_ae não registrado: {exc}")
    else:
        logger.info("ML_LSTM_ENABLED=false — lstm_ae não registrado (perfil sem PyTorch).")


_register_defaults()
