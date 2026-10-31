"""
Orquestrador da detecção de anomalias.

Responsável por:
- executar o laço de detecção em uma thread (start_ml_loop), com modelo por rede ativa,
  alimentando o pipeline online (ai.online);
- salvar anomalias no banco (com deduplicação);
- manter estatísticas em memória.

A camada de regras rápidas (fast_rule) foi descontinuada: fast_detection_enabled
permanece só por compatibilidade e recusa habilitação.
"""

import json
import logging
import math
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional
from sqlmodel import Session, select

from .anomaly_types import ANOMALY_CATALOG, get_all_types
from .feature_engineering import FeatureExtractor
from .online.effective_config import DEFAULT_ONLINE_PRIMARY
from config import settings

if TYPE_CHECKING:
    from .online.pipeline import OnlinePipeline

logger = logging.getLogger(__name__)

# Deduplicação: intervalo mínimo entre anomalias do mesmo tipo/nó (segundos)
DEDUP_INTERVAL_SECONDS = 300  # 5 minutos


def _merge_detection_evidence(classifier_ev: Optional[dict],
                              combiner_ev: Optional[dict]) -> dict:
    """Persiste a evidência do combiner (s_fusion, alpha, tau,
    secondary) junto da evidência do classificador (features, anomaly_score).

    A evidência do combiner entra como base e a do classificador sobrepõe — assim
    o score fundido e o limiar que geraram a decisão ficam gravados, sem perder as
    features do classificador em caso de colisão (ex.: ambos trazem 'features').
    """
    merged = dict(combiner_ev) if isinstance(combiner_ev, dict) else {}
    overlay = classifier_ev if isinstance(classifier_ev, dict) else {}
    for k, v in overlay.items():
        # cópia defensiva 1-nível para não mutar dicts internos do classificador
        merged[k] = dict(v) if isinstance(v, dict) else v
    return merged


def _data_origin_evidence(session, node_id: int, start: datetime, end: datetime) -> dict:
    """Linhas sintéticas que alimentam a decisão no contexto ``(start, end]``.

    Conta telemetria e vizinhos do nó e os enlaces reversos (vizinhos que o observam),
    as fontes da janela do IF e da sequência do LSTM. Não depende de ``SimulationRun``:
    beacons sintéticos sem run aberto e janelas que ainda contêm a simulação depois
    que o run fecha também contam.
    """
    from sqlalchemy import func
    from models import Neighbor, Node, Telemetry

    def count(model, *conditions) -> int:
        return int(session.exec(
            select(func.count()).select_from(model)
            .where(model.timestamp > start, model.timestamp <= end, model.data_origin != "real",
                   *conditions)
        ).one())

    synthetic = count(Telemetry, Telemetry.node_id == node_id) + count(Neighbor, Neighbor.node_id == node_id)
    node = session.exec(select(Node).where(Node.node_id == node_id)).first()
    if node is not None and node.lora_address:
        synthetic += count(Neighbor, Neighbor.neighbor_address == node.lora_address)
    return {"synthetic_rows": synthetic, "context_start": start.isoformat(),
            "context_end": end.isoformat()}


def _find_active_sim_run(node_id: int, detection_time, session, tolerance_minutes: int = 15):
    """Retorna o SimulationRun ativo (ou recentemente encerrado) para o nó no momento
    da detecção, ou None.

    Uma simulação que terminou antes de detection_time ainda pode ter causado a
    detecção enquanto seus dados estavam no contexto analisado. Por isso toleramos
    ended_at até tolerance_minutes antes de detection_time; o detector passa o contexto
    da decisão (janela do IF ou sequência do LSTM, o que for maior).
    """
    from models import SimulationRun
    from sqlmodel import select
    from datetime import timedelta
    tolerance = timedelta(minutes=tolerance_minutes)
    return session.exec(
        select(SimulationRun)
        .where(SimulationRun.node_id == node_id)
        .where(SimulationRun.started_at <= detection_time)
        .where(
            (SimulationRun.ended_at == None) |
            (SimulationRun.ended_at >= detection_time - tolerance)
        )
        .order_by(SimulationRun.started_at.desc())
    ).first()


def _no_pipeline_row(network_id: int, node_id: int, reference_time, status: Optional[str],
                     pipeline_status: str = "no_pipeline"):
    from models import OnlineInference
    return OnlineInference(network_id=network_id, node_id=node_id, reference_time=reference_time,
                           pipeline_status=pipeline_status, training_status=status)


def _signature_sha(signature) -> Optional[str]:
    if signature is None:
        return None
    import hashlib
    encoded = json.dumps(signature, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _component_threshold(result) -> Optional[float]:
    """Limiar gravado. No IF, o ``offset_`` da evidência, na escala de
    ``score_samples`` (anômalo ⇔ bruto < limiar). Só o registro muda: o AlgoResult e
    a evidência persistida na anomalia continuam sem limiar para o IF."""
    if result.threshold is not None:
        return result.threshold
    offset = (result.evidence or {}).get("offset") if result.is_valid else None
    if isinstance(offset, (int, float)) and not isinstance(offset, bool) and math.isfinite(offset):
        return float(offset)
    return None


def _component_fields(prefix: str, result) -> dict:
    if result is None:
        return {}
    return {f"{prefix}_status": result.status, f"{prefix}_reason": result.reason,
            f"{prefix}_is_anomaly": result.is_anomaly, f"{prefix}_score_norm": result.score_norm,
            f"{prefix}_raw_score": result.raw_score,
            f"{prefix}_threshold": _component_threshold(result)}


def _registro_metadata(features: dict, window_minutes) -> dict:
    """Metadados da janela gravados com as features (chaves ``_*``)."""
    from ai.anomaly_definitions import DEFINITIONS_VERSION, REGISTRO_VERSION, RULES_VERSION
    from ai.feature_engineering import FEATURE_SCHEMA_VERSION
    unavailable = features.get("_unavailable")
    return {
        "_unavailable": dict(unavailable) if isinstance(unavailable, dict) else {},
        "_window_minutes": window_minutes,
        "_rules_version": RULES_VERSION,
        "_definitions_version": DEFINITIONS_VERSION,
        "_feature_schema_version": FEATURE_SCHEMA_VERSION,
        "_registro_version": REGISTRO_VERSION,
    }


def _inference_row(network_id: int, node_id: int, reference_time, pipeline, outcome,
                   features: Optional[dict], features_reason: Optional[str], emissions: list,
                   window_minutes=None):
    """Uma linha por nó/ciclo com os dois componentes e a combinação.

    fusion_score/fusion_decision são calculados sempre que os dois componentes são
    válidos — inclusive em shadow e cascata — com o α e o τ do pipeline publicado.
    A decisão usa o score sem arredondar (igual ao combinador); a linha também
    grava os metadados da janela.
    """
    from models import OnlineInference
    from ai.anomaly_rules import evaluate_criteria
    primary, secondary = outcome.primary, outcome.secondary
    alpha, tau = pipeline.fusion_alpha, pipeline.tau
    fusion_score = fusion_decision = None
    ranks_of = getattr(pipeline, "fusion_ranks", None)
    ranks = ranks_of(primary, secondary) if callable(ranks_of) else None
    if ranks is not None:            # mesma escala de posto usada na calibração de τ
        exact = alpha * ranks[0] + (1.0 - alpha) * ranks[1]
        # Compara antes de arredondar, como o combinador (s > τ).
        fusion_decision = None if tau is None else exact > tau
        fusion_score = round(exact, 6)
    combination = outcome.combination
    hits = not_evaluable = None
    observed = None
    if features:
        evaluation = evaluate_criteria(features)
        hits, not_evaluable = evaluation.hits(), evaluation.not_evaluable()
        observed = {k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in features.items() if not k.startswith("_")}
        observed.update(_registro_metadata(features, window_minutes))
    sequence = secondary if getattr(pipeline, "_secondary_runner", None) is not None and \
        getattr(pipeline._secondary_runner, "input_kind", None) == "sequence" else primary
    return OnlineInference(
        network_id=network_id, node_id=node_id, reference_time=reference_time,
        pipeline_status="published", combo=outcome.combo,
        primary_id=outcome.primary_id, secondary_id=outcome.secondary_id,
        bundle_id=getattr(pipeline, "bundle_id", None),
        model_signature_sha=_signature_sha(getattr(pipeline, "model_signature", None)),
        alpha=alpha, tau=tau, tau_status=getattr(pipeline, "tau_status", None),
        **_component_fields("primary", primary), **_component_fields("secondary", secondary),
        sequence_context_end=(sequence.evidence.get("context_end")
                              if sequence is not None and sequence.evidence else None),
        combination_status=combination.status if combination else None,
        combination_reason=combination.reason if combination else None,
        combination_is_anomaly=combination.is_anomaly if combination else None,
        combination_score=combination.score if combination else None,
        primary_rank=None if ranks is None else round(ranks[0], 6),
        secondary_rank=None if ranks is None else round(ranks[1], 6),
        fusion_score=fusion_score, fusion_decision=fusion_decision,
        emitted=len(emissions),
        outcomes_json=json.dumps(emissions) if emissions else None,
        rule_hits=None if hits is None else json.dumps(hits),
        rule_not_evaluable=None if not_evaluable is None else json.dumps(not_evaluable),
        features_json=None if observed is None else json.dumps(observed, default=str),
        features_reason=features_reason,
    )


class AnomalyDetector:
    """Serviço principal de detecção de anomalias por IA.

    Mantém um modelo ML por rede ativa.
    """

    def __init__(self, engine, ml_settings=None):
        """
        Args:
            engine: SQLAlchemy/SQLModel engine para acesso ao banco
            ml_settings: MLSettings do banco (opcional, prevalece sobre config.py)
        """
        self.engine = engine

        # Ciclos por rede (retreino periódico do pipeline online)
        self._network_cycle_counts: dict[int, int] = {}
        # Lock para acesso thread-safe: _ml_loop e cleanup_network() competem
        self._models_lock = threading.Lock()

        # OnlinePipeline por rede (único pipeline de detecção ao vivo)
        self._online_pipelines: dict[int, "OnlinePipeline"] = {}
        self._online_primary: str = getattr(ml_settings, "ml_online_primary", DEFAULT_ONLINE_PRIMARY) if ml_settings else DEFAULT_ONLINE_PRIMARY
        self._online_secondary: Optional[str] = getattr(ml_settings, "ml_online_secondary", None) if ml_settings else None
        self._online_combo: str = getattr(ml_settings, "ml_online_combo", "single") if ml_settings else "single"
        self._online_fusion_alpha: float = getattr(ml_settings, "ml_online_fusion_alpha", 0.5) if ml_settings else 0.5

        # Extrator de features do laço online (o modo de classificação é global)
        self._feature_extractor = FeatureExtractor()

        # Controle da thread de ML em segundo plano
        self.ml_thread: threading.Thread = None
        self.ml_running = False
        self.ml_interval = 60  # segundos

        # Flag legada lida só para registrar aviso; estado efetivo sempre False
        self.fast_detection_enabled: bool = (
            ml_settings.ml_fast_detection if ml_settings else settings.ML_FAST_DETECTION_ENABLED
        )

        # Controla se nós em sim_mode são incluídos na detecção contínua
        self.detect_simulated: bool = (
            ml_settings.ml_detect_simulated if ml_settings else True
        )

        # Estatísticas em memória
        self.stats = {
            "total_detected": 0,
            "by_type": {},
            "by_severity": {"low": 0, "medium": 0, "high": 0},
            "last_detection": None,
            "started_at": datetime.utcnow().isoformat(),
            # AX descartado no fluxo contínuo (rastreabilidade
            # independente do nível de log).
            "ax_discarded_total": 0,
            "ax_discarded_by_algo": {},
        }

    # ═══════════════════════════════════════════════════════════════════
    # Gerenciamento de modelos por rede
    # ═══════════════════════════════════════════════════════════════════

    def _desired_models_signature(self, ml_settings) -> Optional[tuple]:
        """Identidade dos modelos desejados, sem a regra de combinação."""
        from ai.online.effective_config import resolve_effective_config
        try:
            cfg = resolve_effective_config(ml_settings)
        except ValueError as exc:
            logger.error(f"configuração ML inválida — mantendo pipeline atual: {exc}")
            return None
        return cfg.models_signature(self._online_primary, self._online_secondary)

    def _get_pipeline_for_network(self, network_id: int, ml_settings) -> Optional["OnlinePipeline"]:
        """Pipeline publicado (já treinado) da rede, ou None enquanto não houver.

        Mesmos modelos com outra combinação → troca sem retreino. Modelos
        diferentes ou ausentes → carrega o pacote congelado compatível ou pede um
        candidato treinado fora do ciclo; o publicado segue detectando até lá.
        """
        with self._models_lock:
            existing = self._online_pipelines.get(network_id)
        desired = self._desired_models_signature(ml_settings)
        if desired is None:
            return existing
        if existing is not None and getattr(existing, "model_signature", None) == desired:
            self._sync_combination(existing)
            return existing
        if settings.ML_ONLINE_FREEZE_MODELS and self._load_frozen(network_id, ml_settings, desired):
            with self._models_lock:
                return self._online_pipelines.get(network_id)
        reason = "initial" if existing is None else "config_changed"
        if existing is not None:
            logger.info(f"configuração de modelos mudou na rede {network_id} — "
                        "treinando candidato; pipeline atual segue ativo.")
        self._rebuild_online_pipeline(network_id, ml_settings, reason=reason)
        with self._models_lock:
            return self._online_pipelines.get(network_id)

    def _sync_combination(self, pipeline) -> None:
        combo, alpha = self._online_combo, self._online_fusion_alpha
        if (pipeline.combo, pipeline.fusion_alpha) == (combo, alpha):
            return
        try:
            pipeline.set_combination(combo, alpha)
        except ValueError as exc:
            logger.error(f"troca de combinação recusada: {exc}")

    # ─── Candidato treinado fora do ciclo e pacotes congelados ───

    _train_inline = False   # testes: treina na própria thread

    def _training(self, network_id: int) -> dict:
        states = self.__dict__.setdefault("_train_state", {})
        return states.setdefault(network_id, {"status": "idle"})

    def _rebuild_online_pipeline(self, network_id: int, ml_settings, reason: str = "rebuild") -> bool:
        """Pede um candidato com a seleção atual, treinado FORA do ciclo de detecção.

        O candidato só é publicado com todos os componentes treinados e τ calibrado
        (swap atômico real); o pipeline publicado continua detectando até lá. Após
        falha, nova tentativa automática só depois de ML_ONLINE_TRAIN_RETRY_SECONDS;
        o retreino manual ignora a espera. True se o treino começou.
        """
        from ai.online.effective_config import resolve_effective_config
        try:
            cfg = resolve_effective_config(ml_settings)
        except ValueError as exc:
            logger.error(f"configuração inválida — candidato não criado: {exc}")
            return False
        selection = (self._online_primary, self._online_secondary,
                     self._online_combo, self._online_fusion_alpha)
        state = self._training(network_id)
        with self._models_lock:
            thread = state.get("thread")
            if thread is not None and thread.is_alive():
                return False
            if reason != "manual" and time.monotonic() < state.get("retry_at", 0.0):
                return False
            state.update(status="training", reason=reason, last_error=None,
                         started_at=datetime.utcnow().isoformat(), finished_at=None,
                         selection=list(selection), thread=None)
        if self._train_inline:
            return self._train_candidate(network_id, cfg, selection, reason)
        thread = threading.Thread(target=self._train_candidate,
                                  args=(network_id, cfg, selection, reason),
                                  daemon=True, name=f"online-train-{network_id}")
        state["thread"] = thread
        thread.start()
        logger.info(f"treino do candidato da rede {network_id} iniciado ({reason})")
        return True

    def _train_candidate(self, network_id: int, cfg, selection: tuple, reason: str) -> bool:
        from ai.online.pipeline import OnlinePipeline
        state = self._training(network_id)
        try:
            pipeline = OnlinePipeline(primary_id=selection[0], secondary_id=selection[1],
                                      combo=selection[2], fusion_alpha=selection[3])
            pipeline.build_runners(cfg)
            if not pipeline.is_ready():
                raise RuntimeError("build_failed")
            with Session(self.engine) as session:
                ok = pipeline.train(self._training_context(session, network_id, cfg))
            if not ok:
                errors = [r.last_error for r in (pipeline._primary_runner,
                                                 pipeline._secondary_runner)
                          if r is not None and r.last_error]
                raise RuntimeError("; ".join(errors) or "train_failed")
            self._publish_pipeline(network_id, pipeline, reason)
            state.update(status="ready", finished_at=datetime.utcnow().isoformat(),
                         last_error=None, retry_at=0.0)
            return True
        except Exception as exc:
            from ai.online.runners import _retry_seconds
            state.update(status="failed", finished_at=datetime.utcnow().isoformat(),
                         last_error=f"{type(exc).__name__}: {exc}" if not isinstance(exc, RuntimeError)
                         else str(exc),
                         retry_at=time.monotonic() + _retry_seconds())
            logger.warning(f"candidato da rede {network_id} não ficou pronto: "
                           f"{state['last_error']}")
            return False

    def _training_context(self, session, network_id: int, cfg):
        """Contexto próprio da thread de treino: sessão, extrator e corte independentes."""
        from models import Node
        from ai.online.registry import DetectionContext
        from ai.reference_policy import ReferencePolicy
        node_ids = [n.node_id for n in session.exec(
            select(Node).where(Node.network_id == network_id)).all()]
        if not node_ids:
            raise RuntimeError("network_without_nodes")
        extractor = FeatureExtractor()
        extractor.anomaly_classification_mode = self._feature_extractor.anomaly_classification_mode
        return DetectionContext(
            session=session, node_id=node_ids[0], feature_extractor=extractor,
            features_vec=None, network_node_ids=node_ids, network_id=network_id,
            reference_time=datetime.utcnow(),
            reference_policy=ReferencePolicy.from_session(session, cfg.reference_margin_minutes),
        )

    def _publish_pipeline(self, network_id: int, pipeline, reason: str) -> None:
        # A combinação pode ter mudado durante o treino; os modelos continuam válidos.
        self._sync_combination(pipeline)
        pipeline.bundle_id = None
        if settings.ML_ONLINE_FREEZE_MODELS or reason == "manual":
            try:
                from ai.online.bundles import save_bundle
                manifest = save_bundle(pipeline, network_id, settings.ML_ONLINE_BUNDLE_DIR, reason)
                pipeline.bundle_id = manifest["bundle_id"]
            except Exception as exc:
                logger.error(f"pacote da rede {network_id} não foi salvo: {exc}",
                             exc_info=True)
        with self._models_lock:
            self._online_pipelines[network_id] = pipeline
        logger.info(f"pipeline da rede {network_id} publicado ({reason}; "
                    f"τ={pipeline.tau}, {pipeline.tau_status}; pacote={pipeline.bundle_id})")

    def _load_frozen(self, network_id: int, ml_settings, models_signature) -> bool:
        """Publica o pacote congelado mais recente compatível; False se não houver."""
        from ai.online.bundles import load_latest_bundle
        from ai.online.pipeline import OnlinePipeline
        try:
            found = load_latest_bundle(network_id, models_signature, settings.ML_ONLINE_BUNDLE_DIR)
            if found is None:
                return False
            manifest, bundle = found
            pipeline = OnlinePipeline.from_bundle(bundle, ml_settings, self._online_combo,
                                                  self._online_fusion_alpha)
        except Exception as exc:
            logger.error(f"pacote congelado da rede {network_id} não carregou: {exc}",
                         exc_info=True)
            return False
        pipeline.bundle_id = manifest["bundle_id"]
        with self._models_lock:
            self._online_pipelines[network_id] = pipeline
        self._training(network_id).update(status="loaded", reason="frozen_bundle",
                                          bundle_id=manifest["bundle_id"])
        logger.info(f"rede {network_id} usando pacote congelado {manifest['bundle_id']}")
        return True

    def request_online_retrain(self, ml_settings, network_ids: list[int]) -> dict:
        """Retreino manual (ignora congelamento e espera); publica e salva novo pacote."""
        started = {}
        for network_id in network_ids:
            started[network_id] = self._rebuild_online_pipeline(network_id, ml_settings,
                                                                reason="manual")
        return started

    def online_status(self) -> dict:
        """Estado dos pipelines, treino, τ e último ciclo por rede."""
        with self._models_lock:
            pipelines = dict(self._online_pipelines)
        trainings = self.__dict__.get("_train_state", {})
        cycles = self.__dict__.get("_last_cycle", {})
        networks = {}
        for network_id in sorted(set(pipelines) | set(trainings) | set(cycles)):
            pipeline = pipelines.get(network_id)
            published = None
            if pipeline is not None:
                published = {**pipeline.status(), "bundle_id": getattr(pipeline, "bundle_id", None)}
            training = {k: v for k, v in trainings.get(network_id, {}).items()
                        if k not in ("thread", "retry_at")}
            retry_at = trainings.get(network_id, {}).get("retry_at")
            if retry_at:
                training["retry_in_seconds"] = round(max(0.0, retry_at - time.monotonic()), 1)
            networks[network_id] = {"published": published, "training": training,
                                    "last_cycle": cycles.get(network_id)}
        return {
            "frozen": settings.ML_ONLINE_FREEZE_MODELS,
            "selection": {"primary": self._online_primary, "secondary": self._online_secondary,
                          "combo": self._online_combo, "alpha": self._online_fusion_alpha},
            "networks": networks,
        }

    def _maybe_retrain_online(self, network_id: int, new_cycle: int, ml_settings) -> None:
        """Retreino periódico (~1 h) em segundo plano; desligado com modelos congelados."""
        if settings.ML_ONLINE_FREEZE_MODELS:
            return
        if new_cycle % 60 == 0:
            self._rebuild_online_pipeline(network_id, ml_settings, reason="periodic")

    def cleanup_network(self, network_id: int):
        """Remove modelo ML de rede desativada. Thread-safe."""
        with self._models_lock:
            self._network_cycle_counts.pop(network_id, None)
            self._online_pipelines.pop(network_id, None)
        logger.info(f"Modelo ML da rede {network_id} removido da memória.")

    # Camada rápida descontinuada. O estado efetivo é sempre False;
    # flags legadas (env, MLSettings, API) são ignoradas com aviso.
    @property
    def fast_detection_enabled(self) -> bool:
        return False

    @fast_detection_enabled.setter
    def fast_detection_enabled(self, value) -> None:
        if value:
            logger.warning("fast_rule descontinuada — pedido de habilitação ignorado.")

    def set_fast_detection(self, enabled: bool):
        """Mantido por compatibilidade; a camada rápida não pode ser habilitada."""
        self.fast_detection_enabled = enabled

    def apply_settings(self, ml_settings) -> bool:
        """
        Aplica parâmetros ML vindos do banco/API.

        Mudança de modelos (algoritmo, contamination, árvores, W, histórico…)
        treina candidatos do pipeline online; mudança só de combinação ou α
        reaproveita os modelos. Thread-safe via _models_lock.

        Returns:
            True se algum pipeline publicado ganhou candidato novo.
        """
        _online_changed = False
        _net_ids_to_rebuild = []

        with self._models_lock:
            # Valor legado ignorado (setter registra aviso)
            self.fast_detection_enabled = ml_settings.ml_fast_detection

            # Detecção de nós simulados
            self.detect_simulated = ml_settings.ml_detect_simulated

            # Modo de classificação no extrator do laço online
            mode = getattr(ml_settings, "anomaly_classification_mode", "contextual")
            self._feature_extractor.anomaly_classification_mode = mode
            logger.info(f"anomaly_classification_mode={mode!r}")

            # hot-swap da seleção online de algoritmos.
            # Captura a mudança aqui; o rebuild (swap atômico) roda FORA do lock.
            new_primary = getattr(ml_settings, "ml_online_primary", DEFAULT_ONLINE_PRIMARY)
            new_secondary = getattr(ml_settings, "ml_online_secondary", None)
            new_combo = getattr(ml_settings, "ml_online_combo", "single")
            new_alpha = getattr(ml_settings, "ml_online_fusion_alpha", 0.5)
            _online_changed = (
                new_primary != self._online_primary
                or new_secondary != self._online_secondary
                or new_combo != self._online_combo
                or new_alpha != self._online_fusion_alpha
            )
            if _online_changed:
                self._online_primary = new_primary
                self._online_secondary = new_secondary
                self._online_combo = new_combo
                self._online_fusion_alpha = new_alpha
                logger.info(
                    f"seleção online — "
                    f"{new_combo}:{new_primary}+{new_secondary or '-'}"
                )

        # Só outros MODELOS (algoritmos, W, histórico,
        # contamination, árvores…) exigem candidato novo; trocar shadow/fusão/cascata
        # ou α reaproveita os modelos treinados, sem janela cega.
        desired = self._desired_models_signature(ml_settings)
        with self._models_lock:
            published = list(self._online_pipelines.items())
        for _nid, _pipeline in published:
            current = getattr(_pipeline, "model_signature", None)
            if desired is None or not isinstance(current, tuple):
                continue
            if current != desired:
                _net_ids_to_rebuild.append(_nid)
            else:
                self._sync_combination(_pipeline)
        if _net_ids_to_rebuild:
            logger.info(
                f"modelos mudaram — treinando candidatos para as redes {_net_ids_to_rebuild}"
            )

        # Candidato treinado fora do ciclo; o publicado segue ativo até lá.
        for _nid in _net_ids_to_rebuild:
            self._rebuild_online_pipeline(_nid, ml_settings, reason="config_changed")

        return bool(_net_ids_to_rebuild)

    def _save_and_notify(self, session: Session, node_id: int,
                         anomaly: dict, confidence: float = None,
                         detector_algorithm: Optional[str] = None,
                         detector_combo: Optional[str] = None,
                         detector_primary: Optional[str] = None,
                         detector_secondary: Optional[str] = None,
                         network_id: Optional[int] = None,
                         context_minutes: int = 15):
        """
        Salva anomalia no BD.
        Inclui deduplicação: mesma anomalia não repete em < 5 minutos.
        """
        # AX (outlier sem regra A1-A7) não entra no fluxo contínuo —
        # só a faixa determinística A1-A7 gera registro de anomalia.
        # Guarda no topo: descarta antes de dedup, insert, stats e
        # latch de simulação (TTD).
        if anomaly["anomaly_type"] == "AX":
            algo = detector_algorithm or anomaly.get("detector_algorithm") or "?"
            self.stats["ax_discarded_total"] = (
                self.stats.get("ax_discarded_total", 0) + 1
            )
            by_algo = self.stats.setdefault("ax_discarded_by_algo", {})
            by_algo[algo] = by_algo.get(algo, 0) + 1
            logger.debug(
                f"AX descartado no fluxo contínuo "
                f"(nó {node_id}, algo={algo}): {anomaly.get('description', '')}"
            )
            return {"status": "ax_discarded"}

        from models import Anomaly

        # ─── Deduplicação ───
        # Modos shadow emitem anomalias com detector_algorithm distinto no mesmo nó/tipo
        # (if_shadow e lstm_shadow são rastreados independentemente).
        algo = detector_algorithm or anomaly.get("detector_algorithm")
        # Strings shadow legadas + padrão shadow:*
        _is_shadow = algo in {"if_shadow", "lstm_shadow"} or (
            bool(algo) and algo.startswith("shadow:")
        )
        dedup_query = (
            select(Anomaly)
            .where(Anomaly.node_id == node_id)
            .where(Anomaly.anomaly_type == anomaly["anomaly_type"])
            .where(Anomaly.resolved == False)
        )
        if _is_shadow:
            # Para shadow: dedup por (node, type, detector_algorithm) —
            # permite que IF e LSTM-AE registrem a mesma classe de forma independente.
            dedup_query = dedup_query.where(Anomaly.detector_algorithm == algo)
        elif detector_combo:
            # Após a troca de modo, um registro do modo anterior não
            # suprime o primeiro alerta do modo novo (contagens comparáveis).
            dedup_query = dedup_query.where(Anomaly.detector_combo == detector_combo)
        recent = session.exec(dedup_query.order_by(Anomaly.timestamp.desc())).first()

        if recent:
            delta = (datetime.utcnow() - recent.timestamp).total_seconds()
            if delta < DEDUP_INTERVAL_SECONDS:
                logger.debug(
                    f"Anomalia [{anomaly['anomaly_type']}] nó {node_id} "
                    f"ignorada (duplicata, {delta:.0f}s < {DEDUP_INTERVAL_SECONDS}s)"
                )
                return {"status": "dedup", "duplicate_of": recent.id}

        # ─── Salvar no banco ───
        evidence = anomaly.get("evidence", {})
        if isinstance(evidence, dict):
            evidence_str = json.dumps(evidence)
        else:
            evidence_str = str(evidence)

        # Algoritmo de detecção derivado da combinação (ex.: "single:if",
        # "fusion:if+lstm"); "if" quando nenhum foi informado.
        algo = (
            anomaly.get("detector_algorithm")
            or detector_algorithm
            or "if"
        )
        db_anomaly = Anomaly(
            node_id=node_id,
            # A6: o vizinho do enlace assimétrico; o nó da linha é só quem observou.
            neighbor_address=anomaly.get("neighbor_address"),
            anomaly_type=anomaly["anomaly_type"],
            category=anomaly["category"],
            severity=anomaly["severity"],
            description=anomaly["description"],
            suggested_action=anomaly.get("suggested_action"),
            evidence=evidence_str,
            detection_method=anomaly.get("detection_method", "ml"),
            confidence=confidence,
            # Modo efetivo usado nesta detecção (auditoria A/B)
            classification_mode_used=anomaly.get("classification_mode_used"),
            # Rastreabilidade do algoritmo
            detector_algorithm=algo,
            # Proveniência estruturada
            detector_combo=detector_combo or anomaly.get("detector_combo"),
            detector_primary=detector_primary or anomaly.get("detector_primary"),
            detector_secondary=detector_secondary or anomaly.get("detector_secondary"),
        )
        session.add(db_anomaly)
        session.commit()
        session.refresh(db_anomaly)

        # ─── Vincular anomalia ao SimulationRun ativo ───
        detection_time = db_anomaly.timestamp
        active_run = _find_active_sim_run(node_id, detection_time, session,
                                          tolerance_minutes=context_minutes)
        if active_run:
            db_anomaly.sim_run_id = active_run.id
            session.add(db_anomaly)
            # Registrar apenas a primeira detecção (TTD = tempo real de detecção)
            if active_run.detected_at is None:
                active_run.detected_at = detection_time
                active_run.detector_primary = db_anomaly.detector_primary
                active_run.detector_secondary = db_anomaly.detector_secondary
                active_run.detector_combo = db_anomaly.detector_combo
                session.add(active_run)
            session.commit()

        # ─── Atualizar estatísticas em memória ───
        self.stats["total_detected"] += 1
        self.stats["by_type"][anomaly["anomaly_type"]] = \
            self.stats["by_type"].get(anomaly["anomaly_type"], 0) + 1
        self.stats["by_severity"][anomaly["severity"]] = \
            self.stats["by_severity"].get(anomaly["severity"], 0) + 1
        self.stats["last_detection"] = datetime.utcnow().isoformat()

        # ─── Log ───
        severity_emoji = {"low": "ℹ️", "medium": "⚠️", "high": "🔴"}
        method = anomaly.get("detection_method", "ml")
        method_label = "⚡ Rápida" if method == "fast_rule" else "🤖 IF"
        logger.warning(
            f"{severity_emoji.get(anomaly['severity'], '❓')} "
            f"ANOMALIA [{anomaly['anomaly_type']}] [{method_label}] "
            f"Nó {node_id}: {anomaly['description']}"
        )

        return {"status": "saved", "anomaly_id": db_anomaly.id}

    def _log_inference(self, session, rows: list) -> None:
        """Grava as decisões do ciclo; falha de registro não para a detecção."""
        if not rows or not settings.ML_ONLINE_INFERENCE_LOG:
            return
        try:
            session.add_all(rows)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"registro online_inference falhou: {exc}")

    def _record_cycle(self, network_id: int, reference_time, started: float, rows: list) -> None:
        from collections import Counter
        cycles = self.__dict__.setdefault("_last_cycle", {})
        cycles[network_id] = {
            "reference_time": reference_time.isoformat(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "nodes": len(rows),
            "primary_status": dict(Counter(r.primary_status for r in rows)),
            "secondary_status": dict(Counter(r.secondary_status for r in rows)),
            "combination_status": dict(Counter(r.combination_status for r in rows)),
            "emitted": sum(r.emitted for r in rows),
        }

    def get_stats(self, network_id: int = None) -> dict:
        """Estatísticas em memória e estado do pipeline online, que é o único que decide.

        Com ``network_id``, o estado online fica restrito àquela rede.
        """
        online = self.online_status()
        if network_id is not None:
            online["networks"] = {k: v for k, v in online["networks"].items() if k == network_id}
        published = [n["published"] for n in online["networks"].values() if n.get("published")]
        ready = any((p.get("primary") or {}).get("ready") for p in published)
        return {
            **self.stats,
            "ml_status": "active" if self.ml_running else ("trained" if ready else "not_trained"),
            "online": online,
            "features": self._feature_extractor.feature_names,
            "detectable_types": get_all_types(),
            "ml_interval": self.ml_interval,
            "fast_detection_enabled": self.fast_detection_enabled,
        }

    # ═══════════════════════════════════════════════════════════════════
    # ML DETECTION LOOP — por rede ativa
    # ═══════════════════════════════════════════════════════════════════

    def start_ml_loop(self, interval_seconds: int = 60):
        """Inicia thread de detecção ML em background."""
        if self.ml_running:
            logger.warning("ML loop já está rodando")
            return

        self.ml_interval = interval_seconds
        self.ml_running = True
        self.ml_thread = threading.Thread(target=self._ml_loop, daemon=True)
        self.ml_thread.start()

        logger.info(f"🤖 ML detection loop iniciado (intervalo={interval_seconds}s)")

    def _ml_loop(self):
        """Loop principal de detecção ML — itera por rede ativa."""
        time.sleep(30)

        while self.ml_running:
            try:
                with Session(self.engine) as session:
                    from models import Network
                    networks = session.exec(
                        select(Network).where(Network.active == True)
                    ).all()

                    for network in networks:
                        try:
                            self._detect_for_network(session, network.id)
                        except Exception as e:
                            logger.error(
                                f"Erro na detecção para rede {network.id}: {e}",
                                exc_info=True
                            )

            except Exception as e:
                logger.error(f"Erro no ML loop: {e}", exc_info=True)

            time.sleep(self.ml_interval)

    def _detect_for_network(self, session: Session, network_id: int):
        """Executa detecção de anomalias para uma rede específica."""
        from models import Node

        nodes = session.exec(
            select(Node).where(Node.network_id == network_id)
        ).all()

        if not nodes:
            return

        # Filtrar nós simulados da detecção se a flag estiver desativada
        if not self.detect_simulated:
            sim_nodes = [n for n in nodes if n.sim_mode]
            if sim_nodes:
                logger.info(
                    f"[{network_id}] {len(sim_nodes)} nó(s) em simulação excluídos da "
                    f"detecção contínua (ml_detect_simulated=False): "
                    f"{[n.node_id for n in sim_nodes]}"
                )
            nodes = [n for n in nodes if not n.sim_mode]

        if not nodes:
            logger.debug(f"[{network_id}] Nenhum nó disponível para detecção neste ciclo.")
            return

        node_ids = [n.node_id for n in nodes]
        with self._models_lock:
            cycle = self._network_cycle_counts.get(network_id, 0)

        # A operação ao vivo usa SEMPRE o OnlinePipeline (ml_use_hybrid é coluna legada sem efeito).
        from models import MLSettings
        from ai.online.effective_config import resolve_effective_config
        from ai.reference_policy import ReferencePolicy
        _ml_s = session.exec(select(MLSettings).where(MLSettings.id == 1)).first()
        # Um único corte por ciclo, compartilhado por IF, LSTM,
        # classificação e treino, com a mesma política de referência saudável.
        reference_time = datetime.utcnow()
        try:
            margin = resolve_effective_config(_ml_s).reference_margin_minutes
        except ValueError:
            margin = 0.0
        reference_policy = ReferencePolicy.from_session(session, margin_minutes=margin)
        self._run_online_pipeline(
            session, network_id, self._feature_extractor, node_ids, _ml_s,
            reference_time=reference_time, reference_policy=reference_policy,
        )

        new_cycle = cycle + 1
        with self._models_lock:
            self._network_cycle_counts[network_id] = new_cycle
        self._maybe_retrain_online(network_id, new_cycle, _ml_s)

    def _run_online_pipeline(
        self, session, network_id: int, feature_extractor, node_ids: list, ml_settings=None,
        reference_time=None, reference_policy=None,
    ):
        """
        Executa detecção via OnlinePipeline para todos os nós da rede.
        Único caminho de detecção ao vivo.
        Corte explícito por ciclo; ausência de features pontuais não pula
        o nó — o pipeline registra a indisponibilidade de cada componente.
        A decisão de cada nó/ciclo (normal, anômala ou indisponível,
        com os dois componentes) é registrada em online_inference.
        """
        from ai.online.registry import DetectionContext
        from ai.online.effective_config import resolve_effective_config
        from ai.sequence_builder import feature_dict_to_vector
        import numpy as np

        started = time.monotonic()
        if reference_time is None:
            reference_time = datetime.utcnow()
        pipeline = self._get_pipeline_for_network(network_id, ml_settings)
        if pipeline is None:
            training = self._training(network_id)
            status = training.get("status")
            if training.get("last_error"):
                status = f"{status}: {training['last_error']}"
            rows = [_no_pipeline_row(network_id, node_id, reference_time, status)
                    for node_id in node_ids]
            self._log_inference(session, rows)
            self._record_cycle(network_id, reference_time, started, rows)
            return
        try:
            _cfg = resolve_effective_config(ml_settings)
            window_minutes = _cfg.if_window_minutes
            # Contexto da decisão: janela do IF ou sequência do LSTM, o que for maior.
            context_minutes = max(window_minutes, _cfg.lstm_window_size * _cfg.lstm_window_minutes)
        except ValueError:
            window_minutes = context_minutes = settings.ML_WINDOW_MINUTES
        count = 0
        rows = []
        for node_id in node_ids:
            features = vec = features_reason = None
            origin = None
            try:
                features = feature_extractor.extract_features(
                    session, node_id,
                    window_minutes=window_minutes,
                    reference_time=reference_time,
                )
                vec, features_reason = feature_dict_to_vector(features)
                features_vec = None if vec is None else np.array(vec, dtype=np.float32)
                ctx = DetectionContext(
                    session=session,
                    node_id=node_id,
                    feature_extractor=feature_extractor,
                    features_vec=features_vec,
                    network_node_ids=node_ids,
                    network_id=network_id,
                    reference_time=reference_time,
                    reference_policy=reference_policy,
                    features_reason=features_reason,
                )
                outcome = pipeline.detect_detailed(ctx)
                emissions = []
                for rd in outcome.anomalies:
                    from ai.anomaly_ml import classify_anomaly_type
                    if features_vec is not None:
                        # O 2º argumento é um pseudo-score (−0,5·score_norm, menor = mais anômalo)
                        # só para o campo confidence (intensidade relativa, não confiança calibrada); vai para evidence["anomaly_score"]
                        # mas não é o score do modelo (este fica em evidence["score_norm"]).
                        classified = classify_anomaly_type(
                            features, -min(rd["score_norm"] * 0.5, 0.5)
                        )
                        if not classified:
                            # classify_anomaly_type devolve ao menos AX; lista vazia é defeito.
                            raise RuntimeError("classificação vazia para detecção do pipeline online")
                        anomaly = classified[0]
                    else:
                        # Sem vetor válido não há
                        # atribuição de causa → AX (descartado no fluxo contínuo), com os
                        # motivos preservados; ausência não é normalidade.
                        anomaly = {
                            "anomaly_type": "AX", "category": "statistical",
                            "severity": "low",
                            "description": "Detecção sem features pontuais para atribuição de causa",
                            "detection_method": "ml", "confidence": 0,
                            "classification_mode_used": None,
                            "evidence": {
                                "features_reason": features_reason,
                                "unavailable_features": dict((features or {}).get("_unavailable") or {}),
                            },
                        }
                    anomaly["node_id"] = node_id
                    # Preserva s_fusion/alpha/tau (e secondary) do
                    # combiner no evidence persistido, sem perder as features do
                    # classificador — base quantitativa p/ auditar a decisão de fusão.
                    anomaly["evidence"] = _merge_detection_evidence(
                        anomaly.get("evidence"), rd.get("evidence")
                    )
                    # score_norm vem no topo do dict do combiner e se perderia
                    # no merge (que só recebe rd["evidence"]). Copia p/ o evidence — assim
                    # fica persistido no banco como métrica canônica.
                    if rd.get("score_norm") is not None:
                        anomaly["evidence"]["score_norm"] = rd["score_norm"]
                    # Distinguir corte/janela das features do contexto
                    # sequencial e do estado cadastral lido no enfileiramento.
                    from datetime import timedelta
                    anomaly["evidence"]["reference_time"] = reference_time.isoformat()
                    anomaly["evidence"]["feature_window"] = {
                        "start": (reference_time - timedelta(minutes=window_minutes)).isoformat(),
                        "end": reference_time.isoformat(),
                    }
                    if origin is None:
                        origin = _data_origin_evidence(
                            session, node_id,
                            reference_time - timedelta(minutes=context_minutes), reference_time)
                    anomaly["evidence"]["data_origin"] = origin
                    saved = self._save_and_notify(
                        session,
                        node_id=node_id,
                        anomaly=anomaly,
                        confidence=anomaly.get("confidence"),
                        detector_algorithm=rd.get("detector_algorithm"),
                        detector_combo=rd.get("detector_combo"),
                        detector_primary=rd.get("detector_primary"),
                        detector_secondary=rd.get("detector_secondary"),
                        network_id=network_id,
                        context_minutes=context_minutes,
                    )
                    emissions.append({"detector_algorithm": rd.get("detector_algorithm"),
                                      "anomaly_type": anomaly["anomaly_type"],
                                      **(saved if isinstance(saved, dict) else {"status": "unknown"})})
                    count += 1
                rows.append(_inference_row(network_id, node_id, reference_time, pipeline,
                                           outcome, features, features_reason, emissions,
                                           window_minutes=window_minutes))
            except Exception as exc:
                logger.error(f"erro no pipeline online nó {node_id}: {exc}", exc_info=True)
                rows.append(_no_pipeline_row(network_id, node_id, reference_time,
                                             f"cycle_error: {type(exc).__name__}: {exc}",
                                             pipeline_status="published"))

        self._log_inference(session, rows)
        self._record_cycle(network_id, reference_time, started, rows)

        if count:
            logger.info(
                f"Rede {network_id} detectou {count} anomalia(s) "
                f"[{self._online_combo}:{self._online_primary}]"
            )

    def stop_ml_loop(self):
        """Para o loop de detecção ML."""
        if not self.ml_running:
            return

        logger.info("Parando ML detection loop...")
        self.ml_running = False

        if self.ml_thread and self.ml_thread.is_alive():
            self.ml_thread.join(timeout=5)

        logger.info("ML detection loop parado")
