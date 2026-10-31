"""
SequenceBuilder — constrói tensores (batch, W, n_features) para o LSTM-AE.

O IF opera sobre um vetor por janela (stateless). O LSTM-AE precisa de sequências:
    shape (W, n_features) — para inferência de um nó
    shape (N, W, n_features) — para treino (sliding windows)

Política única de lacunas (treino, calibração, replay e inferência):
    - primeira sub-janela ausente → rejeita (`missing_initial_context`); nunca zeros
    - após observação válida, forward-fill até max_impute_gap sub-janelas
      consecutivas, com máscara observado/imputado
    - acima disso → rejeita (`gap_exceeds_limit`)
    - feature obrigatória ausente ou não finita → rejeita (`invalid_feature:*`)
    - feature não observada na janela (`_unavailable`) → rejeita
      (`missing_observation:*`); o valor de apresentação nunca entra como medida

Referência saudável e tempo:
    - `build_reference_dataset` exige `end_time` explícito e aplica `ReferencePolicy`
      (histórico completo de SimulationRun, linhas sintéticas, enlaces reversos)
    - `split_train_val_calib` separa treino | validação | calibração por dois cortes
      GLOBAIS; sequências que cruzam um corte são descartadas (purga)
"""

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
from sqlmodel import Session
from ai.anomaly_definitions import FEATURE_ORDER

logger = logging.getLogger(__name__)

# Nomes canônicos das 11 features
FEATURE_NAMES: List[str] = list(FEATURE_ORDER)
N_FEATURES: int = len(FEATURE_NAMES)

# Motivos de invalidade
MISSING_INITIAL_CONTEXT = "missing_initial_context"
MISSING_OBSERVATION = "missing_observation"
GAP_EXCEEDS_LIMIT = "gap_exceeds_limit"
INVALID_FEATURE = "invalid_feature"
NODE_IN_SIM_MODE = "node_in_sim_mode"


def feature_dict_to_vector(features: Optional[dict]) -> tuple[Optional[List[float]], Optional[str]]:
    """Dict → vetor na ordem canônica, sem preencher ausência com zero.

    Feature listada em ``_unavailable`` não foi observada na janela (ex.: menos de
    dois contadores para a perda, nenhum enlace reverso). O número que o extrator
    deixa nela (0, ou 5 no índice de qualidade) parece saudável; por isso o vetor é
    recusado, e o modelo decide "dados insuficientes", nunca "normal".

    Returns:
        (vetor, None) se todas as features obrigatórias foram observadas e são finitas;
        (None, motivo) caso contrário.
    """
    if features is None:
        return None, "no_data"
    unavailable = features.get("_unavailable") or {}
    vec: List[float] = []
    for name in FEATURE_NAMES:
        value = features.get(name)
        if value is None:
            return None, f"{INVALID_FEATURE}:missing:{name}"
        if name in unavailable:
            return None, f"{MISSING_OBSERVATION}:{name}:{unavailable[name]}"
        value = float(value)
        if not math.isfinite(value):
            return None, f"{INVALID_FEATURE}:non_finite:{name}"
        vec.append(value)
    return vec, None


def vectorize_windows(records, features_of=lambda record: record[1]):
    """Matriz das janelas válidas com os registros retidos alinhados.

    Args:
        records: registros que carregam features (padrão: tuplas ``(node_id, features)``).
        features_of: acessor do dict de features em cada registro.

    Returns:
        ``(X, kept, exclusions)`` — ``X`` float64 ``(len(kept), N_FEATURES)``;
        ``kept`` são os próprios registros válidos, na ordem das linhas de ``X``;
        ``exclusions`` conta os motivos ``invalid_feature:*``/``no_data``. Scores e
        classificações devem ser associados a ``kept``, nunca à lista original.
    """
    rows: List[List[float]] = []
    kept: list = []
    exclusions: Counter = Counter()
    for record in records:
        vec, reason = feature_dict_to_vector(features_of(record))
        if vec is None:
            exclusions[reason] += 1
            continue
        rows.append(vec)
        kept.append(record)
    X = np.array(rows, dtype=np.float64).reshape(len(rows), N_FEATURES)
    return X, kept, exclusions


@dataclass
class SequenceExtraction:
    """Sequência de um nó com fronteiras, máscara de imputação e motivo de invalidade."""
    node_id: int
    start: datetime
    end: datetime
    windows: Optional[np.ndarray] = None          # (W, N_FEATURES) float32 quando válida
    imputed_mask: List[bool] = field(default_factory=list)
    reason: Optional[str] = None                  # None = válida

    @property
    def is_valid(self) -> bool:
        return self.reason is None and self.windows is not None


@dataclass
class ReferenceDataset:
    """Sequências de referência saudável e contagens de exclusão."""
    X: Optional[np.ndarray]
    meta: List[SequenceExtraction]
    exclusions: Counter
    start: datetime
    end: datetime

    def summary(self) -> dict:
        return {
            "n_sequences": 0 if self.X is None else int(len(self.X)),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "exclusions": dict(self.exclusions),
            "imputed_windows": int(sum(sum(m.imputed_mask) for m in self.meta)),
        }


class SequenceBuilder:
    """
    Constrói sequências de feature-vectors a partir do histórico persistido.

    Cada sequência = W janelas de window_minutes minutos consecutivas do mesmo nó.
    Usa FeatureExtractor.extract_for_window() para cada sub-janela (início, fim].

    """

    def __init__(
        self,
        window_size: int = 12,
        window_minutes: int = 5,
        stride: int = 1,
        max_impute_gap: int = 2,
    ):
        """
        Args:
            window_size:      W — sub-janelas por sequência (padrão 12 = 60 min)
            window_minutes:   Duração de cada sub-janela (padrão 5 min)
            stride:           Passo do sliding window de treino (em sub-janelas)
            max_impute_gap:   Máximo de sub-janelas faltantes consecutivas para
                              forward-fill após uma observação válida.
        """
        self.window_size = window_size
        self.window_minutes = window_minutes
        self.stride = stride
        self.max_impute_gap = max_impute_gap

    @property
    def span(self) -> timedelta:
        return timedelta(minutes=self.window_size * self.window_minutes)

    # ──────────────────────────────────────────────────────────────────────────
    # Núcleo: uma sequência com política de lacunas
    # ──────────────────────────────────────────────────────────────────────────

    def extract_sequence(
        self,
        session: Session,
        node_id: int,
        feature_extractor,
        start: datetime,
        reference_policy=None,
    ) -> SequenceExtraction:
        """Extrai W sub-janelas a partir de ``start`` aplicando a política única de lacunas.

        Com ``reference_policy`` (treino/calibração), rejeita sequências cujo
        contexto intersecta simulação histórica ou contém linhas sintéticas.
        """
        ext = SequenceExtraction(node_id=node_id, start=start, end=start + self.span)
        if reference_policy is not None and reference_policy.node_contaminated(
            [node_id], ext.start, ext.end
        ):
            from ai.reference_policy import SIMULATION_OVERLAP
            ext.reason = SIMULATION_OVERLAP
            return ext

        rows: List[List[float]] = []
        mask: List[bool] = []
        last_valid: Optional[List[float]] = None
        consecutive_gap = 0

        for i in range(self.window_size):
            w_start = start + timedelta(minutes=i * self.window_minutes)
            w_end = w_start + timedelta(minutes=self.window_minutes)

            if reference_policy is not None:
                prov: dict = {}
                features = feature_extractor.extract_for_window(
                    session, node_id, w_start, w_end, provenance=prov
                )
                # Checada ANTES de imputar: janela esparsa contaminada não pode
                # virar lacuna comum preenchida por forward-fill.
                reason = reference_policy.check_window(node_id, w_start, w_end, prov)
                if reason:
                    ext.reason = reason
                    return ext
            else:
                features = feature_extractor.extract_for_window(session, node_id, w_start, w_end)

            if features is not None:
                vec, invalid = feature_dict_to_vector(features)
                if invalid:
                    ext.reason = invalid
                    return ext
                consecutive_gap = 0
                last_valid = vec
                rows.append(vec)
                mask.append(False)
                continue

            if last_valid is None:
                ext.reason = MISSING_INITIAL_CONTEXT
                return ext
            consecutive_gap += 1
            if consecutive_gap > self.max_impute_gap:
                ext.reason = GAP_EXCEEDS_LIMIT
                return ext
            rows.append(list(last_valid))
            mask.append(True)

        ext.windows = np.array(rows, dtype=np.float32)
        ext.imputed_mask = mask
        return ext

    # ──────────────────────────────────────────────────────────────────────────
    # API pública de inferência
    # ──────────────────────────────────────────────────────────────────────────

    def build_inference_extraction(
        self,
        session: Session,
        node_id: int,
        feature_extractor,
        reference_time: Optional[datetime] = None,
        exclude_sim: bool = False,
    ) -> SequenceExtraction:
        """As W sub-janelas que terminam em ``reference_time`` (corte explícito)."""
        if reference_time is None:
            reference_time = datetime.utcnow()
        start = reference_time - self.span
        if exclude_sim:
            from models import Node
            node = session.get(Node, node_id)
            if node and node.sim_mode:
                return SequenceExtraction(node_id=node_id, start=start, end=reference_time,
                                          reason=NODE_IN_SIM_MODE)
        return self.extract_sequence(session, node_id, feature_extractor, start)

    def build_inference_sequence(
        self,
        session: Session,
        node_id: int,
        feature_extractor,
        reference_time: Optional[datetime] = None,
        exclude_sim: bool = False,
    ) -> Optional[np.ndarray]:
        """Compatibilidade: array (W, N_FEATURES) ou None quando inválida."""
        ext = self.build_inference_extraction(
            session, node_id, feature_extractor, reference_time, exclude_sim
        )
        return ext.windows if ext.is_valid else None

    # ──────────────────────────────────────────────────────────────────────────
    # API pública de treino
    # ──────────────────────────────────────────────────────────────────────────

    def build_reference_dataset(
        self,
        session: Session,
        node_ids: List[int],
        feature_extractor,
        end_time: Optional[datetime],
        history_hours: float,
        reference_policy,
    ) -> ReferenceDataset:
        """Sequências de referência saudável em ``(end_time − history, end_time]``."""
        if not node_ids:
            raise ValueError("build_reference_dataset: node_ids vazio")
        if end_time is None:
            raise ValueError("build_reference_dataset: end_time explícito é obrigatório")
        start_time = end_time - timedelta(hours=history_hours)
        step = timedelta(minutes=self.window_minutes * self.stride)
        seqs: List[np.ndarray] = []
        meta: List[SequenceExtraction] = []
        exclusions: Counter = Counter()
        for node_id in node_ids:
            seq_start = start_time
            while seq_start + self.span <= end_time:
                ext = self.extract_sequence(
                    session, node_id, feature_extractor, seq_start, reference_policy
                )
                if ext.is_valid:
                    seqs.append(ext.windows)
                    meta.append(ext)
                else:
                    exclusions[ext.reason.split(":")[0]] += 1
                seq_start += step
        X = np.stack(seqs).astype(np.float32) if seqs else None
        logger.info(
            f"SequenceBuilder: {len(seqs)} sequências de referência "
            f"({len(node_ids)} nós, W={self.window_size}, {history_hours:.0f}h até "
            f"{end_time.isoformat()}); exclusões={dict(exclusions)}"
        )
        return ReferenceDataset(X=X, meta=meta, exclusions=exclusions,
                                start=start_time, end=end_time)

    def split_train_val_calib(
        self, meta: List[SequenceExtraction], val_fraction: float = 0.2,
        calib_fraction: float = 0.2,
    ) -> tuple[List[int], List[int], List[int], Optional[datetime], Optional[datetime]]:
        """Três blocos cronológicos GLOBAIS (todos os nós) com purga nos dois cortes.

        Treino ajusta os pesos; validação só escolhe a época (parada antecipada);
        calibração, posterior às duas, define o limiar e τ. Separar a escolha do
        modelo da calibração do limiar evita um limiar otimista; o teste é o que
        vem depois do fim da referência (inferência ao vivo ou cenários).

        Cortes nos quantis (1 − val − calib) e (1 − calib) dos inícios. Treino: fim ≤
        corte 1. Validação: início ≥ corte 1 e fim ≤ corte 2. Calibração: início ≥
        corte 2. Sequências que cruzam um corte ficam fora dos dois blocos.

        Returns:
            (treino, validação, calibração, corte 1, corte 2) — índices em ``meta``.
        """
        if not meta:
            return [], [], [], None, None
        starts = sorted(m.start for m in meta)
        n = len(starts)

        def cut(fraction: float) -> datetime:
            return starts[min(n - 1, max(0, int(math.floor(n * (1.0 - fraction)))))]

        val_cutoff, calib_cutoff = cut(val_fraction + calib_fraction), cut(calib_fraction)
        train_idx = [i for i, m in enumerate(meta) if m.end <= val_cutoff]
        val_idx = [i for i, m in enumerate(meta)
                   if m.start >= val_cutoff and m.end <= calib_cutoff]
        calib_idx = [i for i, m in enumerate(meta) if m.start >= calib_cutoff]
        return train_idx, val_idx, calib_idx, val_cutoff, calib_cutoff

    # ──────────────────────────────────────────────────────────────────────────
    # Métodos internos
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_windows_range(
        self,
        session: Session,
        node_id: int,
        feature_extractor,
        start: datetime,
        n_windows: int,
        exclude_sim: bool,
    ) -> Optional[List[List[float]]]:
        """Compatibilidade: lista (W, N_FEATURES) ou None, com a política única de lacunas.

        Só chamado internamente com n_windows == window_size.
        """
        if exclude_sim:
            from models import Node
            node = session.get(Node, node_id)
            if node and node.sim_mode:
                return None
        ext = self.extract_sequence(session, node_id, feature_extractor, start)
        return ext.windows.tolist() if ext.is_valid else None
