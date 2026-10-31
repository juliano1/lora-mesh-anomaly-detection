"""
Registro de algoritmos online.

Contrato de elegibilidade: só algoritmos com inferência incremental (fit separado de predict/score
em dados *novos*) podem ser registrados. Modelos transdutivos (ex.: LOF do scikit-learn com
novelty=False) são barrados em registro com ValueError explícito.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
import numpy as np


@dataclass(frozen=True)
class AlgorithmSpec:
    id: str
    label: str
    input_kind: str   # "point" | "sequence"
    scaling: str      # "self": o modelo normaliza as próprias entradas (único modo suportado)
    incremental: bool # True = suporta inferência em dados novos; False = transdutivo (bloqueado)
    factory: Callable # (ml_settings) -> AlgorithmRunner


# Estados mínimos de um resultado por algoritmo.
RESULT_STATUSES = ("valid", "warming_up", "insufficient_history", "disabled", "error")


@dataclass
class AlgoResult:
    """Resultado de um algoritmo para um nó/instante.

    ``is_anomaly`` e ``score_norm`` só têm valor quando ``status == "valid"``.
    Nos demais estados ficam nulos: nenhum consumidor pode tratá-los como
    normalidade ou score zero. Um resultado declarado válido com score ausente
    ou não finito é rebaixado a ``error`` na construção.
    """
    is_anomaly: Optional[bool]
    score_norm: Optional[float]   # ∈ [0,1], 1 = mais anômalo (não é probabilidade)
    evidence: dict = field(default_factory=dict)
    status: str = "valid"
    reason: Optional[str] = None
    raw_score: Optional[float] = None
    threshold: Optional[float] = None

    def __post_init__(self):
        if self.status not in RESULT_STATUSES:
            raise ValueError(f"status inválido: {self.status!r}")
        if self.status == "valid" and not self._has_finite_decision():
            self.status = "error"
            self.reason = self.reason or "non_finite_or_missing_score"
        if self.status != "valid":
            # Fora de 'valid' todos os valores numéricos operacionais são nulos;
            # valores não finitos ficam como diagnóstico serializável (texto).
            invalid = {
                name: str(value)
                for name, value in (("raw_score", self.raw_score), ("threshold", self.threshold))
                if value is not None and not math.isfinite(float(value))
            }
            if invalid:
                self.evidence = {**self.evidence, "invalid_values": invalid}
            self.is_anomaly = None
            self.score_norm = None
            self.raw_score = None
            self.threshold = None

    def _has_finite_decision(self) -> bool:
        if self.is_anomaly is None or self.score_norm is None:
            return False
        return all(
            value is None or math.isfinite(float(value))
            for value in (self.score_norm, self.raw_score, self.threshold)
        )

    @property
    def is_valid(self) -> bool:
        return self.status == "valid"

    @classmethod
    def unavailable(cls, status: str, reason: str, evidence: Optional[dict] = None) -> "AlgoResult":
        if status == "valid":
            raise ValueError("unavailable() exige estado diferente de 'valid'")
        return cls(is_anomaly=None, score_norm=None, evidence=evidence or {},
                   status=status, reason=reason)


@dataclass
class DetectionContext:
    session: Any
    node_id: int
    feature_extractor: Any
    features_vec: Optional[np.ndarray]   # shape (n_features,) — vetor pontual, raw; None = indisponível
    network_node_ids: list[int]          # todos os node_ids da rede (para treino)
    network_id: int
    # Corte explícito compartilhado por todos os componentes do ciclo
    reference_time: Optional[datetime] = None
    # Política de referência saudável (ai.reference_policy.ReferencePolicy)
    reference_policy: Any = None
    # Motivo quando features_vec é None
    features_reason: Optional[str] = None


_registry: dict[str, AlgorithmSpec] = {}


def register(spec: AlgorithmSpec) -> None:
    """
    Registra um algoritmo para uso no detector online.

    Lança ValueError se spec.incremental=False (modelo transdutivo — proibido no online).
    """
    if not spec.incremental:
        raise ValueError(
            f"Algoritmo {spec.id!r} declarado como transdutivo (transductive, incremental=False) "
            "e não pode ser registrado para uso online. "
            "Registre uma variante que pontue dados novos separadamente do ajuste."
        )
    if spec.scaling != "self":
        raise ValueError(f"scaling sem suporte: {spec.scaling!r}. Os runners não aplicam scaler externo.")
    if spec.input_kind not in ("point", "sequence"):
        raise ValueError(f"input_kind inválido: {spec.input_kind!r}. Use 'point' ou 'sequence'.")
    _registry[spec.id] = spec


def get(id: str) -> AlgorithmSpec:
    if id not in _registry:
        raise KeyError(f"Algoritmo {id!r} não encontrado no registro online.")
    return _registry[id]


def available() -> list[AlgorithmSpec]:
    return list(_registry.values())
