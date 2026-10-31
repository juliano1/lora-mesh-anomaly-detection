"""Critérios A1–A7 em lógica de três estados.

Módulo puro: avalia critérios e arquiva comparações, sem confiança ou I/O. Cada código
resulta em ``hit``, ``miss`` ou ``not_evaluable``. Um termo é indeterminado
quando a feature é ``None``, não finita ou consta em ``features["_unavailable"]``.
AND/OR seguem a lógica de Kleene: ausência nunca vira ``miss`` automaticamente.

Os cortes vêm de ``anomaly_definitions`` e são candidatos operacionais;
não são limites físicos universais validados pela literatura.
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from ai import anomaly_definitions as definitions

RULES_VERSION = definitions.RULES_VERSION
HIT, MISS, NOT_EVALUABLE = "hit", "miss", "not_evaluable"
CODES = definitions.CODES


@dataclass
class CriterionResult:
    state: str
    reasons: list = field(default_factory=list)
    branches: dict = field(default_factory=dict)
    comparisons: dict = field(default_factory=dict)


@dataclass
class RulesEvaluation:
    criteria: dict
    context_mode: str | None
    context_reason: str | None
    unavailable: dict
    battery_window: dict | None
    rules_version: str = RULES_VERSION
    definition_snapshot: dict = field(default_factory=dict)
    definitions_version: str = definitions.DEFINITIONS_VERSION
    definitions_fingerprint: str = ""

    def hits(self, order=CODES) -> list:
        return [c for c in order if self.criteria[c].state == HIT]

    def not_evaluable(self) -> list:
        return [c for c in CODES if self.criteria[c].state == NOT_EVALUABLE]

    def to_evidence(self, *, include_details: bool = True) -> dict:
        payload = {
            "rules_version": self.rules_version,
            "context_mode": self.context_mode,
            "context_reason": self.context_reason,
            "criteria": {c: {"state": r.state, "reasons": list(r.reasons)}
                         for c, r in self.criteria.items()},
            "unavailable_features": dict(self.unavailable),
            "battery_window": self.battery_window,
            "definitions_version": self.definitions_version,
            "definitions_fingerprint": self.definitions_fingerprint,
        }
        if include_details:
            payload.update(definitions=self.definition_snapshot,
                           comparisons={c: r.comparisons for c, r in self.criteria.items()})
        # Cada exportação pertence ao chamador; não deve alterar a avaliação
        # nem contaminar snapshots produzidos posteriormente.
        return deepcopy(payload)


class _Terms:
    """Avalia comparações registrando o motivo dos termos indeterminados."""

    def __init__(self, features: dict):
        self.features = features
        self.unavailable = dict(features.get("_unavailable") or {})
        self.reasons: list = []

    def value(self, name: str):
        if name in self.unavailable:
            return None, self.unavailable[name]
        raw = self.features.get(name)
        if raw is None:
            return None, "missing"
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None, "non_numeric"
        if not math.isfinite(v):
            return None, "non_finite"
        return v, None

    def compare(self, term: definitions.Comparison) -> dict:
        v, why = self.value(term.feature)
        feature = definitions.FEATURE_DEFINITIONS[term.feature]
        if v is None:
            reason = f"{term.feature}:{why}"
            if reason not in self.reasons:
                self.reasons.append(reason)
            state = NOT_EVALUABLE
            value_text = f"n/d ({why})"
        else:
            matched = v < term.threshold if term.operator == "<" else v > term.threshold
            state = HIT if matched else MISS
            # Apresentação compartilhada; não esconde o lado do corte.
            value_text = definitions.present_feature_value(term.feature, v)
        return dict(**term.to_dict(), value=v, state=state, reason=why,
                    text=f"{term.feature}: {value_text} {term.operator} "
                         f"{feature.format_value(term.threshold)} → {state}")


def _and(*states: str) -> str:
    if MISS in states:
        return MISS
    if NOT_EVALUABLE in states:
        return NOT_EVALUABLE
    return HIT


def _or(*states: str) -> str:
    if HIT in states:
        return HIT
    if NOT_EVALUABLE in states:
        return NOT_EVALUABLE
    return MISS


def _criterion(features: dict, rule: definitions.AnomalyDefinition) -> CriterionResult:
    terms = _Terms(features)
    comparisons = {b.name: [terms.compare(t) for t in b.comparisons] for b in rule.branches}
    branches = {name: _and(*(term["state"] for term in values))
                for name, values in comparisons.items()}
    state = _or(*branches.values())
    return CriterionResult(state, terms.reasons if state == NOT_EVALUABLE else [], branches, comparisons)


def _battery_window(features: dict) -> dict | None:
    first, last = features.get("_battery_first_at"), features.get("_battery_last_at")
    if first is None or last is None:
        return None
    return {"first_at": first.isoformat(), "last_at": last.isoformat(),
            "span_seconds": (last - first).total_seconds()}


def evaluate_criteria(features: dict) -> RulesEvaluation:
    """Estados A1–A7 de uma janela (valores brutos, não normalizados).

    Nesta versão A1 usa sempre o corte absoluto de RSSI médio.
    ``rssi_residual_mean`` é ignorado em qualquer modo (inclusive se vier no
    dict), porque não há caminho documentado de aquisição de distância
    independente; o modo efetivo e o motivo seguem para a evidência.
    """

    active = {code: rule for code, rule in definitions.ANOMALY_DEFINITIONS.items() if rule.branches}
    criteria = {code: _criterion(features, rule) for code, rule in active.items()}
    return RulesEvaluation(
        criteria=criteria,
        context_mode=features.get("_classification_mode"),
        context_reason=features.get("_context_reason"),
        unavailable=dict(features.get("_unavailable") or {}),
        battery_window=_battery_window(features),
        definition_snapshot={code: rule.to_dict() for code, rule in active.items()},
        definitions_version=definitions.DEFINITIONS_VERSION,
        definitions_fingerprint=definitions.definition_fingerprint(),
    )
