"""
Combiner genérico: single / cascade / shadow / fusion.

Opera sobre AlgoResult normalizado; agnóstico de qual algoritmo produziu cada resultado.
evaluate() decide apenas sobre resultados válidos e finitos; indisponibilidade não vira normalidade.
Grava proveniência estruturada (detector_combo, detector_primary, detector_secondary)
em cada dict retornado — consumido por _save_and_notify em anomaly_detector.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ai.online.registry import AlgoResult


COMBOS = ("single", "cascade", "shadow", "fusion")
COMBO_VALID = "valid"
COMBO_UNAVAILABLE = "unavailable"
COMBO_NOT_APPLICABLE = "not_applicable"


@dataclass
class ComboOutcome:
    """Decisão da combinação para um nó/instante.

    ``not_applicable``: shadow não combina — cada componente é registrado à parte.
    ``unavailable``: algum requisito do modo falta; nunca equivale a normalidade.
    """
    status: str
    is_anomaly: Optional[bool] = None
    score: Optional[float] = None
    reason: Optional[str] = None


def _unavailable(reason: str) -> ComboOutcome:
    return ComboOutcome(COMBO_UNAVAILABLE, reason=reason)


def _component_reason(role: str, result: Optional[AlgoResult]) -> str:
    if result is None:
        return f"{role}_missing"
    return f"{role}_{result.status}"


def _is_valid(result: Optional[AlgoResult]) -> bool:
    return result is not None and result.is_valid


def _component_summary(algorithm: Optional[str], result: AlgoResult) -> dict:
    """Decisão individual preservada nos modos combinados (auditoria)."""
    return {
        "algorithm": algorithm,
        "is_anomaly": result.is_anomaly,
        "score_norm": None if result.score_norm is None else round(result.score_norm, 4),
        "raw_score": None if result.raw_score is None else round(result.raw_score, 6),
        "threshold": None if result.threshold is None else round(result.threshold, 6),
    }


def _provenance_dict(
    combo: str,
    primary_id: str,
    secondary_id: Optional[str],
    score_norm: float,
    evidence: dict,
    detector_algorithm: str,
) -> dict:
    return {
        "is_anomaly": True,
        "score_norm": round(score_norm, 4),
        "detector_combo": combo,
        "detector_primary": primary_id,
        "detector_secondary": secondary_id,
        "detector_algorithm": detector_algorithm,
        "evidence": evidence,
    }


class Combiner:
    """
    Combina os resultados de até dois runners segundo o modo configurado.

    combine() retorna lista de dicts com proveniência pronta para _save_and_notify.
    Lista vazia = nenhuma anomalia gerada pela combinação.
    """

    def __init__(self, combo: str):
        if combo not in COMBOS:
            raise ValueError(f"Combo desconhecido: {combo!r}")
        self.combo = combo

    def evaluate(
        self,
        primary: Optional[AlgoResult],
        secondary: Optional[AlgoResult],
        alpha: float = 0.5,
        tau: Optional[float] = None,
        fusion_scores: Optional[tuple] = None,
    ) -> ComboOutcome:
        """Decisão da combinação sobre entradas válidas (sem fallback implícito).

        ``fusion_scores``: (s_primário, s_secundário) na escala de calibração de τ
        (posto na calibração); sem ele, a fusão usa ``score_norm``.
        """
        if self.combo == "single":
            if not _is_valid(primary):
                return _unavailable(_component_reason("primary", primary))
            return ComboOutcome(COMBO_VALID, primary.is_anomaly, primary.score_norm)
        if self.combo == "shadow":
            return ComboOutcome(COMBO_NOT_APPLICABLE, reason="shadow_registra_componentes")
        for role, result in (("primary", primary), ("secondary", secondary)):
            if not _is_valid(result):
                return _unavailable(_component_reason(role, result))
        if self.combo == "cascade":
            # Confirmação AND das decisões individuais; não há score contínuo da cascata.
            is_anomaly = bool(primary.is_anomaly and secondary.is_anomaly)
            score = max(primary.score_norm, secondary.score_norm) if is_anomaly else None
            return ComboOutcome(COMBO_VALID, is_anomaly, score)
        # fusion
        if tau is None or not math.isfinite(float(tau)):
            return _unavailable("tau_unavailable")
        if alpha is None or not math.isfinite(float(alpha)) or not 0.0 <= alpha <= 1.0:
            return _unavailable("invalid_alpha")
        s_p, s_s = (fusion_scores if fusion_scores is not None
                    else (primary.score_norm, secondary.score_norm))
        s_final = alpha * s_p + (1.0 - alpha) * s_s
        if not math.isfinite(s_final):
            return _unavailable("non_finite_fusion_score")
        return ComboOutcome(COMBO_VALID, s_final > tau, s_final)

    def combine(
        self,
        primary: Optional[AlgoResult],
        secondary: Optional[AlgoResult],
        primary_id: str,
        secondary_id: Optional[str],
        alpha: float = 0.5,
        tau: Optional[float] = None,
        fusion_scores: Optional[tuple] = None,
    ) -> list[dict]:
        """Lista de anomalias com proveniência para _save_and_notify (vazia = nada a emitir)."""
        if self.combo == "shadow":
            return self._shadow(primary, secondary, primary_id, secondary_id)
        outcome = self.evaluate(primary, secondary, alpha, tau, fusion_scores)
        if outcome.status != COMBO_VALID or not outcome.is_anomaly:
            return []
        if self.combo == "single":
            algo_str = f"single:{_short(primary_id)}"
            return [_provenance_dict("single", primary_id, None, primary.score_norm,
                                     primary.evidence, algo_str)]
        if self.combo == "cascade":
            algo_str = f"cascade:{_short(primary_id)}>{_short(secondary_id or '')}"
            evidence = {**primary.evidence, "secondary": secondary.evidence,
                        "components": _components(primary, secondary, primary_id, secondary_id)}
            return [_provenance_dict("cascade", primary_id, secondary_id, outcome.score,
                                     evidence, algo_str)]
        algo_str = f"fusion:{_short(primary_id)}+{_short(secondary_id or '')}"
        evidence = {
            **primary.evidence,
            "secondary": secondary.evidence,
            "components": _components(primary, secondary, primary_id, secondary_id),
            "s_fusion": round(outcome.score, 4),
            "alpha": alpha,
            "tau": tau,
            "fusion_scale": "calibration_rank" if fusion_scores is not None else "score_norm",
        }
        if fusion_scores is not None:
            evidence["fusion_inputs"] = {"primary": round(float(fusion_scores[0]), 4),
                                         "secondary": round(float(fusion_scores[1]), 4)}
        return [_provenance_dict("fusion", primary_id, secondary_id, outcome.score,
                                 evidence, algo_str)]

    def _shadow(
        self,
        primary: Optional[AlgoResult],
        secondary: Optional[AlgoResult],
        primary_id: str,
        secondary_id: Optional[str],
    ) -> list[dict]:
        results = []
        if _is_valid(primary) and primary.is_anomaly:
            results.append(_provenance_dict(
                "shadow", primary_id, None, primary.score_norm, primary.evidence,
                f"shadow:{_short(primary_id)}",
            ))
        if _is_valid(secondary) and secondary.is_anomaly and secondary_id is not None:
            results.append(_provenance_dict(
                "shadow", secondary_id, None, secondary.score_norm, secondary.evidence,
                f"shadow:{_short(secondary_id)}",
            ))
        return results


# ─── codebook de IDs curtos para detector_algorithm derivado ────────────────

_SHORT = {
    "isolation_forest": "if",
    "lstm_ae": "lstm",
}


def _components(primary: AlgoResult, secondary: AlgoResult,
                primary_id: str, secondary_id: Optional[str]) -> dict:
    return {"primary": _component_summary(primary_id, primary),
            "secondary": _component_summary(secondary_id, secondary)}


def _short(id: str) -> str:
    return _SHORT.get(id, id[:6])
