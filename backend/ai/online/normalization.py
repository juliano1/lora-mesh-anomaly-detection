"""
Normalização de score por algoritmo.

ScoreNormalizer: min-max calibrado por percentis. O resultado em [0, 1] é uma posição
relativa à referência de treino, não uma probabilidade.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ai.numeric_guards import require_finite


class ScoreNormalizer:
    """
    Mapeia scores brutos de qualquer algoritmo para [0, 1] onde 1 = mais anômalo.

    invert=True:  menor raw = mais anômalo (IF — convenção sklearn lower-is-worse).
    invert=False: maior raw = mais anômalo (LSTM-AE — erro de reconstrução).

    Calibração: fit() sobre scores do conjunto de treino calcula percentis p_low/p_high
    para min-max robusto (ignora outliers extremos).
    """

    def __init__(self, p_low: int = 5, p_high: int = 95, invert: bool = True):
        self.p_low = p_low
        self.p_high = p_high
        self.invert = invert
        self._p_low_val: Optional[float] = None
        self._p_high_val: Optional[float] = None

    def fit(self, scores: np.ndarray) -> None:
        """Calibra os percentis sobre os scores de treino (finitos e não vazios)."""
        arr = np.asarray(scores, dtype=float).ravel()
        if arr.size == 0:
            raise ValueError("ScoreNormalizer.fit: scores de calibração vazios")
        require_finite(arr, "scores de calibração")
        self._p_low_val = float(np.percentile(arr, self.p_low))
        self._p_high_val = float(np.percentile(arr, self.p_high))

    def transform(self, raw: float) -> float:
        """Mapeia um score bruto para [0, 1]. Score não finito → NaN (o chamador trata como erro)."""
        if not math.isfinite(float(raw)):
            return float("nan")
        if not self.is_calibrated:
            # Fallback pré-calibração: não é normalizador calibrado; os runners só pontuam após o treino.
            if self.invert:
                return float(1.0 / (1.0 + np.exp(raw * 10)))
            return float(np.clip(raw, 0.0, 1.0))

        if self.invert:
            val = -raw
            lo = -self._p_high_val  # type: ignore[operator]
            hi = -self._p_low_val   # type: ignore[operator]
        else:
            val = raw
            lo = self._p_low_val    # type: ignore[assignment]
            hi = self._p_high_val   # type: ignore[assignment]

        denom = max(hi - lo, 1e-8)  # type: ignore[operator]
        return float(np.clip((val - lo) / denom, 0.0, 1.0))

    @property
    def is_degenerate(self) -> bool:
        """True quando p_low == p_high: o mapeamento vira degrau 0/1."""
        if not self.is_calibrated:
            return False
        return (self._p_high_val - self._p_low_val) <= 1e-12  # type: ignore[operator]

    @property
    def is_calibrated(self) -> bool:
        return self._p_low_val is not None
