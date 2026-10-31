"""
Guardas numéricas compartilhadas por treino e inferência.

MIN_FEATURE_STD: desvio padrão abaixo do qual a feature é tratada como constante
e recebe escala 1 (mesma convenção do StandardScaler do scikit-learn). Todas as
11 features têm resolução muito maior que 1e-6 (dBm, dB, %, s, razões), então o
limiar só alcança variância numericamente nula. Somar um epsilon ao std
(`std + 1e-8`) amplificaria uma variação de 0,01 para 1.000.000.
"""
from __future__ import annotations

import numpy as np

MIN_FEATURE_STD = 1e-6


def require_finite(values, name: str) -> None:
    """Lança ValueError identificável se houver NaN/Inf."""
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contém NaN/Inf")


def safe_feature_scale(std, min_std: float = MIN_FEATURE_STD) -> np.ndarray:
    """Escala por feature: o próprio std, ou 1 quando a variância é ~0."""
    arr = np.asarray(std, dtype=np.float64)
    require_finite(arr, "desvio padrão das features")
    return np.where(arr < min_std, 1.0, arr)
