"""
Catálogo de tipos de anomalias detectáveis por IA.

Catálogo com 7 tipos (A1-A7). Nomenclatura agnóstica ao algoritmo —
os tipos descrevem a anomalia, não o detector.
"""

from dataclasses import dataclass
from ai import anomaly_definitions as definitions
# Reexportações preservam os imports públicos anteriores.
from ai.anomaly_definitions import AnomalyCategory, AnomalySeverity


@dataclass
class AnomalyTypeDef:
    """Definição de um tipo de anomalia."""
    code: str
    category: AnomalyCategory
    severity: AnomalySeverity
    label: str
    description: str
    feature_keys: list[str]
    criteria: str = ""


# ==========================================
# Catálogo de Anomalias (agnóstico ao algoritmo)
# ==========================================

def build_catalog() -> dict[str, AnomalyTypeDef]:
    """Metadados e expressões vêm das mesmas definições do avaliador."""
    return {
        code: AnomalyTypeDef(
            code=code, category=AnomalyCategory(rule.category),
            severity=AnomalySeverity(rule.severity), label=rule.label,
            description=rule.description, feature_keys=rule.feature_keys,
            criteria=rule.criteria,
        )
        for code, rule in definitions.ANOMALY_DEFINITIONS.items()
    }


ANOMALY_CATALOG = build_catalog()


def get_anomaly_def(anomaly_type: str) -> AnomalyTypeDef | None:
    """Retorna a definição de um tipo de anomalia pelo código."""
    return ANOMALY_CATALOG.get(anomaly_type)


def get_all_types() -> list[str]:
    """Retorna lista de todos os tipos detectáveis."""
    return list(ANOMALY_CATALOG.keys())


def get_catalog_info() -> list[dict]:
    """Retorna catálogo completo como lista de dicts (para API)."""
    return [
        {
            "code": t.code,
            "category": t.category.value,
            "severity": t.severity.value,
            "label": t.label,
            "description": t.description,
            "criteria": t.criteria,
            "feature_keys": list(t.feature_keys),
        }
        for t in ANOMALY_CATALOG.values()
    ]
