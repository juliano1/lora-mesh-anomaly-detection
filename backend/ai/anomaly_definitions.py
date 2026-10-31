"""Fonte única, pura e imutável dos critérios e unidades A1–A7.

Comparações em unidades brutas; AND dentro do ramo e OR entre ramos.
Os cortes adotados são escolhas de projeto e não são limites físicos universais.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from numbers import Real
from types import MappingProxyType

RULES_VERSION = "spec065-rules-v1"
DEFINITIONS_VERSION = "spec066-definitions-v3"  # v3: A6 é do enlace (radio), sem nó de origem
# Marca do formato dos metadados de janela gravados em online_inference.features_json
# (chaves "_*"). Valor preservado para comparar registros antigos.
REGISTRO_VERSION = "spec069-registro-v1"
CLASSIFICATION_ORDER = ("A5", "A1", "A2", "A3", "A4", "A6", "A7")
# Contrato explícito do vetor dos modelos, independente da ordem dos metadados.
FEATURE_ORDER = (
    "rssi_mean", "rssi_std", "snr_mean", "snr_std", "pdr_mean", "battery_delta",
    "packet_loss_rate", "rssi_snr_divergence", "beacon_interval_std",
    "route_change_rate", "link_asymmetry",
)


class AnomalyCategory(str, Enum):
    RADIO = "radio"
    HARDWARE = "hardware"
    TOPOLOGY = "topology"
    SECURITY = "security"
    STATISTICAL = "statistical"


class AnomalySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} deve ser texto não vazio")


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} deve ser número real finito")
    return float(value)


@dataclass(frozen=True)
class FeatureDefinition:
    label: str
    unit: str
    description: str
    display_unit: str
    display_scale: float = 1.0

    def __post_init__(self):
        for name in ("label", "unit", "description", "display_unit"):
            _require_text(getattr(self, name), name)
        scale = _finite_number(self.display_scale, "display_scale")
        if scale <= 0:
            raise ValueError("display_scale deve ser positiva")
        object.__setattr__(self, "display_scale", scale)

    def format_value(self, value: float, digits: int | None = 3) -> str:
        """Apresentação com até ``digits`` casas; ``None`` preserva a precisão total."""
        number = float(value) * self.display_scale
        if digits is not None:
            number = round(number, digits) + 0.0  # + 0.0 elimina "-0"
        text = str(int(number)) if number.is_integer() else (
            str(number) if digits is None else f"{number:.{digits}f}".rstrip("0"))
        gap = "" if self.display_unit in ("%", "/10") else " "
        return f"{text}{gap}{self.display_unit}"


# Metadados não determinam a posição das entradas no vetor IF/LSTM.
FEATURE_DEFINITIONS = MappingProxyType({
    "rssi_mean": FeatureDefinition("RSSI médio", "dBm",
        "Média do RSSI dos registros de vizinhos do nó na janela.", "dBm"),
    "rssi_std": FeatureDefinition("Desvio RSSI por enlace", "dB",
        "Média aritmética, sem ponderação, dos desvios padrão populacionais (ddof=0) do RSSI por enlace elegível (≥2 pares RSSI/SNR finitos); nulo sem enlace elegível. Não é o desvio agregado de todos os vizinhos.", "dB"),
    "snr_mean": FeatureDefinition("SNR médio", "dB",
        "Média do SNR dos registros de vizinhos na janela.", "dB"),
    "snr_std": FeatureDefinition("Desvio SNR por enlace", "dB",
        "Média aritmética, sem ponderação, dos desvios padrão populacionais (ddof=0) do SNR por enlace elegível; nulo sem enlace elegível.", "dB"),
    "pdr_mean": FeatureDefinition("Índice de qualidade de sinal", "índice 0–10",
        "Média de link_quality, índice heurístico de qualidade de sinal calculado de RSSI/SNR pelo firmware; default 5 sem observação. Indicador heurístico, não é PDR físico medido.", "/10"),
    "battery_delta": FeatureDefinition("Δ do indicador de bateria na janela", "pontos percentuais",
        "Último − primeiro nível válido do indicador de bateria na janela; default 0 com menos de 2 leituras. Não é mV nem taxa por hora.", "p.p."),
    "packet_loss_rate": FeatureDefinition("Perda inferida por sequência", "fração 0–1",
        "Lacunas de sequência da telemetria recebida até o backend na época atual do contador; default 0 com menos de 2 contadores distintos (ver unavailable_features).", "%", 100.0),
    "rssi_snr_divergence": FeatureDefinition("Diferença entre variabilidades SNR/RSSI", "dB",
        "|média do desvio de SNR por enlace − média do desvio de RSSI por enlace|; diferença de variabilidades, não correlação.", "dB"),
    "beacon_interval_std": FeatureDefinition("Desvio dos intervalos entre telemetrias", "s",
        "Desvio padrão dos intervalos entre telemetrias; default 0 com menos de 3.", "s"),
    "route_change_rate": FeatureDefinition("Trocas de parent ÷ registros de parent", "trocas/registro",
        "Trocas do parent reportado ÷ registros de parent na janela.", "%", 100.0),
    "link_asymmetry": FeatureDefinition("Assimetria média por enlace", "dB",
        "Média de |RSSI local − RSSI reverso| por enlace com registro reverso; default 0 sem enlace reverso.", "dB"),
})


@dataclass(frozen=True)
class Comparison:
    feature: str
    operator: str
    threshold: float

    def __post_init__(self):
        if self.feature not in FEATURE_DEFINITIONS:
            raise ValueError(f"Feature desconhecida: {self.feature}")
        if self.operator not in ("<", ">"):
            raise ValueError("Comparação deve ter operador estrito")
        object.__setattr__(self, "threshold", _finite_number(self.threshold, "threshold"))

    def expression(self) -> str:
        feature = FEATURE_DEFINITIONS[self.feature]
        return f"{feature.label} {self.operator} {feature.format_value(self.threshold)}"

    def to_dict(self) -> dict:
        feature = FEATURE_DEFINITIONS[self.feature]
        return dict(feature=self.feature, operator=self.operator, threshold=self.threshold,
                    unit=feature.unit, display_unit=feature.display_unit,
                    display_scale=feature.display_scale)


@dataclass(frozen=True)
class RuleBranch:
    name: str
    comparisons: tuple[Comparison, ...]

    def __post_init__(self):
        _require_text(self.name, "name")
        if (not isinstance(self.comparisons, tuple) or not self.comparisons
                or not all(isinstance(t, Comparison) for t in self.comparisons)):
            raise ValueError("Ramo deve conter tupla não vazia de comparações")


@dataclass(frozen=True)
class AnomalyDefinition:
    code: str
    category: AnomalyCategory
    severity: AnomalySeverity
    label: str
    description: str
    branches: tuple[RuleBranch, ...] = ()

    def __post_init__(self):
        for name in ("code", "label", "description"):
            _require_text(getattr(self, name), name)
        object.__setattr__(self, "category", AnomalyCategory(self.category))
        object.__setattr__(self, "severity", AnomalySeverity(self.severity))
        if (not isinstance(self.branches, tuple)
                or not all(isinstance(b, RuleBranch) for b in self.branches)):
            raise ValueError("branches deve ser tupla de ramos")
        if len({b.name for b in self.branches}) != len(self.branches):
            raise ValueError("Nomes de ramos devem ser únicos")
        if bool(self.branches) == (self.code == "AX"):
            raise ValueError("Somente AX deve ser fallback sem comparações")

    @property
    def feature_keys(self) -> list[str]:
        return list(dict.fromkeys(t.feature for b in self.branches for t in b.comparisons))

    @property
    def criteria(self) -> str:
        if not self.branches:
            return "Fallback: anômalo pelo detector, sem regra A1–A7 correspondente"
        expressions = [" e ".join(t.expression() for t in b.comparisons) for b in self.branches]
        if len(expressions) == 1:
            return expressions[0]
        return " ou ".join(f"({expr})" for expr in expressions)

    def to_dict(self) -> dict:
        return dict(code=self.code, label=self.label, description=self.description,
                    category=self.category.value, severity=self.severity.value, expression=self.criteria,
                    branches={b.name: [t.to_dict() for t in b.comparisons] for b in self.branches})


ANOMALY_DEFINITIONS = MappingProxyType({
    "A1": AnomalyDefinition("A1", "radio", "medium", "Degradação de sinal",
        "RSSI baixo ou variabilidade temporal alta por enlace com SNR baixo.", (
            RuleBranch("level", (Comparison("rssi_mean", "<", -100.0),)),
            RuleBranch("variability", (Comparison("rssi_std", ">", 10.0), Comparison("snr_mean", "<", 0.0))),
        )),
    "A2": AnomalyDefinition("A2", "radio", "high", "Perda de telemetria inferida",
        "Lacunas de sequência na telemetria recebida até o backend (fim a fim; não localiza o enlace); somente com estimativa disponível.", (
            RuleBranch("loss", (Comparison("packet_loss_rate", ">", 0.30),)),
        )),
    "A3": AnomalyDefinition("A3", "hardware", "medium", "Queda do indicador de bateria",
        "Queda do indicador percentual de bateria entre leituras válidas da janela; não é taxa por hora nem tensão medida.", (
            RuleBranch("battery", (Comparison("battery_delta", "<", -20.0),)),
        )),
    "A4": AnomalyDefinition("A4", "topology", "high", "Possível instabilidade do enlace",
        "Alta variabilidade temporal de RSSI/SNR por enlace com índice de qualidade baixo.", (
            RuleBranch("instability", (Comparison("rssi_std", ">", 8.0), Comparison("snr_std", ">", 5.0), Comparison("pdr_mean", "<", 4.0))),
        )),
    "A5": AnomalyDefinition("A5", "security", "high", "Suspeita de interferência",
        "Padrão compatível com interferência; não comprova jamming, intenção ou causa física.", (
            RuleBranch("divergence", (Comparison("rssi_snr_divergence", ">", 5.0), Comparison("snr_mean", "<", 3.0), Comparison("pdr_mean", "<", 3.0))),
            RuleBranch("snr_variability", (Comparison("rssi_mean", ">", -100.0), Comparison("snr_std", ">", 5.0), Comparison("pdr_mean", "<", 2.0))),
        )),
    "A6": AnomalyDefinition("A6", "radio", "medium", "Assimetria bidirecional elevada",
        "Diferença de RSSI entre os sentidos de um enlace, registrada no nó que a observou; a origem pode estar em qualquer um dos dois nós. Potência, calibração e instantes de observação também explicam a diferença; não comprova defeito de TX.", (
            RuleBranch("asymmetry", (Comparison("link_asymmetry", ">", 15.0),)),
        )),
    "A7": AnomalyDefinition("A7", "topology", "high", "Alternância do parent reportado",
        "Trocas frequentes do parent reportado pelo nó na janela; não confirma mudança da rota efetiva.", (
            RuleBranch("parent_changes", (Comparison("route_change_rate", ">", 0.30),)),
        )),
    "AX": AnomalyDefinition("AX", "statistical", "low", "Outlier estatístico não classificado",
        "Ponto fora do padrão detectado pelo modelo ML, sem critério A1–A7 acionado; avaliar o conjunto e a disponibilidade das features."),
})
CODES = tuple(code for code, rule in ANOMALY_DEFINITIONS.items() if rule.branches)


def _strict_cuts(key: str) -> set[float]:
    return {term.threshold for rule in ANOMALY_DEFINITIONS.values()
            for branch in rule.branches for term in branch.comparisons if term.feature == key}


def present_feature_value(key: str, value: float) -> str:
    """Apresentação única do caso atual, dos exemplos e das comparações.

    Escala de exibição com até três casas; amplia a precisão só quando o texto
    igualaria um corte estrito diferente do valor, sem esconder o lado decisivo.
    """
    feature = FEATURE_DEFINITIONS[key]
    cuts = [cut for cut in _strict_cuts(key) if cut != value]
    for digits in range(3, 16):
        text = feature.format_value(value, digits)
        if all(feature.format_value(cut, digits) != text for cut in cuts):
            return text
    return feature.format_value(value, digits=None)


def is_at_cut(key: str, value: float) -> bool:
    """Valor sobre um corte estrito: se veio arredondado, o lado é indeterminado."""
    return float(value) in _strict_cuts(key)


def validate_definitions() -> None:
    """Falha na inicialização se o registro perder integridade ou compatibilidade."""
    if len(set(FEATURE_ORDER)) != len(FEATURE_ORDER) or set(FEATURE_ORDER) != set(FEATURE_DEFINITIONS):
        raise ValueError("FEATURE_ORDER deve enumerar cada feature exatamente uma vez")
    active = {code for code, rule in ANOMALY_DEFINITIONS.items() if rule.branches}
    if (len(set(CLASSIFICATION_ORDER)) != len(CLASSIFICATION_ORDER)
            or set(CLASSIFICATION_ORDER) != active):
        raise ValueError("CLASSIFICATION_ORDER deve enumerar cada critério exatamente uma vez")
    if "AX" not in ANOMALY_DEFINITIONS:
        raise ValueError("Fallback AX ausente")
    for code, rule in ANOMALY_DEFINITIONS.items():
        if code != rule.code:
            raise ValueError("Chave do registro difere do código do critério")


_FINGERPRINT_CACHE: tuple | None = None


def definition_fingerprint() -> str:
    """SHA-256 canônico do contrato; não depende da ordem dos dicionários.

    Calculado uma vez por objeto de contrato: os registros são imutáveis, e
    substituí-los (inclusive em testes) troca a identidade e invalida o cache.
    """
    global _FINGERPRINT_CACHE
    key = (RULES_VERSION, DEFINITIONS_VERSION, FEATURE_ORDER, CLASSIFICATION_ORDER,
           FEATURE_DEFINITIONS, ANOMALY_DEFINITIONS)
    cached = _FINGERPRINT_CACHE
    if cached is not None and all(a is b for a, b in zip(cached[0], key)):
        return cached[1]
    fingerprint = _compute_fingerprint()
    _FINGERPRINT_CACHE = (key, fingerprint)
    return fingerprint


def _compute_fingerprint() -> str:
    contract = dict(
        rules_version=RULES_VERSION, definitions_version=DEFINITIONS_VERSION,
        feature_order=FEATURE_ORDER, classification_order=CLASSIFICATION_ORDER,
        features={key: asdict(value) for key, value in FEATURE_DEFINITIONS.items()},
        anomalies={key: value.to_dict() for key, value in ANOMALY_DEFINITIONS.items()},
    )
    encoded = json.dumps(contract, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_rule_reference() -> str:
    """Referência textual derivada das mesmas comparações executadas pelo backend."""
    return "\n".join(
        f"- {r.code}: {r.label}. {r.criteria}{'' if r.criteria.endswith('.') else '.'} {r.description}"
        for r in ANOMALY_DEFINITIONS.values())


validate_definitions()
