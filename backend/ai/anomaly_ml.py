"""
Detecção de anomalias por IA com interface abstrata.

Arquitetura:
- BaseAnomalyModel: interface abstrata dos modelos pontuais
- IsolationForestModel: Isolation Forest (scikit-learn), usado pelo PointRunner do
  pipeline online (ai.online); o laço de detecção não tem outro IF
- classify_anomaly_type: atribuição A1–A7/AX depois da decisão do modelo

Tipos A1–A7 (ver anomaly_definitions.py): atribuídos depois da detecção, não detectados
pelas regras. Ex.: A6 — assimetria bidirecional elevada (link_asymmetry; não comprova
defeito de TX); A7 — alternância do parent reportado (route_change_rate).
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
import logging
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ai.anomaly_types import get_anomaly_def
from ai.anomaly_definitions import CLASSIFICATION_ORDER, FEATURE_DEFINITIONS
from ai.numeric_guards import require_finite

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Interface Abstrata — trocar algoritmo sem alterar o restante
# ═══════════════════════════════════════════════════════════════════

class BaseAnomalyModel(ABC):
    """
    Interface abstrata para modelos de detecção de anomalias.
    
    Para trocar o algoritmo pontual:
    1. Crie uma classe que herda BaseAnomalyModel
    2. Implemente train(), predict(), score() e get_info()
    3. Instancie-a na factory do AlgorithmSpec correspondente
       (``_register_defaults`` em ai.online.pipeline)
    """
    
    @abstractmethod
    def train(self, X: np.ndarray) -> bool:
        """Treina o modelo com dados de features. Retorna True se sucesso."""
        ...
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediz anomalias. Retorna array: -1=anômalo, 1=normal."""
        ...
    
    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Retorna scores de anomalia. Quanto menor, mais anômalo."""
        ...
    
    @abstractmethod
    def get_info(self) -> dict:
        """Retorna metadados do modelo (nome, parâmetros, etc)."""
        ...
    
    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Se o modelo já foi treinado."""
        ...


# ═══════════════════════════════════════════════════════════════════
# Implementação: Isolation Forest
# ═══════════════════════════════════════════════════════════════════

class IsolationForestModel(BaseAnomalyModel):
    """
    Implementação com scikit-learn Isolation Forest.

    O desempenho nesta rede está no autoteste (README, "Sensibilidade da detecção"):
    não há comparação publicada que o substitua.
    """
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100,
                 random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self._is_trained = False
        self.training_samples = 0
        self.last_train_time: Optional[datetime] = None
        self.last_train_error: Optional[str] = None

    def train(self, X: np.ndarray) -> bool:
        """Ajusta scaler + floresta em estado candidato.

        Entrada inválida (não 2D, vazia, NaN/Inf) → ValueError. Falha de ajuste ou
        referência degenerada (nenhuma árvore com divisão) → False com motivo em
        ``last_train_error``. Scaler e floresta ativos só mudam após sucesso.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"X deve ser 2D e não vazio, recebido shape {X.shape}")
        require_finite(X, "X")
        self.last_train_error = None
        try:
            # Normalizar features antes do treino (escalas distintas: rssi em dBm,
            # pdr 0-10, beacon_interval_std em s). StandardScaler usa escala 1
            # para variância nula.
            scaler = StandardScaler().fit(X)
            require_finite(scaler.mean_, "scaler.mean_")
            require_finite(scaler.scale_, "scaler.scale_")
            forest = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            ).fit(scaler.transform(X))
            if all(est.tree_.node_count <= 1 for est in forest.estimators_):
                self.last_train_error = "degenerate_reference: nenhuma árvore com divisão"
                logger.warning(f"IsolationForest: {self.last_train_error}")
                return False
            require_finite([forest.offset_], "offset_")
        except Exception as e:
            self.last_train_error = f"{type(e).__name__}: {e}"
            logger.error(f"IsolationForest: falha no treino: {e}")
            return False

        self.scaler = scaler
        self.model = forest
        self._is_trained = True
        self.training_samples = len(X)
        self.last_train_time = datetime.utcnow()
        return True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained or self.model is None or self.scaler is None:
            return np.ones(len(X))  # sem treino, tudo normal
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained or self.model is None or self.scaler is None:
            return np.zeros(len(X))
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)

    def margin(self, X: np.ndarray) -> np.ndarray:
        """Margem bruta ``score_samples − offset_``; ``< 0`` ⇔ ``predict == −1``."""
        if not self._is_trained or self.model is None or self.scaler is None:
            raise RuntimeError("IsolationForestModel não treinado")
        return self.model.decision_function(self.scaler.transform(X))

    @property
    def offset(self) -> Optional[float]:
        return float(self.model.offset_) if self.model is not None else None

    def get_info(self) -> dict:
        return {
            "name": "Isolation Forest",
            "library": "scikit-learn",
            "parameters": {
                "n_estimators": self.n_estimators,
                "contamination": self.contamination,
                "random_state": self.random_state,
            },
            "normalization": "StandardScaler",
            "training_samples": self.training_samples,
            "last_trained": self.last_train_time.isoformat() if self.last_train_time else None,
        }

    @property
    def is_trained(self) -> bool:
        return self._is_trained


# ═══════════════════════════════════════════════════════════════════
# Funções autônomas de classificação — reutilizáveis por qualquer detector
# (Isolation Forest e LSTM-AE)
# ═══════════════════════════════════════════════════════════════════

def build_anomaly(type_code: str, confidence: int, description: str,
                  suggested_action: str, features: dict, score: float,
                  rules: dict | None = None) -> dict:
    """
    Constrói dict de anomalia usando o catálogo.
    Função autônoma — não depende de instância de classe.
    Reutilizável por todos os algoritmos de detecção.
    ``rules`` (``RulesEvaluation.to_evidence()``) entra na evidência.
    """
    type_def = get_anomaly_def(type_code)
    evidence = {
        "features": {k: round(v, 3) if isinstance(v, float) else v
                    for k, v in features.items()
                    if not k.startswith("_")},  # chaves internas excluídas da evidência
        "anomaly_score": round(score, 4),
    }
    if rules is not None:
        evidence["rules"] = rules
    return {
        "anomaly_type": type_code,
        "category": type_def.category.value if type_def else "radio",
        "severity": type_def.severity.value if type_def else "medium",
        "description": description,
        "suggested_action": suggested_action,
        "evidence": evidence,
        "detection_method": "ml",
        "confidence": confidence,
        # Modo efetivo de classificação para auditoria A/B
        "classification_mode_used": features.get("_classification_mode"),
    }


# Ordem de atribuição pós-ML (o pipeline online persiste o primeiro tipo).
_CLASSIFY_ORDER = CLASSIFICATION_ORDER


def _fmt(value, spec: str, unit: str = "") -> str:
    """Valor formatado ou ``n/d`` quando ausente/não finito (apresentação)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/d"
    if v != v or v in (float("inf"), float("-inf")):
        return "n/d"
    return f"{format(v, spec)}{unit}"


def _battery_span(evaluation) -> str:
    window = evaluation.battery_window
    if not window:
        return ""
    minutes = window["span_seconds"] / 60.0
    return (f" entre leituras válidas de {window['first_at'][11:16]} e "
            f"{window['last_at'][11:16]} UTC ({minutes:.0f} min)")


def _feat(f: dict, name: str, spec: str) -> str:
    """Feature indisponível aparece como ``n/d (motivo)``, nunca o valor de apresentação.

    Os modelos não chegam a ver esse valor: ``feature_dict_to_vector`` recusa o vetor
    com feature não observada.
    """
    reason = (f.get("_unavailable") or {}).get(name)
    if reason:
        return f"n/d ({reason})"
    value = f.get(name)
    definition = FEATURE_DEFINITIONS[name]
    gap = "" if definition.display_unit in ("%", "/10") else " "
    unit = gap + definition.display_unit
    try:
        value = float(value) * definition.display_scale
    except (TypeError, ValueError):
        return "n/d"
    return _fmt(value, spec, unit)


def _asymmetric_links(f: dict) -> list:
    """Enlaces com os dois sentidos observados, da maior para a menor diferença."""
    links = [dict(link) for link in (f.get("_asymmetry_links") or [])]
    return sorted(links, key=lambda link: -link["delta_db"])


def _describe(code: str, f: dict, evaluation) -> tuple:
    """Descrição e ação de um critério acionado."""
    idx = _feat(f, "pdr_mean", ".1f")
    if code == "A1":
        return (
            f"Degradação de sinal: RSSI médio={_feat(f, 'rssi_mean', '.0f')}, "
            f"desvio temporal médio por enlace={_feat(f, 'rssi_std', '.1f')}, "
            f"SNR médio={_feat(f, 'snr_mean', '.1f')}",
            "Verificar obstáculos, orientação da antena ou distância entre nós.")
    if code == "A2":
        return (
            f"Perda de telemetria inferida por lacunas de sequência: "
            f"{_feat(f, 'packet_loss_rate', '.1f')} no trecho observável "
            f"(índice de qualidade de sinal={idx}; o índice não é taxa de entrega)",
            "Verificar o enlace e o caminho até o gateway; a medida é fim a fim e "
            "não localiza o enlace afetado.")
    if code == "A3":
        return (
            f"Queda do indicador de bateria: Δ={_feat(f, 'battery_delta', '.1f')}"
            f"{_battery_span(evaluation)}. Não é taxa por hora nem tensão medida",
            "Verificar consumo de energia e saúde da bateria.")
    if code == "A4":
        return (
            f"Possível instabilidade do enlace: desvio temporal médio por enlace "
            f"RSSI={_feat(f, 'rssi_std', '.1f')}, SNR={_feat(f, 'snr_std', '.1f')}, "
            f"índice de qualidade={idx}",
            "Enlace com alta variabilidade temporal. Considerar rota alternativa.")
    if code == "A5":
        return (
            f"Suspeita de interferência: RSSI médio={_feat(f, 'rssi_mean', '.0f')}, "
            f"SNR médio={_feat(f, 'snr_mean', '.1f')}, índice de qualidade={idx}, "
            f"diferença entre variabilidades SNR/RSSI={_feat(f, 'rssi_snr_divergence', '.1f')}",
            "Investigar fontes de interferência na faixa de 915 MHz. O padrão não comprova "
            "jamming, intenção ou reatividade; confirmar com instrumentação independente.")
    if code == "A6":
        links = _asymmetric_links(f)
        where = ""
        if links:
            top = links[0]
            where = (f"; maior diferença no enlace com {top['neighbor']} (ouvido aqui a "
                     f"{top['rssi_heard_here']:.0f} dBm, ouvido pelo vizinho a "
                     f"{top['rssi_heard_by_neighbor']:.0f} dBm)")
        return (
            f"Assimetria bidirecional elevada: {_feat(f, 'link_asymmetry', '.1f')} "
            f"(média de |RSSI local − RSSI reverso| por enlace){where}. O alarme fica no "
            f"nó que observou o enlace; a origem pode estar em qualquer um dos dois nós",
            "Conferir os dois nós do enlace: potência, antenas e calibração. Diferenças de "
            "potência, calibração e instantes de observação também explicam assimetria; "
            "defeito de transmissor exige evidência externa.")
    return (
        f"Alternância do parent reportado: {_feat(f, 'route_change_rate', '.0f')} "
        f"de trocas por registro de parent na janela",
        "Indicador reportado pelo nó; não confirma mudança da rota efetivamente usada. "
        "Verificar a qualidade dos enlaces candidatos a parent.")


def classify_anomaly_type(features: dict, score: float) -> List[Dict]:
    """
    Classifica o tipo de anomalia baseado no padrão das features.

    Função autônoma e agnóstica ao algoritmo: o algoritmo decide QUE é anômalo;
    esta função decide POR QUÊ, sempre após a decisão do modelo. Os critérios
    vêm de :func:`ai.anomaly_rules.evaluate_criteria`, em três
    estados; só ``hit`` gera tipo. Sem nenhum ``hit``, mantém o contrato AX
    com os critérios ``not_evaluable`` na evidência — indisponibilidade
    não vira normalidade.

    O campo ``confidence`` (0–100) é a intensidade relativa do score,
    ``int(|clamp(score, -0.5, 0)| / 0.5 · 100)`` — no pipeline online,
    ``min(score_norm, 1) · 100`` —, com 20 pontos a menos (mínimo 10) no AX.
    Não é uma confiança calibrada nem uma probabilidade.

    Args:
        features: Dicionário de features (valores brutos, não normalizados)
        score: Score de anomalia normalizado para [-0.5, 0.0] (menor = mais anômalo)

    Returns:
        Lista de anomalias classificadas com tipo A1-A7 ou AX.
    """
    from ai.anomaly_rules import evaluate_criteria

    clamped_score = max(-0.5, min(0.0, score))
    # Intensidade relativa (0–100), não confiança calibrada.
    confidence = int(abs(clamped_score) / 0.5 * 100)
    evaluation = evaluate_criteria(features)
    rules = evaluation.to_evidence()

    anomalies: List[Dict] = []
    for code in evaluation.hits(order=_CLASSIFY_ORDER):
        description, action = _describe(code, features, evaluation)
        anomaly = build_anomaly(code, confidence, description, action,
                                features, score, rules=rules)
        if code == "A6":
            # Detecção e localização são coisas distintas: o enlace e os dois nós ficam
            # registrados, sem eleger um deles como causa.
            links = _asymmetric_links(features)
            anomaly["evidence"]["asymmetric_links"] = links
            anomaly["evidence"]["origin"] = "not_determined"
            if links:
                anomaly["neighbor_address"] = links[0]["neighbor"]
        anomalies.append(anomaly)

    if not anomalies:
        anomalies.append({
            "anomaly_type": "AX",
            "category": "statistical",
            "severity": "low",
            "description": f"Outlier estatístico não classificado pelas regras A1-A7 (score={score:.3f})",
            "suggested_action": (
                "Monitorar padrão e revisar as features da janela. O modelo detectou "
                "um ponto fora do padrão, mas sem evidência suficiente para atribuir "
                "uma causa A1-A7."
            ),
            "evidence": {
                "features": {
                    k: round(v, 3) if isinstance(v, float) else v
                    for k, v in features.items()
                    if not k.startswith("_")
                },
                "anomaly_score": round(score, 4),
                "rules": rules,
            },
            "detection_method": "ml",
            "confidence": max(confidence - 20, 10),
            "classification_mode_used": features.get("_classification_mode"),
        })

    return anomalies
