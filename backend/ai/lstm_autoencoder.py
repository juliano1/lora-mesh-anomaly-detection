"""
LSTM Autoencoder para detecção de anomalias temporais multivariadas.

Arquitetura:
    Input  (batch, W, 11)
    → Encoder LSTM (hidden=32, num_layers=1)
    → Bottleneck: último hidden state → shape (batch, 32)
    → RepeatVector × W → shape (batch, W, 32)
    → Decoder LSTM (hidden=32, num_layers=1)
    → TimeDistributed Linear (32 → 11)
    → Output (batch, W, 11)

Loss: MSE elemento-a-elemento sobre reconstrução.
Otimizador: Adam, LR=1e-3, ReduceLROnPlateau.
Regularização: Dropout 0.2, weight_decay=1e-5, early stopping (paciência 5).

Agregações de score:
    S1 mean — mean_{i,f}(e_i,f)           [Malhotra 2016; Elsayed 2020]
    S2 last — mean_f(e_{W-1,f})           [Hundman 2018]
    S3 max  — max_i( mean_f(e_{i,f}) )    [Provotar 2019]

Threshold: percentil p=99 do erro num bloco de calibração separado da validação
           que escolhe a época (sem ele, cai na validação: uso legado/direto).

Persistência: state_dict PyTorch em backend/ai/models/lstm_ae_v{N}.pt
              metadados JSON em backend/ai/models/lstm_ae_v{N}.json

Reprodutibilidade: seeds fixas configuráveis via random_seed.

Referências:
    Priyanto, Hendry & Purnomo 2021 (ICITech)
    Malhotra et al. 2016 (ICML AnomalyDetection WS)
    Elsayed et al. 2020 (ACM Q2SWinet)
    Provotar et al. 2019 (IEEE AICT)
    Hundman et al. 2018 (KDD)
    Gandhimathinathan et al. 2024 (IEEE Sensors Letters)
"""

import json
import logging
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ai.numeric_guards import safe_feature_scale

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Lazy import do PyTorch — falha graciosamente se não instalado
# ──────────────────────────────────────────────────────────────────────────────

_torch_available = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _torch_available = True
except ImportError:
    logger.warning(
        "PyTorch não instalado — LSTMAutoencoderModel indisponível. "
        "Instale com: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    )

MODELS_DIR = Path(__file__).parent / "models"


class IncompatibleFeatureSchema(ValueError):
    """Artefato treinado com outra definição/ordem de features."""


def _feature_identity() -> tuple:
    from ai.feature_engineering import FEATURE_SCHEMA_VERSION
    from ai.sequence_builder import FEATURE_NAMES
    return FEATURE_SCHEMA_VERSION, list(FEATURE_NAMES)


def _extractor_revision() -> Optional[str]:
    """Rastreabilidade do código do extrator; não decide compatibilidade."""
    try:
        from dataset_export.features import extractor_revision
        return extractor_revision()
    except Exception:
        return None
N_FEATURES = 11
SCORE_AGGREGATIONS = ("mean", "last", "max")


def _set_seeds(seed: int) -> None:
    """Fixa seeds para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    if _torch_available:
        torch.manual_seed(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Rede neural PyTorch — definida apenas se torch disponível
# ──────────────────────────────────────────────────────────────────────────────

if _torch_available:

    class _LSTMAENet(nn.Module):
        """Encoder-Decoder LSTM compacto.

        Parâmetros: 4h(F+h+2) + 4h(2h+2) + F(h+1), independente de W
        (14.571 para hidden=32, F=11).
        """

        def __init__(
            self,
            n_features: int,
            hidden_size: int,
            window_size: int,
            dropout: float,
        ):
            super().__init__()
            self.n_features = n_features
            self.hidden_size = hidden_size
            self.window_size = window_size

            self.encoder = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.dropout = nn.Dropout(p=dropout)
            self.decoder = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.output_layer = nn.Linear(hidden_size, n_features)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x:          (batch, W, n_features)
            _, (h_n, _) = self.encoder(x)              # h_n: (1, batch, hidden)
            bottleneck = self.dropout(h_n.squeeze(0))  # (batch, hidden)
            repeated = bottleneck.unsqueeze(1).repeat(
                1, self.window_size, 1
            )  # (batch, W, hidden)
            decoded, _ = self.decoder(repeated)        # (batch, W, hidden)
            return self.output_layer(decoded)          # (batch, W, n_features)


def _apply_scaler(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Normalização com estatísticas explícitas (treino ou candidato)."""
    return ((X - mean) / scale).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Classe pública
# ──────────────────────────────────────────────────────────────────────────────


class LSTMAutoencoderModel:
    """
    LSTM Autoencoder para detecção de anomalias em séries temporais multivariadas.

    Interface compatível com os detectores existentes, adaptada para dados 3D
    (sequências) em vez de vetores 2D pontuais.

    Exemplo de uso::

        model = LSTMAutoencoderModel(window_size=12)
        model.train(X_train)           # X_train shape (N, 12, 11)
        score = model.score(sequence)  # sequence shape (12, 11)
        is_anomaly = model.predict(sequence)
    """

    def __init__(
        self,
        window_size: int = 12,
        hidden_size: int = 32,
        n_features: int = N_FEATURES,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 5,
        threshold_percentile: float = 99.0,
        score_aggregation: str = "mean",
        random_seed: int = 42,
        model_version: int = 1,
    ):
        """
        Args:
            window_size:           W — sub-janelas por sequência
            hidden_size:           Dimensão do estado oculto LSTM
            n_features:            Número de features de entrada (padrão 11)
            dropout:               Dropout entre encoder e decoder
            learning_rate:         Adam LR inicial
            weight_decay:          Regularização L2 (Adam)
            batch_size:            Tamanho do minilote
            epochs:                Máximo de épocas de treino
            patience:              Early stopping: épocas sem melhora na validação
            threshold_percentile:  Percentil para calibrar threshold (padrão p=99)
            score_aggregation:     "mean" (S1), "last" (S2) ou "max" (S3)
            random_seed:           Seed para reprodutibilidade
            model_version:         Versão usada para nomear arquivos de persistência
        """
        if not _torch_available:
            raise ImportError(
                "PyTorch é necessário para LSTMAutoencoderModel. "
                "Instale com: pip install torch --index-url https://download.pytorch.org/whl/cpu"
            )
        if score_aggregation not in SCORE_AGGREGATIONS:
            raise ValueError(
                f"score_aggregation deve ser um de {SCORE_AGGREGATIONS}"
            )

        self.window_size = window_size
        self.hidden_size = hidden_size
        self.n_features = n_features
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.threshold_percentile = threshold_percentile
        self.score_aggregation = score_aggregation
        self.random_seed = random_seed
        self.model_version = model_version

        self._net: Optional["_LSTMAENet"] = None
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None
        self._threshold: float = float("inf")
        self._is_trained: bool = False
        self.training_samples: int = 0
        self.last_train_time: Optional[datetime] = None
        self.last_val_loss: float = float("inf")
        self.threshold_source: Optional[str] = None   # "calibration" | "validation"
        self.last_train_error: Optional[str] = None

        MODELS_DIR.mkdir(exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Treino
    # ──────────────────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        X_calib: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Treina o LSTM-AE.

        Contrato:
            - Entrada inválida (dimensão, W, n_features, conjunto vazio, NaN/Inf)
              → ValueError. O chamador não pode convertê-la em sucesso/normalidade.
            - Dados insuficientes ou treino numericamente inválido → False, com o
              motivo em ``last_train_error``.
            - O estado ativo (pesos, scaler, limiar) só é substituído quando o
              candidato termina com perdas, pesos, scores e limiar finitos.

        Args:
            X:     Sequências de treino, shape (N, W, n_features).
            X_val: Sequências de validação, shape (M, W, n_features): só escolhem a
                   época (parada antecipada e LR).
                   Se None, os últimos 20% de X são usados (holdout temporal).
            X_calib: Sequências de calibração, posteriores à validação: definem o
                   limiar. Se None, o limiar sai da validação (``threshold_source``
                   = "validation"), o que tende a um limiar otimista.
        """
        X = self._validate_input(X, "X")
        if X_val is not None:
            X_val = self._validate_input(X_val, "X_val")
        if X_calib is not None:
            X_calib = self._validate_input(X_calib, "X_calib")
        self.last_train_error = None

        if len(X) < 10:
            self.last_train_error = f"insufficient_sequences: {len(X)} < 10"
            logger.warning(f"LSTM-AE: poucas sequências de treino ({len(X)} < 10)")
            return False

        # Divisão temporal: 80% treino / 20% validação (sem shuffle)
        if X_val is None:
            split = max(1, int(len(X) * 0.8))
            X_train, X_val = X[:split], X[split:]
        else:
            X_train = X
        if len(X_val) == 0:
            self.last_train_error = "empty_validation"
            return False

        try:
            _set_seeds(self.random_seed)
            # ── Normalização ajustada no treino; escala 1 p/ variância nula ──
            # Estatísticas acumuladas em float64: em float32, uma constante decimal
            # (ex.: RSSI −83,2) gera desvio espúrio > MIN_FEATURE_STD e escala ~7e-4.
            # Só depois se converte para o dtype de inferência.
            flat = X_train.reshape(-1, self.n_features).astype(np.float64)
            mean = flat.mean(axis=0).astype(np.float32)
            scale = safe_feature_scale(flat.std(axis=0)).astype(np.float32)
            X_train_norm = _apply_scaler(X_train, mean, scale)
            X_val_norm = _apply_scaler(X_val, mean, scale)

            # ── Candidato isolado ─────────────────────────────────────────────
            net = _LSTMAENet(self.n_features, self.hidden_size, self.window_size, self.dropout)
            optimizer = torch.optim.Adam(
                net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=0.5
            )
            criterion = nn.MSELoss()

            train_tensor = torch.from_numpy(X_train_norm)
            val_tensor = torch.from_numpy(X_val_norm)
            loader = DataLoader(
                TensorDataset(train_tensor),
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
            )

            best_val_loss = float("inf")
            patience_counter = 0
            best_state: Dict = {}

            for epoch in range(self.epochs):
                net.train()
                for (batch,) in loader:
                    optimizer.zero_grad()
                    loss = criterion(net(batch), batch)
                    loss.backward()
                    optimizer.step()

                net.eval()
                with torch.no_grad():
                    val_loss = criterion(net(val_tensor), val_tensor).item()

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.debug(
                            f"LSTM-AE early stopping na época {epoch + 1} "
                            f"(val_loss={best_val_loss:.6f})"
                        )
                        break

            if not best_state or not math.isfinite(best_val_loss):
                self.last_train_error = "non_finite_validation_loss"
                logger.error("LSTM-AE: perda de validação não finita — candidato descartado")
                return False

            net.load_state_dict(best_state)
            net.eval()
            for name, tensor in net.state_dict().items():
                if not torch.isfinite(tensor).all():
                    self.last_train_error = f"non_finite_weights: {name}"
                    return False

            # ── Limiar na calibração (ou, sem ela, na validação) ─────────────
            threshold_source = "calibration" if X_calib is not None else "validation"
            threshold_X = X_val_norm if X_calib is None else _apply_scaler(X_calib, mean, scale)
            val_scores = self._scores_with(net, threshold_X)
            threshold = float(np.percentile(val_scores, self.threshold_percentile))
            if not np.all(np.isfinite(val_scores)) or not math.isfinite(threshold):
                self.last_train_error = "non_finite_threshold"
                return False
        except Exception as exc:
            self.last_train_error = f"{type(exc).__name__}: {exc}"
            logger.error(f"LSTM-AE treino falhou: {exc}", exc_info=True)
            return False

        # ── Publicação do candidato (somente após validação completa) ─────────
        self._net = net
        self._scaler_mean = mean
        self._scaler_std = scale
        self._threshold = threshold
        self.threshold_source = threshold_source
        self.last_val_loss = best_val_loss
        self._is_trained = True
        self.training_samples = len(X_train)
        self.last_train_time = datetime.utcnow()

        logger.info(
            f"LSTM-AE treinado: {len(X_train)} seqs, "
            f"val_loss={best_val_loss:.6f}, "
            f"threshold(p{self.threshold_percentile:.0f}, {threshold_source})={threshold:.6f}"
        )
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Inferência
    # ──────────────────────────────────────────────────────────────────────────

    def score(self, X: np.ndarray) -> float:
        """
        Score de anomalia para uma sequência.

        Contrato: shape exato (W, n_features) e valores finitos,
        senão ValueError — sem broadcasting implícito. Score não finito também
        gera ValueError. Modelo não treinado gera RuntimeError: ausência de modelo
        não pode virar o score mais normal possível.

        Args:
            X: shape (W, n_features)

        Returns:
            float — score de anomalia (maior = mais anômalo)
        """
        X = self._validate_sequence(X)
        if not self._is_trained or self._net is None:
            raise RuntimeError("LSTM-AE não treinado: sem score")
        scores = self._compute_scores_batch(self._normalize(X[np.newaxis, ...]))
        return float(self._require_finite_scores(scores)[0])

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Scores de anomalia para um batch de sequências.

        Mesmo contrato de ``score``; batch vazio devolve array vazio.

        Args:
            X: shape (N, W, n_features)

        Returns:
            scores: shape (N,)
        """
        X = np.asarray(X, dtype=np.float32)
        self._validate_shape(X, "X")
        if len(X) == 0:
            return np.zeros(0)
        if not np.all(np.isfinite(X)):
            raise ValueError("X contém NaN/Inf")
        if not self._is_trained or self._net is None:
            raise RuntimeError("LSTM-AE não treinado: sem score")
        return self._require_finite_scores(self._compute_scores_batch(self._normalize(X)))

    def predict(self, X: np.ndarray) -> bool:
        """True se a sequência é anômala (score > threshold). Limiar não finito → ValueError."""
        score = self.score(X)
        if not math.isfinite(self._threshold):
            raise ValueError(f"limiar não finito: {self._threshold}")
        return score > self._threshold

    def get_residuals(self, X: np.ndarray) -> np.ndarray:
        """
        Erro de reconstrução elemento-a-elemento para uma sequência.

        Shape: (W, n_features). Serve ao diagnóstico do erro por feature;
        a decisão online usa só o score agregado.
        """
        if not self._is_trained or self._net is None:
            return np.zeros((self.window_size, self.n_features))
        X_batch = X[np.newaxis, ...]        # (1, W, n_features)
        X_norm = self._normalize(X_batch)
        with torch.no_grad():
            X_t = torch.from_numpy(X_norm)
            recon = self._net(X_t).numpy()
        errors = (X_norm - recon) ** 2      # (1, W, n_features)
        return errors[0]                    # (W, n_features)

    def get_residuals_vector(self, X: np.ndarray) -> np.ndarray:
        """
        Resíduos de ``get_residuals`` achatados em um único vetor.

        Shape: (W * n_features,)
        """
        return self.get_residuals(X).flatten()

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    # ──────────────────────────────────────────────────────────────────────────
    # Normalização
    # ──────────────────────────────────────────────────────────────────────────

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Aplica o scaler ajustado no treino. X shape: (..., n_features)."""
        return _apply_scaler(X, self._scaler_mean, self._scaler_std)

    def _compute_scores_batch(self, X_norm: np.ndarray) -> np.ndarray:
        """Computa scores para batch já normalizado. Shape (N,)."""
        return self._scores_with(self._net, X_norm)

    def _scores_with(self, net, X_norm: np.ndarray) -> np.ndarray:
        net.eval()
        with torch.no_grad():
            recon = net(torch.from_numpy(X_norm)).numpy()
        errors = (X_norm - recon) ** 2      # (N, W, n_features)
        return self._aggregate(errors)

    def _aggregate(self, errors: np.ndarray) -> np.ndarray:
        """
        Agrega tensor de erros (N, W, n_features) → scores (N,).

        S1 mean — mean_{i,f}(e_i,f)           [Malhotra 2016]
        S2 last — mean_f(e_{W-1,f})           [Hundman 2018]
        S3 max  — max_i(mean_f(e_{i,f}))      [Provotar 2019]
        """
        if self.score_aggregation == "last":            # S2
            return errors[:, -1, :].mean(axis=1)
        elif self.score_aggregation == "max":           # S3
            return errors.mean(axis=2).max(axis=1)
        else:                                           # "mean" — S1 (padrão)
            return errors.mean(axis=(1, 2))

    # ──────────────────────────────────────────────────────────────────────────
    # Persistência
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, version: Optional[int] = None) -> Path:
        """
        Salva state_dict + metadados JSON.

        Returns:
            Path do arquivo .pt salvo.
        """
        if not self._is_trained or self._net is None:
            raise ValueError("Modelo não treinado — nada para salvar")

        v = version if version is not None else self.model_version
        model_path = MODELS_DIR / f"lstm_ae_v{v}.pt"
        meta_path = MODELS_DIR / f"lstm_ae_v{v}.json"

        torch.save(self._net.state_dict(), model_path)

        meta = {
            "version": v,
            "window_size": self.window_size,
            "hidden_size": self.hidden_size,
            "n_features": self.n_features,
            "dropout": self.dropout,
            "score_aggregation": self.score_aggregation,
            "threshold": self._threshold,
            "threshold_percentile": self.threshold_percentile,
            "training_samples": self.training_samples,
            "last_train_time": (
                self.last_train_time.isoformat() if self.last_train_time else None
            ),
            "last_val_loss": self.last_val_loss,
            "threshold_source": self.threshold_source,
            "scaler_mean": self._scaler_mean.tolist(),
            "scaler_std": self._scaler_std.tolist(),
            "feature_schema_version": _feature_identity()[0],
            "feature_names": _feature_identity()[1],
            "extractor_revision": _extractor_revision(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        logger.info(f"LSTM-AE v{v} salvo: {model_path}")
        return model_path

    @classmethod
    def load(cls, version: int) -> "LSTMAutoencoderModel":
        """Carrega modelo do disco. Retorna instância treinada."""
        model_path = MODELS_DIR / f"lstm_ae_v{version}.pt"
        meta_path = MODELS_DIR / f"lstm_ae_v{version}.json"

        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Modelo v{version} não encontrado em {MODELS_DIR}"
            )

        meta = json.loads(meta_path.read_text())
        schema_version, feature_names = _feature_identity()
        if (meta.get("feature_schema_version") != schema_version
                or meta.get("feature_names") != feature_names):
            raise IncompatibleFeatureSchema(
                f"lstm_ae_v{version}: feature_schema_version="
                f"{meta.get('feature_schema_version')!r} (esperado {schema_version!r}); "
                f"nomes/ordem {'iguais' if meta.get('feature_names') == feature_names else 'divergentes'}. "
                "Retreinar o modelo ou usar a versão do código que gerou este artefato."
            )

        instance = cls(
            window_size=meta["window_size"],
            hidden_size=meta["hidden_size"],
            n_features=meta["n_features"],
            dropout=meta["dropout"],
            score_aggregation=meta["score_aggregation"],
            threshold_percentile=meta["threshold_percentile"],
            model_version=version,
        )
        instance._net = _LSTMAENet(
            meta["n_features"],
            meta["hidden_size"],
            meta["window_size"],
            meta["dropout"],
        )
        instance._net.load_state_dict(
            torch.load(model_path, weights_only=True, map_location="cpu")
        )
        instance._net.eval()

        instance._threshold = meta["threshold"]
        instance._scaler_mean = np.array(meta["scaler_mean"], dtype=np.float32)
        instance._scaler_std = np.array(meta["scaler_std"], dtype=np.float32)
        instance.training_samples = meta.get("training_samples", 0)
        instance.last_val_loss = meta.get("last_val_loss", float("inf"))
        instance.threshold_source = meta.get("threshold_source")
        if meta.get("last_train_time"):
            instance.last_train_time = datetime.fromisoformat(meta["last_train_time"])
        instance._is_trained = True

        logger.info(f"LSTM-AE v{version} carregado de {model_path}")
        return instance

    @classmethod
    def load_latest(cls) -> Optional["LSTMAutoencoderModel"]:
        """Carrega a versão mais recente disponível em MODELS_DIR. Retorna None se nenhuma."""
        MODELS_DIR.mkdir(exist_ok=True)
        versions = sorted(
            [
                int(p.stem.replace("lstm_ae_v", ""))
                for p in MODELS_DIR.glob("lstm_ae_v*.pt")
            ],
            reverse=True,
        )
        if not versions:
            return None
        return cls.load(versions[0])

    def next_version(self) -> int:
        """Retorna o próximo número de versão disponível para salvar."""
        MODELS_DIR.mkdir(exist_ok=True)
        existing = [
            int(p.stem.replace("lstm_ae_v", ""))
            for p in MODELS_DIR.glob("lstm_ae_v*.pt")
        ]
        return max(existing, default=0) + 1

    def get_info(self) -> dict:
        return {
            "name": "LSTM Autoencoder",
            "version": self.model_version,
            "window_size": self.window_size,
            "hidden_size": self.hidden_size,
            "n_features": self.n_features,
            "score_aggregation": self.score_aggregation,
            "threshold": self._threshold if self._is_trained else None,
            "threshold_percentile": self.threshold_percentile,
            "training_samples": self.training_samples,
            "last_trained": (
                self.last_train_time.isoformat() if self.last_train_time else None
            ),
            "last_val_loss": self.last_val_loss,
            "threshold_source": self.threshold_source,
            "is_trained": self._is_trained,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Utilitários internos
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_shape(self, X: np.ndarray, name: str) -> None:
        if X.ndim != 3:
            raise ValueError(
                f"{name} deve ser 3D (N, W, n_features), recebido shape {X.shape}"
            )
        if X.shape[1] != self.window_size:
            raise ValueError(
                f"{name} dimensão W={X.shape[1]} ≠ window_size={self.window_size}"
            )
        if X.shape[2] != self.n_features:
            raise ValueError(
                f"{name} dimensão features={X.shape[2]} ≠ n_features={self.n_features}"
            )

    def _validate_input(self, X: np.ndarray, name: str) -> np.ndarray:
        """shape, conjunto não vazio e finitude — ValueError se inválido."""
        X = np.asarray(X, dtype=np.float32)
        self._validate_shape(X, name)
        if len(X) == 0:
            raise ValueError(f"{name} vazio")
        if not np.all(np.isfinite(X)):
            raise ValueError(f"{name} contém NaN/Inf")
        return X

    def _validate_sequence(self, X: np.ndarray) -> np.ndarray:
        """Uma sequência (W, n_features) finita — ValueError caso contrário."""
        X = np.asarray(X, dtype=np.float32)
        if X.shape != (self.window_size, self.n_features):
            raise ValueError(
                f"sequência deve ter shape ({self.window_size}, {self.n_features}), "
                f"recebido {X.shape}"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("sequência contém NaN/Inf")
        return X

    @staticmethod
    def _require_finite_scores(scores: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(scores)):
            raise ValueError("score de reconstrução não finito")
        return scores
