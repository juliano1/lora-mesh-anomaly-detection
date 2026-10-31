import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Caminho absoluto para o banco — sempre em backend/, independente do CWD
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_PATH = os.path.join(_BACKEND_DIR, "lora_mesh.db").replace("\\", "/")

class Settings(BaseSettings):
    MQTT_BROKER: str = "localhost"
    MQTT_PORT: int = 1883
    # Tópicos fixos do contrato com o gateway (mqtt_service_sync.py):
    #   assina lab/+/uplink e lab/+/send_result; publica comandos em lora/mesh/{rede}/command.
    MQTT_TOPIC_STATUS: str = "lora/status"  # mudanças online/offline publicadas pelo monitor

    # Credenciais MQTT (definir no .env quando o broker exigir autenticação)
    MQTT_USER: str = ""
    MQTT_PASS: str = ""

    # Banco de dados — SQLite por padrão; PostgreSQL opcional via .env
    DATABASE_URL: str = f"sqlite:///{_DB_PATH}"
    # Exemplo: DATABASE_URL=postgresql://<usuario>:<senha>@localhost:5432/lora_mesh

    # Endereço da API. Padrão em loopback: expor a outra interface exige TLS e
    # autenticação na frente da aplicação.
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    # Origens de navegador permitidas (vazio = nenhuma; a versão pública não tem interface).
    CORS_ORIGINS: str = ""

    # Configurações de Health Check
    NODE_TIMEOUT_SECONDS: int = 240  # Nó considerado offline após 240s sem dados (12 intervalos de beacon de 20 s)
    HEALTH_CHECK_INTERVAL: int = 15  # Verificar saúde dos nós a cada 15 segundos

    # Configurações de IA/ML (Isolation Forest)
    # ML_CONTAMINATION, ML_N_ESTIMATORS e ML_LSTM_WINDOW só valem quando a linha `mlsettings`
    # não existe: ela nasce com 0.03/100/12 e prevalece. Para mudar, use PATCH /api/settings/ml.
    ML_CONTAMINATION: float = 0.03   # Fração esperada de anomalias — 0.03 reduz FP em redes estáveis de laboratório
    ML_N_ESTIMATORS: int = 100  # Número de árvores do Isolation Forest
    ML_MIN_SAMPLES: int = 3  # Sem uso no pipeline online (mínimo de nós online para treinar)

    # Sliding Window para treino do Isolation Forest
    ML_HISTORY_HOURS: int = 24       # Período histórico para treino (h) — 24h captura variações diárias
    ML_WINDOW_MINUTES: int = 15      # Duração da janela de features (min)
    ML_TRAIN_OVERLAP_PCT: float = 0.5  # Overlap entre janelas de treino (50% = passo de 7 min, arredondado para baixo)
    ML_MIN_TRAIN_SAMPLES: int = 10   # Mínimo de janelas de referência (somando os nós) para treinar o IF
    ML_MIN_TRAIN_SAMPLES_FULL: int = 30  # Sem uso no pipeline online (mínimo para treino completo)

    # Obsoletos: a camada de regras rápidas foi removida (as regras só classificam)
    ML_FAST_WINDOW_MINUTES: int = 5       # OBSOLETO: camada fast_rule removida
    ML_FAST_DETECTION_ENABLED: bool = False  # OBSOLETO: ignorado; não reativa a camada

    # Controle de retry de treino
    ML_RETRY_INTERVAL_CYCLES: int = 5  # Sem uso: a espera após falha vem de ML_ONLINE_TRAIN_RETRY_SECONDS

    # LSTM-AE do pipeline online. ML_LSTM_WINDOW é sobrescrito por `mlsettings` (ver acima).
    ML_LSTM_WINDOW: int = 12             # W — sub-janelas por sequência (12 × 5 min = 60 min)
    ML_LSTM_WINDOW_MINUTES: int = 5      # Duração de cada sub-janela (min)
    ML_LSTM_HIDDEN: int = 32             # Tamanho do estado oculto LSTM
    ML_LSTM_EPOCHS: int = 100            # Máximo de épocas de treino
    ML_LSTM_PATIENCE: int = 5            # Early stopping: épocas sem melhora
    # NÃO aplicado automaticamente. O histórico efetivo do LSTM vem de
    # MLSettings.ml_lstm_history_hours (API) ou, se nulo, de ML_HISTORY_HOURS (24 h).
    # Mantido só como referência documental.
    ML_LSTM_HISTORY_HOURS: float = 168.0
    ML_LSTM_ALPHA: float = 0.5           # Sem uso: o peso da fusão vem de mlsettings.ml_online_fusion_alpha
    # forward-fill máximo após observação válida (sub-janelas)
    ML_LSTM_MAX_IMPUTE_GAP: int = 2
    # Seed única registrada no manifesto/config efetiva
    ML_RANDOM_SEED: int = 42
    # Margem contextual (min) ao redor de simulações na referência saudável
    ML_REFERENCE_MARGIN_MINUTES: float = 0.0

    # Ciclo de vida dos modelos online (treino, congelamento e registro por ciclo)
    ML_ONLINE_TRAIN_RETRY_SECONDS: int = 300   # espera após falha de treino (por runner)
    ML_ONLINE_FREEZE_MODELS: bool = False      # True: sem retreino periódico; reusa pacote salvo
    ML_ONLINE_BUNDLE_DIR: str = os.path.join(_BACKEND_DIR, "data", "online_models")
    ML_ONLINE_INFERENCE_LOG: bool = True       # registro por nó/ciclo (tabela online_inference)

    # False desabilita o LSTM-AE (ambientes sem PyTorch instalado).
    ML_LSTM_ENABLED: bool = True

    # Acervo de dados para experimentos externos
    DATASET_SOURCE_ID: str = ""
    DATA_ORIGIN_TRACKED_SINCE: Optional[str] = None
    DATASET_EXPORT_MAX_DAYS: float = 31.0

    # Autenticação JWT
    JWT_SECRET_KEY: str = ""  # Obrigatória; validada antes de iniciar os serviços.
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 480  # 8 horas
    # Falhas de login por cliente+usuário antes do bloqueio temporário (memória do processo).
    LOGIN_MAX_FAILURES: int = 5
    LOGIN_LOCKOUT_SECONDS: int = 900
    ADMIN100_PASSWORD: str = ""  # Vazio: não cria esta conta.
    ADMIN200_PASSWORD: str = ""  # Vazio: não cria esta conta.

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


HMAC_JWT_ALGORITHMS = frozenset({"HS256", "HS384", "HS512"})


def validate_bootstrap_password(password: str) -> None:
    """Recusa senhas de exemplo e tamanhos incompatíveis com bcrypt."""
    if password and (
        password.strip().upper().replace("-", "_") in {"CHANGE_ME", "CHANGEME"}
        or not 12 <= len(password.encode("utf-8")) <= 72
    ):
        raise ValueError("Senha de bootstrap deve ter entre 12 e 72 bytes e não ser um exemplo")


def validate_security_settings(config: Settings) -> None:
    """Falha antes de acessar banco ou rede quando o bootstrap é inseguro."""
    if len(config.JWT_SECRET_KEY.strip().encode("utf-8")) < 32:
        raise ValueError("Defina JWT_SECRET_KEY com pelo menos 32 bytes aleatórios")
    if config.JWT_ALGORITHM not in HMAC_JWT_ALGORITHMS:
        # A chave é um segredo compartilhado: algoritmos assimétricos ou "none" não se aplicam.
        raise ValueError(f"JWT_ALGORITHM deve ser um de {sorted(HMAC_JWT_ALGORITHMS)}")
    for password in (config.ADMIN100_PASSWORD, config.ADMIN200_PASSWORD):
        validate_bootstrap_password(password)

settings = Settings()
