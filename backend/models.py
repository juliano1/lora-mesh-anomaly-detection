from typing import Optional, List, Literal
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, DateTime
from datetime import datetime, timezone
from pydantic import BaseModel, Field as PydanticField


def _utcnow():
    return datetime.now(timezone.utc)


# ==========================================
# Tabela de Usuários (Autenticação)
# ==========================================
class User(SQLModel, table=True):
    """Usuários do sistema com autenticação por senha."""
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(max_length=50, unique=True, index=True)
    password_hash: str = Field(max_length=128)
    is_active: bool = Field(default=True)
    is_admin: bool = Field(default=False, description="acesso de administração")
    created_at: datetime = Field(default_factory=_utcnow)


# ==========================================
# Tabela de Redes LoRa Mesh
# ==========================================
class Network(SQLModel, table=True):
    """Cadastro de redes LoRa Mesh."""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=100, unique=True, description="Nome da rede (ex: 'Rede Lab 1')")
    description: Optional[str] = Field(default=None, max_length=500, description="Descrição informada pelo usuário")
    active: bool = Field(default=True, index=True, description="Rede ativa para operação")
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: Optional[datetime] = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False),
    )

    # Contexto de ambiente (básico; opcional na API, ver environment_context_complete)
    environment_type: Optional[str] = Field(default=None, max_length=20,
        description="Tipo de ambiente: indoor, outdoor, hybrid")
    building_type: Optional[str] = Field(default=None, max_length=30,
        description="Tipo de construção predominante")
    floors_count: Optional[int] = Field(default=None,
        description="Número de pavimentos cobertos (0 = terreno aberto/outdoor puro)")
    predominant_obstacle_material: Optional[str] = Field(default=None, max_length=20,
        description="Material predominante de obstáculo")
    obstacle_density: Optional[str] = Field(default=None, max_length=10,
        description="Densidade de obstáculos: low, medium, high")
    typical_link_distance_class: Optional[str] = Field(default=None, max_length=10,
        description="Faixa de distância: short (<50m), medium (50-200m), long (>200m)")
    availability_criticality: Optional[str] = Field(default=None, max_length=10,
        description="Criticidade de disponibilidade: low, medium, high")

    # Contexto de ambiente avançado (opcional)
    avg_install_height_m: Optional[float] = Field(default=None,
        description="Altura média de instalação dos nós (metros)")
    avg_walls_per_link: Optional[int] = Field(default=None,
        description="Quantidade média de paredes por enlace")
    node_mobility_profile: Optional[str] = Field(default=None, max_length=15,
        description="Mobilidade dos nós: fixed, semi_mobile, mobile")

    # Relacionamentos
    nodes: List["Node"] = Relationship(back_populates="network")


# ==========================================
# Tabela de Nós (O "Cadastro" do Dispositivo)
# ==========================================
class Node(SQLModel, table=True):
    node_id: int = Field(primary_key=True, description="Identificador numérico do nó, sequencial no cadastro de redes (o endereço LoRa fica em lora_address)")
    name: Optional[str] = Field(default=None, description="Nome amigável")
    lora_address: Optional[str] = Field(default=None, description="Endereço LoRa em hex (ex: 0x0002)", unique=True, index=True)

    # Estado Atual
    status: str = Field(default="offline")
    last_seen: datetime = Field(default_factory=datetime.utcnow)

    # Dados mais recentes (cache para consulta)
    battery_level: Optional[int] = Field(default=None)
    # Cache otimista do estado do LED (última ação enviada com sucesso)
    led_state: Optional[str] = Field(default="off", description="Estado do LED: 'on' ou 'off'")

    # Associação com rede
    network_id: Optional[int] = Field(default=None, foreign_key="network.id", index=True,
                                       description="Rede à qual este nó pertence")
    is_gateway: bool = Field(default=False, description="True se é gateway da rede")
    device_model: Optional[str] = Field(default=None, max_length=100,
                                         description="Modelo do dispositivo (ex: 'Heltec LoRa 32 V2')")

    # Especificações de bateria e alimentação (opcional)
    battery_type: Optional[str] = Field(
        default=None, max_length=20,
        description="Tipo da bateria: lipo, li_ion, lifepo4, nimh, lead_acid, alkaline, other"
    )
    battery_capacity_mah: Optional[int] = Field(
        default=None,
        description="Capacidade nominal da bateria em mAh (ex: 800, 1000, 3000)"
    )
    power_source: Optional[str] = Field(
        default=None, max_length=20,
        description="Fonte de alimentação: battery_only, solar, usb_mains, usb_only"
    )

    # Modo de simulação
    sim_mode: bool = Field(default=False, description="True se o nó está em modo de simulação de anomalia")
    sim_anomaly: Optional[str] = Field(default=None, max_length=3,
        description="Código da anomalia simulada: A1..A7")
    sim_started_at: Optional[datetime] = Field(default=None,
        description="Início da simulação (UTC)")
    sim_ended_at: Optional[datetime] = Field(default=None,
        description="Fim da simulação (UTC)")

    # FK para o SimulationRun ativo. NULL fora de simulação. Persiste após reinício do backend.
    current_sim_run_id: Optional[int] = Field(
        default=None,
        foreign_key="simulation_run.id",
    )

    # Relacionamento com histórico
    telemetries: List["Telemetry"] = Relationship(back_populates="node")
    neighbors: List["Neighbor"] = Relationship(back_populates="node")

    # Relacionamento com rede
    network: Optional["Network"] = Relationship(back_populates="nodes")

# ==========================================
# Tabela de Telemetria (O Histórico de Leituras)
# ==========================================
class Telemetry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    node_id: int = Field(foreign_key="node.node_id", index=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)

    # Dados de Rede (Vindos do JSON do ESP32)
    rssi: int
    snr: float
    packet_id: Optional[int] = Field(default=None)
    parent_id: Optional[int] = Field(default=None, description="Node ID do parent (nó que encaminhou)", index=True)
    
    # Métricas adicionais
    uptime: Optional[int] = Field(default=None, description="Uptime do nó em minutos")
    sequence: Optional[int] = Field(default=None, description="Número de sequência do pacote")
    battery_level: Optional[int] = Field(default=None, description="Nível de bateria 0-100%")
    
    # Dados de Sensores (Extraídos do campo 'payload' do JSON)
    temperature: Optional[float] = Field(default=None)
    humidity: Optional[float] = Field(default=None)
    
    # Payload bruto do uplink (JSON recebido via MQTT)
    payload_raw: str

    # Origem do dado — 'real' (campo) ou 'synthetic' (simulação)
    data_origin: str = Field(
        default="real",
        max_length=9,
        sa_column_kwargs={"server_default": "real"},
    )

    node: Node = Relationship(back_populates="telemetries")

# ==========================================
# Tabela de Mensagens Enviadas
# ==========================================
class MessageLog(SQLModel, table=True):
    """
    Histórico de comandos publicados para a malha (simulação e resultados do gateway).
    Rastreabilidade completa: envio → entrega → resposta.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Destino
    target_node_id: Optional[int] = Field(default=None, description="Node ID destino (None = broadcast)")
    target_lora_address: Optional[str] = Field(default=None, description="Endereço LoRa hex do destino")
    message_type: str = Field(default="unicast", description="unicast ou broadcast")
    
    # Conteúdo
    command: str = Field(description="Comando enviado (ex: LED_ON, STATUS_REQUEST)")
    payload_sent: Optional[str] = Field(default=None, description="JSON completo publicado no MQTT")
    
    # Opções de envio
    reliable: bool = Field(default=False, description="True = sendReliable (com ACK)")
    
    # Resultado
    status: str = Field(default="pending", description="pending, sent ou failed (delivered só em registros antigos)")
    rtt_ms: Optional[int] = Field(default=None, description="Round-trip time em ms (se reliable)")
    error_message: Optional[str] = Field(default=None, description="Mensagem de erro se falhou")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Quando foi criado")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Última atualização de status")


# ==========================================
# Tabela de Vizinhos (relato de múltiplos enlaces)
# ==========================================
class Neighbor(SQLModel, table=True):
    """
    Armazena todos os vizinhos diretos detectados por cada nó.
    Permite visualização de múltiplos links na topologia mesh.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    node_id: int = Field(foreign_key="node.node_id", index=True, description="Nó que reporta o vizinho")
    neighbor_address: str = Field(description="Endereço LoRa do vizinho em hex (ex: 0x0001)", index=True)
    rssi: int = Field(description="RSSI da conexão em dBm")
    snr: float = Field(description="SNR da conexão em dB")
    is_parent: bool = Field(default=False, description="True se este vizinho é o parent (rota ativa)")
    hops: Optional[int] = Field(default=None, description="Número de saltos (hops) até o destino")
    distance_m: Optional[float] = Field(default=None, description="Distância estimada em metros (modelo log-distance)")
    link_quality: Optional[int] = Field(default=None, description="índice heurístico de qualidade 0-10 derivado de RSSI/SNR (não é PDR medido)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True, description="Quando foi detectado")

    # Origem do dado — 'real' ou 'synthetic'
    data_origin: str = Field(
        default="real",
        max_length=9,
        sa_column_kwargs={"server_default": "real"},
    )

    # Relacionamento
    node: Node = Relationship(back_populates="neighbors")


# ==========================================
# Tabela de Anomalias
# ==========================================
class Anomaly(SQLModel, table=True):
    """
    Registro de anomalias detectadas na rede mesh.
    Detectadas pelo pipeline online (IF e/ou LSTM-AE); as regras A1–A7 só atribuem o tipo.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Contexto
    node_id: int = Field(foreign_key="node.node_id", index=True,
                         description="Nó afetado pela anomalia")
    neighbor_address: Optional[str] = Field(default=None,
                         description="Vizinho envolvido (se anomalia de link)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    # Classificação
    anomaly_type: str = Field(index=True,
                         description="Tipo: A1 a A7")
    category: str = Field(index=True,
                         description="Categoria: radio, hardware, topology, security, statistical")
    severity: str = Field(default="medium",
                         description="low, medium, high")
    
    # Detalhes
    description: str = Field(description="Descrição legível da anomalia")
    suggested_action: Optional[str] = Field(default=None,
                         description="Ação sugerida ao operador")
    
    # Evidência
    evidence: Optional[str] = Field(default=None,
                         description="JSON com dados que evidenciam a anomalia")
    detection_method: str = Field(default="ml",
                         description="Método de detecção (ml)")
    # Nome mantido por compatibilidade de esquema/exportação: é o score normalizado
    # reescalado para 0–100 (50/0 fixos sem classificação), não uma probabilidade.
    confidence: Optional[float] = Field(default=None,
                         description="Intensidade relativa do score (0-100); não é probabilidade")
    
    # Estado
    resolved: bool = Field(default=False, index=True,
                         description="Operador marcou como resolvida")
    resolved_at: Optional[datetime] = Field(default=None)
    notes: Optional[str] = Field(default=None,
                         description="Notas do operador sobre esta anomalia")

    # Modo de classificação ativo no momento da detecção (auditoria A/B)
    # Valores: "absolute" | "contextual" | "contextual_fallback"
    classification_mode_used: Optional[str] = Field(
        default=None, max_length=30, index=True,
        description="Modo de classificação usado na detecção: absolute, contextual ou contextual_fallback",
    )

    # Rastreabilidade do algoritmo de detecção
    # Texto livre (max 40): valores de versões anteriores (if_only, hybrid_A…) e os
    # derivados da combinação gravados por esta versão (single:if…)
    detector_algorithm: Optional[str] = Field(
        default=None, max_length=40, index=True,
        description=(
            "algoritmo que gerou esta anomalia — "
            "versões anteriores: if_only, lstm_only, if_shadow, lstm_shadow, hybrid_A/B/C/D, if; "
            "esta versão (derivado): single:if, cascade:if>lstm, shadow:lstm, fusion:if+lstm"
        ),
    )

    # Procedência estruturada (fonte de verdade para queries/filtros)
    detector_combo: Optional[str] = Field(
        default=None, max_length=10, index=True,
        description="modo de combinação — single|cascade|shadow|fusion",
    )
    detector_primary: Optional[str] = Field(
        default=None, max_length=30, index=True,
        description="id do algoritmo que produziu este registro",
    )
    detector_secondary: Optional[str] = Field(
        default=None, max_length=30, index=True,
        description="id do parceiro na combinação; None quando combinação = 1 algoritmo",
    )

    # FK para SimulationRun ativa no momento da detecção
    sim_run_id: Optional[int] = Field(
        default=None,
        foreign_key="simulation_run.id",
        index=True,
    )


# ==========================================
# Tabela de Execuções de Simulação
# ==========================================
class SimulationRun(SQLModel, table=True):
    __tablename__ = "simulation_run"

    id: Optional[int] = Field(default=None, primary_key=True)
    node_id: int = Field(foreign_key="node.node_id", index=True)
    anomaly: str = Field(max_length=3)
    started_at: datetime = Field(index=True)
    ended_at: Optional[datetime] = Field(default=None, index=True)
    duration_planned: int
    ended_reason: Optional[str] = Field(default=None, max_length=20)
    telemetry_count: int = Field(default=0)
    # Instante da 1ª telemetria sintética do run (onset real da
    # anomalia). Permite latência do detector (detected_at - onset) separada da
    # latência total/ponta-a-ponta (detected_at - started_at), que embute o atraso
    # do firmware (~60-90s) até começar a emitir a assinatura da anomalia.
    anomaly_onset_at: Optional[datetime] = Field(default=None)
    detected_at: Optional[datetime] = Field(default=None)
    detector_primary: Optional[str] = Field(default=None, max_length=30)
    detector_secondary: Optional[str] = Field(default=None, max_length=30)
    detector_combo: Optional[str] = Field(default=None, max_length=10)


# ==========================================
# Schemas Pydantic para CRUD de Redes
# ==========================================

class DeviceRegistration(BaseModel):
    """Dispositivo dentro do cadastro de rede."""
    name: str = PydanticField(max_length=100, description="Nome do dispositivo")
    lora_address: str = PydanticField(
        pattern=r'^0x[0-9A-Fa-f]{4}$',
        description="Endereço LoRa em hex (ex: '0x0002') — exatamente 4 dígitos hex",
    )
    device_model: Optional[str] = PydanticField(default=None, description="Modelo (apenas gateways)")

    # Bateria e alimentação (opcional)
    battery_type: Optional[str] = PydanticField(
        default=None,
        pattern=r'^(lipo|li_ion|lifepo4|nimh|lead_acid|alkaline|other)$',
        description="Tipo da bateria"
    )
    battery_capacity_mah: Optional[int] = PydanticField(
        default=None, ge=1, le=99999,
        description="Capacidade em mAh"
    )
    power_source: Optional[str] = PydanticField(
        default=None,
        pattern=r'^(battery_only|solar|usb_mains|usb_only)$',
        description="Fonte de alimentação principal"
    )


class NetworkCreate(BaseModel):
    """Schema de criação/edição de rede."""
    name: str = PydanticField(max_length=100, description="Nome da rede")
    description: Optional[str] = PydanticField(default=None, max_length=500)
    gateways: List[DeviceRegistration] = PydanticField(min_length=1, description="Pelo menos 1 gateway")
    devices: List[DeviceRegistration] = PydanticField(default=[], description="End devices da rede")

    # Contexto de ambiente — opcional na API. Redes sem contexto são válidas e usam
    # o padrão de propagação (n=2.7, sem perda por paredes); environment_context_complete
    # na resposta indica se os 5 campos obrigatórios abaixo estão preenchidos.
    environment_type: Optional[Literal["indoor", "outdoor", "hybrid"]] = None
    building_type: Optional[Literal[
        "warehouse", "office", "residential", "industrial",
        "open_field", "urban_street", "other"
    ]] = None
    obstacle_density: Optional[Literal["low", "medium", "high"]] = None
    typical_link_distance_class: Optional[Literal["short", "medium", "long"]] = None
    availability_criticality: Optional[Literal["low", "medium", "high"]] = None

    # Opcionais
    floors_count: Optional[int] = PydanticField(default=None, ge=0, le=200,
        description="Número de pavimentos (0 = outdoor puro)")
    predominant_obstacle_material: Optional[Literal[
        "concrete", "drywall", "glass", "wood", "metal", "brick", "mixed"
    ]] = None

    # Contexto de ambiente avançado (opcional)
    avg_install_height_m: Optional[float] = PydanticField(default=None, gt=0, le=100,
        description="Altura média de instalação (metros)")
    avg_walls_per_link: Optional[int] = PydanticField(default=None, ge=0, le=20,
        description="Quantidade média de paredes por enlace")
    node_mobility_profile: Optional[Literal["fixed", "semi_mobile", "mobile"]] = None


class DeviceResponse(BaseModel):
    """Resposta com dados de um dispositivo."""
    node_id: int
    name: Optional[str]
    lora_address: str
    is_gateway: bool
    device_model: Optional[str]
    status: str
    battery_type: Optional[str] = None
    battery_capacity_mah: Optional[int] = None
    power_source: Optional[str] = None

    @classmethod
    def from_node(cls, node: "Node") -> "DeviceResponse":
        return cls(
            node_id=node.node_id,
            name=node.name,
            lora_address=node.lora_address or "?",
            is_gateway=node.is_gateway,
            device_model=node.device_model,
            status=node.status,
            battery_type=getattr(node, "battery_type", None),
            battery_capacity_mah=getattr(node, "battery_capacity_mah", None),
            power_source=getattr(node, "power_source", None),
        )


class NetworkResponse(BaseModel):
    """Resposta com dados completos de uma rede."""
    id: int
    name: str
    description: Optional[str]
    active: bool
    gateways: List[DeviceResponse]
    devices: List[DeviceResponse]
    created_at: datetime
    updated_at: datetime

    # Contexto de ambiente
    environment_type: Optional[str] = None
    building_type: Optional[str] = None
    floors_count: Optional[int] = None
    predominant_obstacle_material: Optional[str] = None
    obstacle_density: Optional[str] = None
    typical_link_distance_class: Optional[str] = None
    availability_criticality: Optional[str] = None
    avg_install_height_m: Optional[float] = None
    avg_walls_per_link: Optional[int] = None
    node_mobility_profile: Optional[str] = None
    # True se os 5 campos obrigatórios (environment_type, building_type, obstacle_density,
    # typical_link_distance_class, availability_criticality) estão preenchidos
    environment_context_complete: bool = False


# ==========================================
# Configurações de Parâmetros ML
# ==========================================
class MLSettings(SQLModel, table=True):
    """Configurações de parâmetros ML para detecção contínua — singleton (id=1)."""
    id: int = Field(default=1, primary_key=True)

    # Isolation Forest (detecção contínua)
    ml_if_contamination: float = Field(default=0.03, description="IF contamination (contínua)")
    ml_if_n_estimators: int = Field(default=100, description="IF número de árvores (contínua)")

    # Legado sem uso (LOF não faz parte desta versão): colunas mantidas para bancos existentes,
    # onde são NOT NULL sem default no banco; a API não as expõe.
    ml_lof_contamination: float = Field(default=0.05, description="LOF contamination (contínua)")
    ml_lof_n_neighbors: int = Field(default=20, description="LOF vizinhos (contínua)")

    # Camada 1 — detecção rápida por regras
    ml_fast_detection: bool = Field(default=False, description="OBSOLETO: camada fast_rule removida; valor ignorado e reativação recusada pela API")

    # Controle de nós simulados na detecção contínua
    ml_detect_simulated: bool = Field(default=True, description="Quando True: nós em sim_mode são incluídos na detecção. Quando False: excluídos.")

    # Flag global de modo de classificação de A1/A2
    # "contextual" (padrão): usa resíduos do contexto de ambiente quando cadastrado, absoluto como fallback por rede.
    # "absolute": força thresholds fixos em todas as redes, ignora o contexto de ambiente.
    anomaly_classification_mode: str = Field(
        default="contextual",
        description="Modo de classificação de anomalias: 'contextual' ou 'absolute' (clássico)",
    )

    # Legado sem efeito: o HybridDetector não faz parte desta versão e a detecção usa sempre
    # o OnlinePipeline (campo `online`). Colunas mantidas para bancos existentes.
    ml_use_hybrid: bool = Field(
        default=False,
        description="Legado sem efeito; a detecção usa sempre a seleção online.",
    )
    ml_hybrid_mode: str = Field(
        default="if_only",
        description="Legado sem efeito; mantido para bancos existentes.",
    )
    ml_lstm_window: int = Field(
        default=12,
        description="W — número de sub-janelas de 5 min por sequência LSTM",
    )
    ml_lstm_alpha: float = Field(
        default=0.5,
        description="Legado sem efeito; o peso da fusão vem de ml_online_fusion_alpha.",
    )
    ml_lstm_threshold_strategy: str = Field(
        default="percentile",
        description="estratégia de limiar do LSTM-AE; só 'percentile' é aceita",
    )

    # Parâmetros LSTM explícitos (NULL = default efetivo documentado)
    ml_lstm_history_hours: Optional[float] = Field(
        default=None, description="histórico de treino do LSTM (h); NULL = ML_HISTORY_HOURS",
    )
    ml_lstm_hidden: Optional[int] = Field(default=None, description="hidden_size do LSTM-AE")
    ml_lstm_epochs: Optional[int] = Field(default=None, description="épocas máximas")
    ml_lstm_patience: Optional[int] = Field(default=None, description="paciência do early stopping")
    ml_lstm_score_aggregation: Optional[str] = Field(
        default=None, description="mean|last|max; mudar exige recalibração",
    )
    ml_lstm_threshold_percentile: Optional[float] = Field(
        default=None, description="percentil do limiar LSTM no bloco de calibração",
    )

    # Seleção online de algoritmos
    ml_online_primary: str = Field(
        default="lstm_ae",  # = DEFAULT_ONLINE_PRIMARY; sem PyTorch o pipeline cai para o IF e registra o motivo
        description="id do algoritmo primário online (registry)",
    )
    ml_online_secondary: Optional[str] = Field(
        default=None,
        description="id do algoritmo secundário online; None = single",
    )
    ml_online_combo: str = Field(
        default="single",
        description="modo de combinação — single|cascade|shadow|fusion",
    )
    ml_online_fusion_alpha: float = Field(
        default=0.5,
        description="peso do primário no fusion (0.0–1.0)",
    )

    # Legado sem uso nesta versão (análise por LLM e alertas Telegram de versões
    # anteriores): colunas mantidas para bancos existentes; a API não as expõe.
    llm_enabled: bool = Field(default=False, description="Habilitar análise LLM")
    llm_provider: str = Field(default="gemini", description="Provider LLM: gemini, anthropic ou ollama")
    llm_model: str = Field(default="gemini-2.0-flash", description="Modelo LLM a utilizar")
    # Configuração de provider local + parâmetros
    llm_base_url: str = Field(default="http://ollama:11434", description="base_url quando provider=ollama")
    llm_temperature: float = Field(default=0.2, description="temperatura da geração")
    llm_max_tokens: int = Field(default=300, description="max_tokens / num_predict")
    llm_num_ctx: int = Field(default=4096, description="janela de contexto (Ollama)")
    llm_timeout_seconds: int = Field(default=30, description="timeout por chamada (recomendar 120 p/ ollama)")
    llm_max_retries: int = Field(default=2, description="tentativas extras (0-5); total = 1 + valor")
    telegram_enabled: bool = Field(default=False, description="Habilitar notificações Telegram")

    updated_at: Optional[datetime] = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False),
    )


# ==========================================
# Log de Análises LLM
# ==========================================
class LLMAnalysis(SQLModel, table=True):
    """
    Legado: análises por LLM de versões anteriores. Esta versão não tem código que a
    use; a tabela continua no esquema para manter bancos existentes atualizáveis.
    """
    __tablename__ = "llm_analysis"

    id: Optional[int] = Field(default=None, primary_key=True)

    # Referência à anomalia original
    anomaly_id: int = Field(foreign_key="anomaly.id", index=True,
                            description="Anomalia analisada")
    node_id: int = Field(foreign_key="node.node_id", index=True)
    anomaly_type: str = Field(description="Tipo da anomalia (A1-A7)")
    timestamp: datetime = Field(default_factory=_utcnow, index=True)

    # Decisão da LLM
    verdict: Optional[str] = Field(default=None, index=True,
                         description="REAL, WARNING, FALSO_POSITIVO, INCONCLUSIVO (null enquanto pending/erro)")
    confidence_llm: Optional[float] = Field(default=None, description="Confiança da LLM (0-1)")
    confidence_ml: float = Field(description="Confiança do modelo ML original (0-1)")
    severity_llm: Optional[str] = Field(default=None, description="Severidade atribuída pela LLM")
    reasoning: Optional[str] = Field(default=None, description="Explicação em linguagem natural")
    action: Optional[str] = Field(default=None,
                                  description="Ação sugerida pela LLM")

    # Metadata
    model_used: str = Field(description="Modelo LLM utilizado (ex: gemini-2.0-flash)")
    features_json: Optional[str] = Field(default=None,
                                         description="Features em JSON (base para ajuste fino do LLM por LoRA, Low-Rank Adaptation; sem relação com o rádio LoRa)")

    # Rede resolvida + estado de execução (fila)
    network_id: Optional[int] = Field(default=None, index=True, description="rede da anomalia")
    status: str = Field(default="success", index=True,
                        description="pending|processing|success|timeout|error|parse_error")
    # Snapshot de entrada
    system_prompt: Optional[str] = Field(default=None)
    user_prompt: Optional[str] = Field(default=None)
    prompt_version: Optional[str] = Field(default=None)
    context_reference_time: Optional[datetime] = Field(default=None)
    # Resposta crua + erro de parsing (mensagem textual)
    raw_response: Optional[str] = Field(default=None)
    parse_error: Optional[str] = Field(default=None)
    # Parâmetros efetivos (snapshot)
    provider: Optional[str] = Field(default=None)
    mode: Optional[str] = Field(default=None, description="api|local")
    temperature: Optional[float] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    num_ctx: Optional[int] = Field(default=None)
    top_p: Optional[float] = Field(default=None)
    seed: Optional[int] = Field(default=None)
    llm_timeout_seconds: Optional[int] = Field(default=None)
    llm_max_retries: Optional[int] = Field(default=None)
    # Métricas agregadas
    started_at: Optional[datetime] = Field(default=None)
    finished_at: Optional[datetime] = Field(default=None)
    latency_ms: Optional[int] = Field(default=None)
    input_tokens: Optional[int] = Field(default=None)
    output_tokens: Optional[int] = Field(default=None)
    attempt_count: int = Field(default=0)
    lease_recovery_count: int = Field(default=0)
    # Campos de fila (claim/lease)
    enqueued_at: Optional[datetime] = Field(default=None, index=True)
    claimed_at: Optional[datetime] = Field(default=None)
    worker_id: Optional[str] = Field(default=None)
    lease_expires_at: Optional[datetime] = Field(default=None)
    # Estado rico de Telegram
    telegram_status: Optional[str] = Field(default=None,
        description="sent|not_applicable|suppressed_shadow|disabled|failed|pending|legacy_unknown")
    telegram_sent_at: Optional[datetime] = Field(default=None)
    telegram_error: Optional[str] = Field(default=None)

    # Feedback humano (preenchido manualmente pelo operador)
    human_verdict: Optional[str] = Field(default=None,
                                         description="Classificação correta pelo operador")
    human_notes: Optional[str] = Field(default=None,
                                       description="Notas do operador sobre a decisão")

    # Telegram
    telegram_sent: bool = Field(default=False,
                                description="Se alerta foi enviado ao Telegram")


# ==========================================
# Auditoria por tentativa + histórico de revisão humana
# ==========================================
class LLMInferenceAttempt(SQLModel, table=True):
    """Legado (sem uso nesta versão): uma linha por chamada ao provedor de LLM."""
    __tablename__ = "llm_inference_attempt"

    id: Optional[int] = Field(default=None, primary_key=True)
    analysis_id: int = Field(foreign_key="llm_analysis.id", index=True)
    attempt_number: int = Field(description="1, 2, 3...")
    provider: Optional[str] = Field(default=None)
    model: Optional[str] = Field(default=None)
    model_digest: Optional[str] = Field(default=None)
    started_at: Optional[datetime] = Field(default=None)
    finished_at: Optional[datetime] = Field(default=None)
    latency_ms: Optional[int] = Field(default=None)
    status: str = Field(description="success|timeout|error|parse_error")
    input_tokens: Optional[int] = Field(default=None)
    output_tokens: Optional[int] = Field(default=None)
    raw_response: Optional[str] = Field(default=None)
    provider_metadata_json: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)


class LLMHumanReview(SQLModel, table=True):
    """Legado (sem uso nesta versão): validações humanas de análises por LLM."""
    __tablename__ = "llm_human_review"

    id: Optional[int] = Field(default=None, primary_key=True)
    analysis_id: int = Field(foreign_key="llm_analysis.id", index=True)
    reviewer_user_id: Optional[int] = Field(default=None, foreign_key="user.id",
                                            description="null em reviews legadas")
    human_verdict: str = Field(description="REAL|WARNING|FALSO_POSITIVO|INCONCLUSIVO")
    notes: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=_utcnow, index=True)
    supersedes_review_id: Optional[int] = Field(default=None,
                                                foreign_key="llm_human_review.id")
    source: str = Field(default="human", description="human|legacy")


class OnlineInference(SQLModel, table=True):
    """Decisão do pipeline online por nó e ciclo, inclusive normal e
    indisponível. Guarda as decisões e scores dos dois componentes em qualquer modo,
    para avaliar shadow, fusão e cascata sobre os mesmos pontos e calcular falso
    positivo/recall por janela — não só sobre os alertas emitidos."""
    __tablename__ = "online_inference"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    reference_time: datetime = Field(index=True, description="corte do ciclo (UTC)")
    network_id: int = Field(index=True)
    node_id: int = Field(index=True)
    pipeline_status: str = Field(description="published|no_pipeline")
    training_status: Optional[str] = Field(default=None, description="estado do treino sem pipeline")
    combo: Optional[str] = Field(default=None)
    primary_id: Optional[str] = Field(default=None)
    secondary_id: Optional[str] = Field(default=None)
    bundle_id: Optional[str] = Field(default=None, description="pacote congelado em uso")
    model_signature_sha: Optional[str] = Field(default=None)
    alpha: Optional[float] = Field(default=None)
    tau: Optional[float] = Field(default=None)
    tau_status: Optional[str] = Field(default=None)
    primary_status: Optional[str] = Field(default=None)
    primary_reason: Optional[str] = Field(default=None)
    primary_is_anomaly: Optional[bool] = Field(default=None)
    primary_score_norm: Optional[float] = Field(default=None)
    primary_raw_score: Optional[float] = Field(default=None)
    primary_threshold: Optional[float] = Field(default=None)
    secondary_status: Optional[str] = Field(default=None)
    secondary_reason: Optional[str] = Field(default=None)
    secondary_is_anomaly: Optional[bool] = Field(default=None)
    secondary_score_norm: Optional[float] = Field(default=None)
    secondary_raw_score: Optional[float] = Field(default=None)
    secondary_threshold: Optional[float] = Field(default=None)
    sequence_context_end: Optional[str] = Field(default=None)
    combination_status: Optional[str] = Field(default=None)
    combination_reason: Optional[str] = Field(default=None)
    combination_is_anomaly: Optional[bool] = Field(default=None)
    combination_score: Optional[float] = Field(default=None)
    primary_rank: Optional[float] = Field(default=None, description="posto do primário em relação aos scores de treino (escala da fusão)")
    secondary_rank: Optional[float] = Field(default=None, description="posto do secundário em relação aos scores de treino")
    fusion_score: Optional[float] = Field(default=None,
        description="α·posto_primário+(1−α)·posto_secundário sempre que ambos válidos (contrafactual fora da fusão)")
    fusion_decision: Optional[bool] = Field(default=None, description="fusion_score > τ com τ válido")
    emitted: int = Field(default=0, description="anomalias emitidas pela combinação")
    outcomes_json: Optional[str] = Field(default=None, description="destino de cada emissão")
    rule_hits: Optional[str] = Field(default=None, description="JSON: critérios A1–A7 em hit")
    rule_not_evaluable: Optional[str] = Field(default=None)
    features_json: Optional[str] = Field(default=None)
    features_reason: Optional[str] = Field(default=None)
