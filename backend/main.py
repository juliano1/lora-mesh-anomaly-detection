from fastapi import APIRouter, FastAPI, Depends, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, ConfigDict, Field, field_validator
import asyncio
import csv
import io
import json
import logging
import threading
from datetime import datetime, timedelta
from sqlmodel import SQLModel, Session, select, func
from typing import List, Literal, Optional
from jwt import InvalidTokenError

from database import engine, get_session
from models import (
    Node, Telemetry, Anomaly, Network, MLSettings,
    NetworkCreate, NetworkResponse, DeviceResponse,
)
import mqtt_service_sync
from mqtt_service_sync import (
    start_mqtt_client, stop_mqtt_client, publish_to_downlink,
    build_topology, _close_simulation_run,
)
from routers.auth import router as auth_router
from auth import decode_access_token, get_current_user, get_current_admin
from seed_users import seed_users
from config import settings, validate_security_settings


# Configuração Global de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Migrações executadas a cada inicialização, em ordem. As colunas e tabelas de LLM de
# versões anteriores continuam sendo migradas só para manter bancos existentes
# atualizáveis; esta versão não as usa.
STARTUP_MIGRATIONS = (
    "add_hybrid_fields",
    "add_online_selection",
    "add_lstm_config_fields",
    "add_detector_procedence",
    "add_data_origin_and_simulation_run",
    "add_user_is_admin",
    "add_mlsettings_llm_fields",
    "add_llm_analysis_fields",
    "llm_analysis_nullable",
    "backfill_legacy",
)

# Migrações das tabelas usadas pela ingestão MQTT e pelo detector, com o nome da função
# de cada módulo. A ordem importa: as três primeiras adicionam colunas que o ORM de
# MLSettings já lê em add_ml_settings_table.
SERVICE_MIGRATIONS = (
    ("add_llm_fields_to_ml_settings", "migrate"),
    ("add_ml_detect_simulated", "upgrade"),
    ("add_classification_mode_to_ml_settings", "upgrade"),
    ("add_ml_settings_table", "migrate"),
    ("add_battery_specs_to_node", "migrate"),
    ("rename_anomaly_types_if_to_a", "migrate"),
    ("add_simulation_fields", "migrate"),
    ("add_environment_fields_to_network", "migrate"),
    ("add_classification_mode_to_anomaly", "upgrade"),
)


def run_startup_migrations(target_engine) -> None:
    """Aplica STARTUP_MIGRATIONS e SERVICE_MIGRATIONS; RuntimeError na primeira que falhar."""
    import importlib
    steps = [(name, "upgrade") for name in STARTUP_MIGRATIONS] + list(SERVICE_MIGRATIONS)
    for name, function in steps:
        try:
            getattr(importlib.import_module(f"migrations.{name}"), function)(target_engine)
        except Exception as exc:
            raise RuntimeError(f"migração {name} falhou; inicialização interrompida") from exc
    logger.info("🔀 Migrações de inicialização aplicadas.")


# ==========================================
# Ciclo de Vida (Startup/Shutdown)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_security_settings(settings)
    # 1. Criar tabelas no banco (se não existirem)
    SQLModel.metadata.create_all(engine)
    logger.info("🗄️  Banco de dados inicializado.")

    # 1b. Migrações idempotentes para bancos criados por versões anteriores. Uma falha
    # interrompe a inicialização: seguir com o esquema incompleto quebraria a
    # autorização (is_admin) e a gravação de anomalias de forma silenciosa.
    run_startup_migrations(engine)

    # 1d. registrar algoritmos padrão (IF e, se habilitado, LSTM-AE) no registry online
    try:
        import ai.online.pipeline  # noqa: F401 — importar dispara _register_defaults()
        logger.info("algoritmos online registrados.")
    except Exception as _reg_err:
        logger.warning(f"falha ao registrar algoritmos online: {_reg_err}")

    # 2. Seed de usuários
    seed_users()
    logger.info("👤 Seed de usuários concluído.")

    # 3. Iniciar o cliente MQTT em thread separada (loop_forever do paho é bloqueante)
    loop = asyncio.get_running_loop()
    mqtt_thread = threading.Thread(target=start_mqtt_client, args=(loop,), daemon=True)
    mqtt_thread.start()
    logger.info("📡 Thread MQTT iniciada")

    # 4. Watchdog de simulação (timeout e execuções órfãs)
    async def _simulation_watchdog_loop():
        from mqtt_service_sync import _simulation_watchdog_tick
        while True:
            await asyncio.sleep(60)
            try:
                _simulation_watchdog_tick()
            except Exception as _e:
                logger.error(f"simulation_watchdog_loop error: {_e}")

    watchdog_task = asyncio.create_task(_simulation_watchdog_loop())
    logger.info("⏱️  Watchdog de simulação iniciado")

    yield

    watchdog_task.cancel()

    # Shutdown: Parar MQTT
    stop_mqtt_client()
    logger.info("🛑 Servidor desligando...")

app = FastAPI(title="LoRa Mesh Network Monitor API", lifespan=lifespan)
protected = APIRouter(dependencies=[Depends(get_current_user)])

# Registrar router de autenticação (sem proteção JWT)
app.include_router(auth_router)

# CORS só para origens listadas em CORS_ORIGINS. O token vai no cabeçalho
# Authorization, então não há cookies a compartilhar com o navegador.
_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

# ==========================================
# Middleware de autenticação JWT
# ==========================================
# Rotas públicas que NÃO exigem autenticação
PUBLIC_PATHS = {"/", "/docs", "/openapi.json", "/redoc", "/api/auth/login", "/health"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Valida token JWT em todas as rotas protegidas."""
    path = request.url.path

    # Rotas públicas e preflight CORS (OPTIONS sem token) não passam pelo middleware
    if path in PUBLIC_PATHS or request.method == "OPTIONS":
        return await call_next(request)

    # Extrair token do header Authorization: Bearer <token>
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return JSONResponse(
            status_code=401,
            content={"detail": "Token não fornecido"},
        )

    token = auth_header.split(" ", 1)[1]
    try:
        username = decode_access_token(token)
        if not username:
            return JSONResponse(
                status_code=401,
                content={"detail": "Token inválido"},
            )
        # Disponibilizar username no request state
        request.state.username = username
    except InvalidTokenError:
        return JSONResponse(
            status_code=401,
            content={"detail": "Token inválido ou expirado"},
        )

    return await call_next(request)


# ==========================================
# Rotas da API (REST)
# ==========================================

@protected.get("/nodes", response_model=list[Node])
def get_nodes(session: Session = Depends(get_session)):
    """Retorna lista de todos os nós conhecidos"""
    return session.exec(select(Node)).all()

@protected.get("/history/{node_id}", response_model=list[Telemetry])
def get_history(
    node_id: int,
    limit: int = Query(50, ge=1, le=5000),
    network_id: Optional[int] = None,
    session: Session = Depends(get_session)
):
    """Retorna histórico recente de um nó específico.
    network_id valida que o nó pertence à rede.
    """
    # Validar pertinência à rede se network_id fornecido
    if network_id is not None:
        node = session.get(Node, node_id)
        if node and node.network_id != network_id:
            raise HTTPException(status_code=404, detail="Nó não pertence à rede selecionada")

    statement = select(Telemetry).where(Telemetry.node_id == node_id).order_by(Telemetry.timestamp.desc()).limit(limit)
    return session.exec(statement).all()


# ==========================================
# Endpoints do modo de simulação
# ==========================================

# Duração fixa por tipo de anomalia (s), calibrada para o pipeline de detecção
ANOMALY_DURATIONS = {
    "A1": 1800, "A2": 1800, "A3": 480, "A4": 1800,
    "A5": 1800, "A6": 1800, "A7": 1800,
}

ANOMALY_DURATION_MIN = 900
ANOMALY_DURATION_MAX = 3600


def _calculate_duration(anomaly: str, duration_override) -> int:
    """Duração efetiva: A3 fixo; nos demais, o override é limitado a [ANOMALY_DURATION_MIN, ANOMALY_DURATION_MAX]."""
    if anomaly == "A3":
        return ANOMALY_DURATIONS["A3"]
    if duration_override is not None:
        return max(ANOMALY_DURATION_MIN, min(ANOMALY_DURATION_MAX, duration_override))
    return ANOMALY_DURATIONS[anomaly]


class SimulateRequest(BaseModel):
    anomaly: str
    duration_override: Optional[int] = None


@protected.post("/api/nodes/{node_id}/simulate", dependencies=[Depends(get_current_admin)])
def start_simulation(node_id: int, body: SimulateRequest, session: Session = Depends(get_session)):
    """
    Inicia a simulação de anomalia em um nó.
    O nó recebe o comando SIM_START por downlink LoRa e passa a emitir
    beacons sintéticos com o perfil da anomalia.
    """
    node = session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Nó {node_id} não encontrado")
    if node.is_gateway:
        raise HTTPException(status_code=400, detail="Não é possível simular no gateway")
    if node.status == "offline":
        raise HTTPException(status_code=400, detail=f"Nó {node_id} está offline")
    if node.sim_mode:
        raise HTTPException(status_code=400, detail=f"Nó {node_id} já está em modo de simulação")

    anomaly = body.anomaly.upper()
    if anomaly not in ANOMALY_DURATIONS:
        raise HTTPException(status_code=400, detail=f"Anomalia inválida: {anomaly}. Use A1..A7")

    # Guard A6/A7: vizinhos suficientes
    if anomaly in ("A6", "A7"):
        from models import Neighbor as NeighborModel
        neighbor_count = session.exec(
            select(func.count()).select_from(NeighborModel)
            .where(NeighborModel.node_id == node_id)
            .where(NeighborModel.timestamp >= datetime.utcnow() - timedelta(minutes=5))
        ).one()
        if anomaly == "A7" and neighbor_count < 2:
            raise HTTPException(
                status_code=400,
                detail=f"A7 requer pelo menos 2 vizinhos ativos — nó tem {neighbor_count}",
            )
        if anomaly == "A6" and neighbor_count < 1:
            raise HTTPException(
                status_code=400,
                detail="A6 requer pelo menos 1 vizinho ativo — nó está isolado",
            )

    duration = _calculate_duration(anomaly, body.duration_override)

    # Criar SimulationRun
    from models import SimulationRun
    run = SimulationRun(
        node_id=node_id,
        anomaly=anomaly,
        started_at=datetime.utcnow(),
        duration_planned=duration,
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    # Atualizar estado do nó
    node.sim_mode = True
    node.sim_anomaly = anomaly
    node.sim_started_at = datetime.utcnow()
    node.sim_ended_at = None
    node.current_sim_run_id = run.id
    session.add(node)
    session.commit()

    mqtt_payload = {
        "cmd": "send",
        "dest": node.lora_address,
        "msg": f"SIM_START:{anomaly}:{duration}",
        "msg_id": 9039,
    }
    published = publish_to_downlink(mqtt_payload, node.network_id)
    logger.info(f"🧪 SIM START: Node {node_id}, anomaly={anomaly}, duration={duration}s, run_id={run.id}")

    if not published:
        run.ended_at = datetime.utcnow()
        run.ended_reason = "firmware_rejected"
        node.sim_mode = False
        node.sim_anomaly = None
        node.sim_started_at = None
        node.current_sim_run_id = None
        session.add(run); session.add(node); session.commit()
        raise HTTPException(status_code=503, detail="Broker MQTT indisponível")

    return {
        "status": "ok",
        "node_id": node_id,
        "anomaly": anomaly,
        "duration": duration,
        "started_at": node.sim_started_at.isoformat(),
        "sim_run_id": run.id,
    }


@protected.delete("/api/nodes/{node_id}/simulate", dependencies=[Depends(get_current_admin)])
def stop_simulation(node_id: int, session: Session = Depends(get_session)):
    """Para simulação e fecha SimulationRun."""
    node = session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Nó {node_id} não encontrado")
    if not node.sim_mode:
        raise HTTPException(status_code=400, detail=f"Nó {node_id} não está em modo de simulação")

    mqtt_payload = {
        "cmd": "send",
        "dest": node.lora_address,
        "msg": "SIM_STOP",
        "msg_id": 9040,
    }
    if not publish_to_downlink(mqtt_payload, node.network_id):
        # Sem publicação, o firmware pode continuar simulando. Preservar o estado
        # permite ao operador tentar novamente sem perder o intervalo de referência.
        raise HTTPException(status_code=503, detail="Broker MQTT indisponível")

    from models import SimulationRun
    if node.current_sim_run_id:
        run = session.get(SimulationRun, node.current_sim_run_id)
        if run:
            _close_simulation_run(run, "operator_stop", session)
    else:
        # Sem SimulationRun vinculado (estado de versão anterior ou ID perdido): só encerra no nó
        node.sim_mode = False
        node.sim_ended_at = datetime.utcnow()
        session.add(node)
        session.commit()

    logger.info(f"🧪 SIM STOP: Node {node_id}")
    return {"status": "ok", "node_id": node_id}


@protected.get("/api/nodes/{node_id}/simulate")
def get_simulation_status(node_id: int, session: Session = Depends(get_session)):
    """Estado da simulação de um nó."""
    node = session.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Nó {node_id} não encontrado")

    result = {
        "node_id": node_id,
        "sim_mode": node.sim_mode,
        "anomaly": node.sim_anomaly,
        "started_at": node.sim_started_at.isoformat() if node.sim_started_at else None,
        "ended_at": node.sim_ended_at.isoformat() if node.sim_ended_at else None,
    }
    if node.sim_mode and node.sim_started_at:
        from models import SimulationRun
        elapsed = (datetime.utcnow() - node.sim_started_at).total_seconds()
        run = session.get(SimulationRun, node.current_sim_run_id) if node.current_sim_run_id else None
        # A duração planejada do run inclui o duration_override do pedido.
        duration = run.duration_planned if run else ANOMALY_DURATIONS.get(node.sim_anomaly, 300)
        result["elapsed_seconds"] = int(elapsed)
        result["duration"] = duration
    return result


@protected.get("/api/simulation/active")
def get_active_simulation(session: Session = Depends(get_session)):
    """Retorna simulação ativa com telemetry_count e sim_run_id."""
    node = session.exec(select(Node).where(Node.sim_mode == True)).first()
    if not node:
        return {"active": False}

    from models import SimulationRun
    elapsed = (datetime.utcnow() - node.sim_started_at).total_seconds() if node.sim_started_at else 0
    duration = ANOMALY_DURATIONS.get(node.sim_anomaly, 1800)

    telemetry_count = 0
    sim_run_id = node.current_sim_run_id
    if sim_run_id:
        run = session.get(SimulationRun, sim_run_id)
        if run:
            telemetry_count = run.telemetry_count
            duration = run.duration_planned

    return {
        "active": True,
        "node_id": node.node_id,
        "node_name": node.name,
        "lora_address": node.lora_address,
        "anomaly": node.sim_anomaly,
        "started_at": node.sim_started_at.isoformat() if node.sim_started_at else None,
        "elapsed_seconds": int(elapsed),
        "duration": duration,
        "telemetry_count": telemetry_count,
        "sim_run_id": sim_run_id,
    }


@protected.get("/api/simulation/run/{run_id}")
def get_simulation_run(run_id: int, session: Session = Depends(get_session)):
    """Resultado de uma execução de simulação (pós-encerramento)."""
    from models import SimulationRun
    run = session.get(SimulationRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"SimulationRun {run_id} não encontrado")

    # Latência total/ponta-a-ponta (comando → detecção). Embute o atraso
    # do firmware (~60-90s) até passar a emitir a anomalia.
    ttd_seconds = None
    if run.detected_at and run.started_at:
        ttd_seconds = int((run.detected_at - run.started_at).total_seconds())

    # Latência do detector (onset → detecção), sem o atraso do firmware.
    # None se o onset não foi registrado.
    detector_latency_seconds = None
    if run.detected_at and run.anomaly_onset_at:
        detector_latency_seconds = int(
            (run.detected_at - run.anomaly_onset_at).total_seconds()
        )

    duration_actual = None
    if run.ended_at and run.started_at:
        duration_actual = int((run.ended_at - run.started_at).total_seconds())

    return {
        "id": run.id,
        "node_id": run.node_id,
        "anomaly": run.anomaly,
        "started_at": run.started_at.isoformat() + "Z",
        "ended_at": run.ended_at.isoformat() + "Z" if run.ended_at else None,
        "duration_planned": run.duration_planned,
        "duration_actual": duration_actual,
        "ended_reason": run.ended_reason,
        "telemetry_count": run.telemetry_count,
        "anomaly_onset_at": run.anomaly_onset_at.isoformat() + "Z" if run.anomaly_onset_at else None,
        "detected_at": run.detected_at.isoformat() + "Z" if run.detected_at else None,
        "ttd_seconds": ttd_seconds,
        "detector_latency_seconds": detector_latency_seconds,
        "detector_primary": run.detector_primary,
        "detector_secondary": run.detector_secondary,
        "detector_combo": run.detector_combo,
    }


@protected.get("/topology")
def get_topology(network_id: Optional[int] = None, session: Session = Depends(get_session)):
    """Retorna a topologia da rede usando build_topology, filtrada por rede."""
    topology = build_topology(session, network_id=network_id)
    if network_id is not None:
        network = session.get(Network, network_id)
        topology["network_name"] = network.name if network else None
    return topology


# ==========================================
# CRUD de Redes LoRa Mesh
# ==========================================

def _build_network_response(network: Network, session: Session) -> NetworkResponse:
    all_nodes = session.exec(select(Node).where(Node.network_id == network.id)).all()
    gateways = [DeviceResponse.from_node(n) for n in all_nodes if n.is_gateway]
    devices = [DeviceResponse.from_node(n) for n in all_nodes if not n.is_gateway]
    # Contexto completo quando os 5 campos obrigatórios estão preenchidos
    _basic_fields = [
        network.environment_type, network.building_type,
        network.obstacle_density, network.typical_link_distance_class,
        network.availability_criticality,
    ]
    context_complete = all(f is not None for f in _basic_fields)
    return NetworkResponse(
        id=network.id,
        name=network.name,
        description=network.description,
        active=network.active,
        gateways=gateways,
        devices=devices,
        created_at=network.created_at,
        updated_at=network.updated_at,
        environment_type=network.environment_type,
        building_type=network.building_type,
        floors_count=network.floors_count,
        predominant_obstacle_material=network.predominant_obstacle_material,
        obstacle_density=network.obstacle_density,
        typical_link_distance_class=network.typical_link_distance_class,
        availability_criticality=network.availability_criticality,
        avg_install_height_m=network.avg_install_height_m,
        avg_walls_per_link=network.avg_walls_per_link,
        node_mobility_profile=network.node_mobility_profile,
        environment_context_complete=context_complete,
    )


def _get_next_node_id(session: Session) -> int:
    """Retorna o próximo node_id disponível (max atual + 1)."""
    result = session.exec(select(func.max(Node.node_id))).first()
    return (result or 0) + 1


def _normalize_lora(addr: str) -> str:
    """Normaliza endereço LoRa: 0x + hex maiúsculo (ex: '0xab12' → '0xAB12')."""
    return "0x" + addr[2:].upper() if addr.startswith(("0x", "0X")) else addr.upper()


def _validate_lora_addresses(addresses: list[str], exclude_network_id: Optional[int], session: Session):
    """Valida que os endereços LoRa não estão em uso em outra rede."""
    for addr in addresses:
        query = select(Node).where(Node.lora_address == addr)
        if exclude_network_id is not None:
            query = query.where(Node.network_id != exclude_network_id)
        existing = session.exec(query).first()
        if existing:
            raise HTTPException(status_code=409, detail=f"Endereço {addr} já cadastrado em outra rede")


@protected.post("/api/networks", response_model=NetworkResponse, status_code=201,
                dependencies=[Depends(get_current_admin)])
def create_network(request: NetworkCreate, session: Session = Depends(get_session)):
    """Cria nova rede com gateways e end devices."""
    # Validar unicidade do nome
    existing = session.exec(select(Network).where(Network.name == request.name)).first()
    if existing:
        raise HTTPException(status_code=409, detail="Já existe uma rede com este nome")

    # Validar device_model obrigatório para gateways
    for gw in request.gateways:
        if not gw.device_model:
            raise HTTPException(status_code=422, detail=f"Informe o modelo do gateway '{gw.name}'")

    # Normalizar e validar endereços LoRa únicos globalmente
    all_addresses = [_normalize_lora(d.lora_address) for d in request.gateways + request.devices]
    if len(all_addresses) != len(set(all_addresses)):
        raise HTTPException(status_code=422, detail="Endereços LoRa duplicados na requisição")
    _validate_lora_addresses(all_addresses, exclude_network_id=None, session=session)

    # Criar rede
    network = Network(
        name=request.name,
        description=request.description,
        environment_type=request.environment_type,
        building_type=request.building_type,
        floors_count=request.floors_count,
        predominant_obstacle_material=request.predominant_obstacle_material,
        obstacle_density=request.obstacle_density,
        typical_link_distance_class=request.typical_link_distance_class,
        availability_criticality=request.availability_criticality,
        avg_install_height_m=request.avg_install_height_m,
        avg_walls_per_link=request.avg_walls_per_link,
        node_mobility_profile=request.node_mobility_profile,
    )
    session.add(network)
    session.commit()
    session.refresh(network)

    # Criar gateways
    for gw in request.gateways:
        addr = _normalize_lora(gw.lora_address)
        existing_node = session.exec(select(Node).where(func.lower(Node.lora_address) == addr.lower())).first()
        if existing_node:
            existing_node.network_id = network.id
            existing_node.is_gateway = True
            existing_node.device_model = gw.device_model
            existing_node.name = gw.name
            existing_node.battery_type = gw.battery_type
            existing_node.battery_capacity_mah = gw.battery_capacity_mah
            existing_node.power_source = gw.power_source
            session.add(existing_node)
        else:
            node_id = _get_next_node_id(session)
            node = Node(
                node_id=node_id,
                name=gw.name,
                lora_address=addr,
                network_id=network.id,
                is_gateway=True,
                device_model=gw.device_model,
                battery_type=gw.battery_type,
                battery_capacity_mah=gw.battery_capacity_mah,
                power_source=gw.power_source,
                status="offline",
                last_seen=datetime(2000, 1, 1),
            )
            session.add(node)
            session.commit()  # commit para next_node_id ser sequencial

    # Criar end devices
    for dev in request.devices:
        addr = _normalize_lora(dev.lora_address)
        existing_node = session.exec(select(Node).where(func.lower(Node.lora_address) == addr.lower())).first()
        if existing_node:
            existing_node.network_id = network.id
            existing_node.is_gateway = False
            existing_node.name = dev.name
            existing_node.battery_type = dev.battery_type
            existing_node.battery_capacity_mah = dev.battery_capacity_mah
            existing_node.power_source = dev.power_source
            session.add(existing_node)
        else:
            node_id = _get_next_node_id(session)
            node = Node(
                node_id=node_id,
                name=dev.name,
                lora_address=addr,
                network_id=network.id,
                is_gateway=False,
                battery_type=dev.battery_type,
                battery_capacity_mah=dev.battery_capacity_mah,
                power_source=dev.power_source,
                status="offline",
                last_seen=datetime(2000, 1, 1),
            )
            session.add(node)
            session.commit()

    session.commit()
    session.refresh(network)
    return _build_network_response(network, session)


@protected.get("/api/networks", response_model=List[NetworkResponse])
def list_networks(include_inactive: bool = False, session: Session = Depends(get_session)):
    """Lista redes ativas (ou todas com ?include_inactive=true). Ordenado por id."""
    query = select(Network)
    if not include_inactive:
        query = query.where(Network.active == True)
    query = query.order_by(Network.id)
    networks = session.exec(query).all()
    return [_build_network_response(n, session) for n in networks]


@protected.get("/api/networks/{network_id}", response_model=NetworkResponse)
def get_network(network_id: int, session: Session = Depends(get_session)):
    """Retorna rede com seus dispositivos."""
    network = session.get(Network, network_id)
    if not network:
        raise HTTPException(status_code=404, detail="Rede não encontrada")
    return _build_network_response(network, session)


@protected.put("/api/networks/{network_id}", response_model=NetworkResponse,
               dependencies=[Depends(get_current_admin)])
def update_network(network_id: int, request: NetworkCreate, session: Session = Depends(get_session)):
    """Atualiza rede: nome, descrição e sincroniza dispositivos."""
    network = session.get(Network, network_id)
    if not network:
        raise HTTPException(status_code=404, detail="Rede não encontrada")

    # Validar unicidade do nome (excluindo a própria rede)
    existing = session.exec(
        select(Network).where(Network.name == request.name, Network.id != network_id)
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="Já existe uma rede com este nome")

    # Validar device_model para gateways
    for gw in request.gateways:
        if not gw.device_model:
            raise HTTPException(status_code=422, detail=f"Informe o modelo do gateway '{gw.name}'")

    # Validar endereços únicos globalmente (excluindo própria rede)
    all_addresses = [_normalize_lora(d.lora_address) for d in request.gateways + request.devices]
    if len(all_addresses) != len(set(all_addresses)):
        raise HTTPException(status_code=422, detail="Endereços LoRa duplicados na requisição")
    _validate_lora_addresses(all_addresses, exclude_network_id=network_id, session=session)

    # Atualizar rede
    network.name = request.name
    network.description = request.description
    network.environment_type = request.environment_type
    network.building_type = request.building_type
    network.floors_count = request.floors_count
    network.predominant_obstacle_material = request.predominant_obstacle_material
    network.obstacle_density = request.obstacle_density
    network.typical_link_distance_class = request.typical_link_distance_class
    network.availability_criticality = request.availability_criticality
    network.avg_install_height_m = request.avg_install_height_m
    network.avg_walls_per_link = request.avg_walls_per_link
    network.node_mobility_profile = request.node_mobility_profile
    session.add(network)

    # Desassociar dispositivos removidos
    current_nodes = session.exec(select(Node).where(Node.network_id == network_id)).all()
    new_addresses = set(all_addresses)
    for node in current_nodes:
        if node.lora_address not in new_addresses:
            node.network_id = None
            node.is_gateway = False
            session.add(node)

    # Sincronizar gateways
    for gw in request.gateways:
        addr = _normalize_lora(gw.lora_address)
        existing_node = session.exec(select(Node).where(func.lower(Node.lora_address) == addr.lower())).first()
        if existing_node:
            existing_node.network_id = network_id
            existing_node.is_gateway = True
            existing_node.device_model = gw.device_model
            existing_node.name = gw.name
            existing_node.battery_type = gw.battery_type
            existing_node.battery_capacity_mah = gw.battery_capacity_mah
            existing_node.power_source = gw.power_source
            session.add(existing_node)
        else:
            node_id = _get_next_node_id(session)
            session.add(Node(
                node_id=node_id, name=gw.name, lora_address=addr,
                network_id=network_id, is_gateway=True, device_model=gw.device_model,
                battery_type=gw.battery_type, battery_capacity_mah=gw.battery_capacity_mah,
                power_source=gw.power_source, status="offline",
                last_seen=datetime(2000, 1, 1),
            ))
            session.commit()

    # Sincronizar end devices
    for dev in request.devices:
        addr = _normalize_lora(dev.lora_address)
        existing_node = session.exec(select(Node).where(func.lower(Node.lora_address) == addr.lower())).first()
        if existing_node:
            existing_node.network_id = network_id
            existing_node.is_gateway = False
            existing_node.name = dev.name
            existing_node.battery_type = dev.battery_type
            existing_node.battery_capacity_mah = dev.battery_capacity_mah
            existing_node.power_source = dev.power_source
            session.add(existing_node)
        else:
            node_id = _get_next_node_id(session)
            session.add(Node(
                node_id=node_id, name=dev.name, lora_address=addr,
                network_id=network_id, is_gateway=False,
                battery_type=dev.battery_type, battery_capacity_mah=dev.battery_capacity_mah,
                power_source=dev.power_source, status="offline",
                last_seen=datetime(2000, 1, 1),
            ))
            session.commit()

    session.commit()
    session.refresh(network)
    return _build_network_response(network, session)


@protected.delete("/api/networks/{network_id}", status_code=204, dependencies=[Depends(get_current_admin)])
def deactivate_network(network_id: int, session: Session = Depends(get_session)):
    """Soft delete: marca active=False. Não exclui dados históricos."""
    network = session.get(Network, network_id)
    if not network:
        raise HTTPException(status_code=404, detail="Rede não encontrada")
    network.active = False
    session.add(network)
    session.commit()

    # Liberar modelo ML da rede desativada
    if mqtt_service_sync.anomaly_detector:
        try:
            mqtt_service_sync.anomaly_detector.cleanup_network(network_id)
        except AttributeError:
            pass


# ==========================================
# Endpoints de anomalias (somente detecção por ML)
# ==========================================

@protected.get("/api/anomalies")
def get_anomalies(
    network_id: Optional[int] = None,
    node_id: Optional[int] = None,
    category: Optional[str] = None,
    severity: Optional[str] = None,
    anomaly_type: Optional[str] = None,
    resolved: Optional[bool] = None,
    hours: int = Query(default=24, ge=1, le=8760),
    from_date: Optional[datetime] = Query(None),
    to_date: Optional[datetime] = Query(None),
    limit: int = Query(default=100, ge=1, le=1000),
    session: Session = Depends(get_session)
):
    """
    Lista anomalias detectadas por IA com filtros.
    Aceita network_id para filtrar por rede via JOIN.
    Aceita from_date/to_date para intervalo arbitrário; hours como fallback.
    Retorna as mais recentes primeiro.
    """
    if from_date and to_date:
        time_start, time_end = from_date, to_date
    else:
        time_end = datetime.utcnow()
        time_start = time_end - timedelta(hours=hours)

    query = (
        select(Anomaly)
        .where(Anomaly.timestamp >= time_start, Anomaly.timestamp <= time_end)
        .where(Anomaly.detection_method == "ml")
        .order_by(Anomaly.timestamp.desc())
    )

    # Filtro por rede via JOIN com Node
    if network_id is not None:
        query = query.join(Node, Node.node_id == Anomaly.node_id).where(
            Node.network_id == network_id
        )

    if node_id is not None:
        query = query.where(Anomaly.node_id == node_id)
    if category is not None:
        query = query.where(Anomaly.category == category)
    if severity is not None:
        query = query.where(Anomaly.severity == severity)
    if anomaly_type is not None:
        query = query.where(Anomaly.anomaly_type == anomaly_type)
    if resolved is not None:
        query = query.where(Anomaly.resolved == resolved)
    
    query = query.limit(limit)
    anomalies = session.exec(query).all()
    
    return [
        {
            "id": a.id,
            "node_id": a.node_id,
            "neighbor_address": a.neighbor_address,
            "timestamp": a.timestamp.isoformat(),
            "anomaly_type": a.anomaly_type,
            "category": a.category,
            "severity": a.severity,
            "description": a.description,
            "suggested_action": a.suggested_action,
            "evidence": json.loads(a.evidence) if a.evidence else {},
            "detection_method": a.detection_method,
            "confidence": a.confidence,
            "resolved": a.resolved,
            "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
            "notes": a.notes,
            "detector_algorithm": a.detector_algorithm,
            "detector_combo": a.detector_combo,
            "detector_primary": a.detector_primary,
            "detector_secondary": a.detector_secondary,
            **_synthetic_fields(a),
            "sim_run_id": a.sim_run_id,
        }
        for a in anomalies
    ]


@protected.get("/api/anomalies/stats")
def get_anomaly_stats(
    hours: int = Query(default=24, ge=1, le=8760),
    from_date: Optional[datetime] = Query(None),
    to_date: Optional[datetime] = Query(None),
    network_id: Optional[int] = None,
    session: Session = Depends(get_session)
):
    """
    Estatísticas agregadas de anomalias.
    Aceita network_id para filtrar por rede.
    Aceita from_date/to_date; hours como fallback.
    Retorna contagens por severidade, categoria, tipo e nó.
    """
    if from_date and to_date:
        time_start, time_end = from_date, to_date
    else:
        time_end = datetime.utcnow()
        time_start = time_end - timedelta(hours=hours)

    # Filtro por rede via JOIN
    query = (
        select(Anomaly)
        .where(Anomaly.timestamp >= time_start, Anomaly.timestamp <= time_end)
        .where(Anomaly.detection_method == "ml")
    )
    if network_id is not None:
        query = query.join(Node, Node.node_id == Anomaly.node_id).where(
            Node.network_id == network_id
        )
    anomalies = session.exec(query).all()

    total = len(anomalies)
    
    # Agregar por severidade
    by_severity = {"high": 0, "medium": 0, "low": 0}
    by_category = {}
    by_type = {}
    by_node = {}
    active_count = 0
    
    for a in anomalies:
        by_severity[a.severity] = by_severity.get(a.severity, 0) + 1
        by_category[a.category] = by_category.get(a.category, 0) + 1
        by_type[a.anomaly_type] = by_type.get(a.anomaly_type, 0) + 1
        by_node[a.node_id] = by_node.get(a.node_id, 0) + 1
        if not a.resolved:
            active_count += 1
    
    # Nó mais afetado
    most_affected = None
    if by_node:
        most_node = max(by_node, key=by_node.get)
        most_affected = {"node_id": most_node, "count": by_node[most_node]}
    
    # Tipo mais comum
    most_common = None
    if by_type:
        most_type = max(by_type, key=by_type.get)
        most_common = {"type": most_type, "count": by_type[most_type]}
    
    # Estatísticas do detector em memória
    from mqtt_service_sync import anomaly_detector
    detector_stats = anomaly_detector.get_stats() if anomaly_detector else {}
    
    return {
        "period_hours": hours,
        "total": total,
        "active": active_count,
        "resolved": total - active_count,
        "by_severity": by_severity,
        "by_category": by_category,
        "by_type": by_type,
        "by_node": {str(k): v for k, v in by_node.items()},
        "most_affected_node": most_affected,
        "most_common_type": most_common,
        "detector": detector_stats,
    }


@protected.get("/api/anomalies/total")
def anomaly_total(
    network_id: Optional[int] = None,
    session: Session = Depends(get_session),
):
    """
    Retorna o total de anomalias detectadas por IA, opcionalmente filtrado por rede.
    Serve de contador para clientes da API.
    """
    query = select(func.count(Anomaly.id)).where(Anomaly.detection_method == "ml")
    if network_id is not None:
        query = query.join(Node, Node.node_id == Anomaly.node_id).where(
            Node.network_id == network_id
        )
    count = session.exec(query).one()
    return {"total": count}


@protected.patch("/api/anomalies/{anomaly_id}/resolve")
def resolve_anomaly(
    anomaly_id: int,
    notes: Optional[str] = None,
    session: Session = Depends(get_session)
):
    """
    Marca anomalia como resolvida com notas opcionais do operador.
    """
    anomaly = session.get(Anomaly, anomaly_id)
    if not anomaly:
        return {"error": f"Anomalia {anomaly_id} não encontrada"}
    
    anomaly.resolved = True
    anomaly.resolved_at = datetime.utcnow()
    if notes:
        anomaly.notes = notes
    
    session.add(anomaly)
    session.commit()
    
    logger.info(f"✅ Anomalia #{anomaly_id} [{anomaly.anomaly_type}] marcada como resolvida")
    
    return {
        "id": anomaly.id,
        "anomaly_type": anomaly.anomaly_type,
        "resolved": True,
        "resolved_at": anomaly.resolved_at.isoformat(),
        "notes": anomaly.notes,
    }


FAST_RULE_DISCONTINUED = (
    "a camada de regras rápidas (fast_rule) foi descontinuada e não pode ser habilitada."
)


@protected.patch("/api/settings/fast-detection")
def toggle_fast_detection(enabled: bool):
    """Camada rápida descontinuada — habilitação recusada (409), sem persistência."""
    if enabled:
        raise HTTPException(status_code=409, detail=FAST_RULE_DISCONTINUED)
    return {"fast_detection_enabled": False, "deprecated": True}


@protected.get("/api/settings/fast-detection")
def get_fast_detection_status():
    """Estado efetivo sempre desligado."""
    return {"fast_detection_enabled": False, "deprecated": True}


# ==========================================
# Configurações de Parâmetros ML
# ==========================================

class MLSettingsIFUpdate(BaseModel):
    """Atualização parcial dos parâmetros do Isolation Forest."""
    contamination: Optional[float] = Field(None, ge=0.01, le=0.20)
    n_estimators: Optional[int] = Field(None, ge=50, le=500)

class MLSettingsOnlineUpdate(BaseModel):
    """Seleção de algoritmos na detecção online."""
    primary: Optional[str] = Field(
        None, description="id do algoritmo primário (lstm_ae, o padrão, ou isolation_forest)"
    )
    secondary: Optional[str] = Field(
        None, description="id do algoritmo secundário; None = single"
    )
    combo: Optional[Literal["single", "cascade", "shadow", "fusion"]] = Field(
        None, description="cascade e fusion são experimentais; shadow registra o secundário sem decidir"
    )
    fusion_alpha: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator("primary", "secondary", mode="before")
    @classmethod
    def validate_algo_id(cls, v):
        if v is None:
            return v
        try:
            from ai.online.registry import get as registry_get
            try:
                registry_get(v)
            except KeyError:
                # PATCH pode ser a primeira operação após um startup parcial;
                # garante o mesmo registro lazy usado pelo GET de settings.
                import ai.online.pipeline  # noqa: F401
                registry_get(v)
        except KeyError:
            raise ValueError(
                f"Algoritmo {v!r} não registrado. "
                "Use GET /api/settings/ml para ver disponíveis."
            )
        return v


class MLSettingsLSTMUpdate(BaseModel):
    """Parâmetros explícitos do LSTM-AE (mudança gera novo candidato).

    Campos desconhecidos são rejeitados (422): valores sem suporte não podem ser
    aceitos e ignorados. ``threshold_strategy`` só aceita a estratégia implementada
    e permite reparar um valor legado inválido persistido (ex.: 'dynamic').
    """
    model_config = ConfigDict(extra="forbid")

    window_size: Optional[int] = Field(None, ge=2, le=48)
    history_hours: Optional[float] = Field(None, gt=0, le=720)
    hidden_size: Optional[int] = Field(None, ge=4, le=256)
    epochs: Optional[int] = Field(None, ge=1, le=1000)
    patience: Optional[int] = Field(None, ge=1, le=100)
    score_aggregation: Optional[Literal["mean", "last", "max"]] = None
    threshold_percentile: Optional[float] = Field(None, gt=50, lt=100)
    threshold_strategy: Optional[Literal["percentile"]] = None


_LSTM_FIELD_TO_COLUMN = {
    "window_size": "ml_lstm_window",
    "history_hours": "ml_lstm_history_hours",
    "hidden_size": "ml_lstm_hidden",
    "epochs": "ml_lstm_epochs",
    "patience": "ml_lstm_patience",
    "score_aggregation": "ml_lstm_score_aggregation",
    "threshold_percentile": "ml_lstm_threshold_percentile",
    "threshold_strategy": "ml_lstm_threshold_strategy",
}


def _lstm_settings_block(ml) -> dict:
    """Valores persistidos (None = default efetivo; ver effective_config)."""
    return {fld: getattr(ml, col, None) for fld, col in _LSTM_FIELD_TO_COLUMN.items()}


def _effective_config_block(ml) -> dict:
    from ai.online.effective_config import resolve_effective_config
    try:
        return resolve_effective_config(ml).to_dict()
    except ValueError as exc:
        return {"error": str(exc)}


class MLSettingsUpdate(BaseModel):
    """Schema de atualização parcial para detecção contínua."""
    isolation_forest: Optional[MLSettingsIFUpdate] = None
    fast_detection: Optional[bool] = None  # obsoleto; true é recusado (409)
    ml_detect_simulated: Optional[bool] = None
    # Flag global de modo de classificação de A1/A2
    anomaly_classification_mode: Optional[Literal["contextual", "absolute"]] = None
    # Seleção de algoritmos online
    online: Optional[MLSettingsOnlineUpdate] = None
    # Parâmetros do LSTM-AE
    lstm: Optional[MLSettingsLSTMUpdate] = None


def _get_available_algorithms() -> list[dict]:
    """Retorna algoritmos online registrados; garante registro lazy se lifespan falhou."""
    from ai.online.registry import available as _avail
    algos = _avail()
    if not algos:
        try:
            import ai.online.pipeline  # noqa: F401 - registra os algoritmos padrão
            algos = _avail()
        except Exception as e:
            logger.warning(f"_get_available_algorithms: falha ao registrar defaults: {e}")
    return [{"id": s.id, "label": s.label, "input_kind": s.input_kind} for s in algos]


def _online_selection_block(ml) -> dict:
    """Seleção online persistida, com o padrão e o estado experimental das combinações."""
    from ai.online.effective_config import DEFAULT_ONLINE_PRIMARY, EXPERIMENTAL_COMBOS
    combo = getattr(ml, "ml_online_combo", "single")
    return {
        "primary": getattr(ml, "ml_online_primary", DEFAULT_ONLINE_PRIMARY),
        "secondary": getattr(ml, "ml_online_secondary", None),
        "combo": combo,
        "fusion_alpha": getattr(ml, "ml_online_fusion_alpha", 0.5),
        "default_primary": DEFAULT_ONLINE_PRIMARY,
        # Cascata e fusão não mostraram vantagem sobre o LSTM-AE sozinho no autoteste.
        "experimental_combos": list(EXPERIMENTAL_COMBOS),
        "experimental": combo in EXPERIMENTAL_COMBOS,
    }


@protected.get("/api/settings/ml")
def get_ml_settings(session: Session = Depends(get_session)):
    """Retorna parâmetros ML da detecção contínua."""
    from models import MLSettings
    from ai.anomaly_definitions import RULES_VERSION, DEFINITIONS_VERSION, definition_fingerprint
    from ai.anomaly_types import get_catalog_info
    ml = session.get(MLSettings, 1)
    if not ml:
        ml = MLSettings(id=1)
        session.add(ml)
        session.commit()
        session.refresh(ml)
    return {
        "isolation_forest": {
            "contamination": ml.ml_if_contamination,
            "n_estimators": ml.ml_if_n_estimators,
        },
        "fast_detection": False,              # Descontinuada
        "fast_detection_deprecated": True,
        "ml_detect_simulated": ml.ml_detect_simulated,
        "anomaly_classification_mode": getattr(ml, "anomaly_classification_mode", "contextual"),
        "classification": {
            "rules_version": RULES_VERSION,
            "definitions_version": DEFINITIONS_VERSION,
            "definitions_fingerprint": definition_fingerprint(),
            "catalog": get_catalog_info(),
        },
        # Seleção online atual + algoritmos disponíveis
        "online_selection": _online_selection_block(ml),
        "available_algorithms": _get_available_algorithms(),
        # Parâmetros LSTM persistidos e configuração efetiva resolvida
        "lstm": _lstm_settings_block(ml),
        "effective_config": _effective_config_block(ml),
        "updated_at": ml.updated_at.isoformat() + "Z" if ml.updated_at else None,
    }


@protected.get("/api/ml/online/status")
def get_online_ml_status(_user=Depends(get_current_user)):
    """Modelos publicados, treino em curso, τ e último ciclo por rede."""
    from mqtt_service_sync import anomaly_detector
    if not anomaly_detector:
        raise HTTPException(status_code=503, detail="detector indisponível")
    return anomaly_detector.online_status()


@protected.post("/api/ml/online/retrain")
def post_online_ml_retrain(
    network_id: Optional[int] = None,
    session: Session = Depends(get_session),
    _admin=Depends(get_current_admin),
):
    """Retreino manual fora do ciclo (ignora congelamento e espera).

    O pipeline atual segue detectando até o candidato ficar pronto; o novo pacote
    é salvo em ML_ONLINE_BUNDLE_DIR.
    """
    from mqtt_service_sync import anomaly_detector
    from models import MLSettings, Network
    if not anomaly_detector:
        raise HTTPException(status_code=503, detail="detector indisponível")
    if network_id is not None:
        network_ids = [network_id]
    else:
        network_ids = [n.id for n in session.exec(select(Network).where(Network.active == True)).all()]  # noqa: E712
    ml = session.get(MLSettings, 1) or MLSettings(id=1)
    return {"started": anomaly_detector.request_online_retrain(ml, network_ids)}


@protected.patch("/api/settings/ml")
def patch_ml_settings(
    body: MLSettingsUpdate,
    session: Session = Depends(get_session),
    _admin = Depends(get_current_admin),
):
    """
    Atualiza parâmetros ML (parcial).
    Apenas os campos enviados são modificados.
    Dispara retreino se parâmetros de modelo mudaram.
    """
    from mqtt_service_sync import anomaly_detector
    from models import MLSettings

    # Recusa antes de qualquer escrita, inclusive a criação de MLSettings.
    if body.fast_detection:
        raise HTTPException(status_code=409, detail=FAST_RULE_DISCONTINUED)

    ml = session.get(MLSettings, 1)
    if not ml:
        ml = MLSettings(id=1)
        session.add(ml)
        session.commit()
        session.refresh(ml)

    # Aplicar campos enviados
    if body.isolation_forest:
        if body.isolation_forest.contamination is not None:
            ml.ml_if_contamination = body.isolation_forest.contamination
        if body.isolation_forest.n_estimators is not None:
            ml.ml_if_n_estimators = body.isolation_forest.n_estimators

    # Controle de nós simulados na detecção contínua
    if body.ml_detect_simulated is not None:
        ml.ml_detect_simulated = body.ml_detect_simulated

    # Flag de modo de classificação de anomalias
    if body.anomaly_classification_mode is not None:
        ml.anomaly_classification_mode = body.anomaly_classification_mode

    # Seleção de algoritmos online.
    # Usa model_fields_set para distinguir "campo omitido" de "secondary: null"
    # explícito — só assim o estado final realmente solicitado é mesclado/validado.
    if body.online:
        _online_set = body.online.model_fields_set
        if "primary" in _online_set and body.online.primary is not None:
            ml.ml_online_primary = body.online.primary
        if "secondary" in _online_set:
            ml.ml_online_secondary = body.online.secondary  # honra null explícito
        if "combo" in _online_set and body.online.combo is not None:
            ml.ml_online_combo = body.online.combo
        if "fusion_alpha" in _online_set and body.online.fusion_alpha is not None:
            ml.ml_online_fusion_alpha = body.online.fusion_alpha

    # Parâmetros explícitos do LSTM-AE
    if body.lstm:
        for fld, col in _LSTM_FIELD_TO_COLUMN.items():
            val = getattr(body.lstm, fld)
            if val is not None:
                setattr(ml, col, val)

    # Valida o estado final mesclado da seleção online.
    _combo = ml.ml_online_combo
    _sec = ml.ml_online_secondary
    if _combo in ("cascade", "shadow", "fusion") and _sec is None:
        raise HTTPException(422, f"combo {_combo!r} exige um algoritmo secundário.")
    if _combo == "single" and _sec is not None:
        raise HTTPException(422, "combo 'single' exige secundário nulo.")
    if _sec is not None and _sec == ml.ml_online_primary:
        raise HTTPException(422, "primário e secundário devem ser algoritmos distintos.")

    # Rejeita seleção de algoritmo não disponível (ex.: lstm_ae quando
    # ML_LSTM_ENABLED=false). Evita persistir estado que quebraria o build do pipeline.
    _available_ids = {a["id"] for a in _get_available_algorithms()}
    if ml.ml_online_primary not in _available_ids:
        raise HTTPException(422, f"algoritmo primário {ml.ml_online_primary!r} indisponível neste ambiente.")
    if _sec is not None and _sec not in _available_ids:
        raise HTTPException(422, f"algoritmo secundário {_sec!r} indisponível neste ambiente.")

    # Valida o estado final efetivo antes de gravar — valores sem
    # suporte não podem ser aceitos e ignorados.
    from ai.online.effective_config import resolve_effective_config
    try:
        _effective = resolve_effective_config(ml)
    except ValueError as exc:
        session.rollback()
        raise HTTPException(422, f"configuração ML inválida: {exc}")

    session.add(ml)
    session.commit()
    session.refresh(ml)

    # Aplicar ao detector em memória e agendar retreino
    retrain_triggered = False
    if anomaly_detector:
        retrain_triggered = anomaly_detector.apply_settings(ml)

    return {
        "isolation_forest": {
            "contamination": ml.ml_if_contamination,
            "n_estimators": ml.ml_if_n_estimators,
        },
        "fast_detection": False,              # Descontinuada
        "fast_detection_deprecated": True,
        "ml_detect_simulated": ml.ml_detect_simulated,
        "anomaly_classification_mode": getattr(ml, "anomaly_classification_mode", "contextual"),
        "online_selection": _online_selection_block(ml),
        "lstm": _lstm_settings_block(ml),
        "effective_config": _effective.to_dict(),
        "updated_at": ml.updated_at.isoformat() + "Z" if ml.updated_at else None,
        "retrain_triggered": retrain_triggered,
    }


# Colunas que identificam qual modelo (ou combinação) emitiu a anomalia e se ela veio de uma
# simulação registrada; sem elas, as linhas de cada modelo em shadow ficam indistinguíveis.
DETECTOR_EXPORT_FIELDS = ("detector_algorithm", "detector_combo", "detector_primary",
                          "detector_secondary", "classification_mode_used", "sim_run_id")
SYNTHETIC_EXPORT_FIELDS = ("is_synthetic", "synthetic_basis")


def _detector_fields(a) -> dict:
    return {**{name: getattr(a, name, None) for name in DETECTOR_EXPORT_FIELDS},
            **_synthetic_fields(a)}


def _synthetic_fields(a) -> dict:
    """``is_synthetic`` e a base da marcação.

    Fontes: o run de simulação vinculado e as linhas sintéticas contadas no contexto
    da decisão (``evidence.data_origin``). Registro antigo, sem a contagem e sem run,
    fica ``None`` (não verificado), e não ``False``.
    """
    try:
        evidence = json.loads(a.evidence) if a.evidence else {}
    except (TypeError, ValueError):
        evidence = {}
    origin = evidence.get("data_origin") if isinstance(evidence, dict) else None
    rows = origin.get("synthetic_rows") if isinstance(origin, dict) else None
    basis = [name for name, hit in (("sim_run", a.sim_run_id is not None),
                                    ("window_rows", bool(rows))) if hit]
    if basis:
        return {"is_synthetic": True, "synthetic_basis": "+".join(basis)}
    if rows is None:
        return {"is_synthetic": None, "synthetic_basis": "not_recorded"}
    return {"is_synthetic": False, "synthetic_basis": "window_rows"}


@protected.get("/api/anomalies/export")
def export_anomalies(
    format: str = Query(default="csv", pattern="^(csv|json)$"),
    hours: int = Query(default=168, ge=1, le=8760),
    from_date: Optional[datetime] = Query(None),
    to_date: Optional[datetime] = Query(None),
    network_id: Optional[int] = None,
    session: Session = Depends(get_session)
):
    """
    Exporta anomalias em CSV ou JSON para análise.
    network_id/network_name incluídos em cada registro.
    Aceita from_date/to_date; hours como fallback. Padrão: última semana (168h).
    """
    if from_date and to_date:
        time_start, time_end = from_date, to_date
    else:
        time_end = datetime.utcnow()
        time_start = time_end - timedelta(hours=hours)

    query = (
        select(Anomaly)
        .where(Anomaly.timestamp >= time_start, Anomaly.timestamp <= time_end)
        .where(Anomaly.detection_method == "ml")
        .order_by(Anomaly.timestamp.asc())
    )
    if network_id is not None:
        query = query.join(Node, Node.node_id == Anomaly.node_id).where(
            Node.network_id == network_id
        )
    anomalies = session.exec(query).all()

    # Cache de network_id/name por node_id para o export
    node_network_cache: dict[int, tuple[int | None, str | None]] = {}
    def get_node_network(node_id: int):
        if node_id not in node_network_cache:
            n = session.get(Node, node_id)
            if n and n.network_id:
                net = session.get(Network, n.network_id)
                node_network_cache[node_id] = (n.network_id, net.name if net else None)
            else:
                node_network_cache[node_id] = (None, None)
        return node_network_cache[node_id]

    if format == "json":
        data = []
        for a in anomalies:
            net_id, net_name = get_node_network(a.node_id)
            data.append({
                "id": a.id,
                "timestamp": a.timestamp.isoformat(),
                "node_id": a.node_id,
                "network_id": net_id,
                "network_name": net_name,
                "neighbor_address": a.neighbor_address,
                "anomaly_type": a.anomaly_type,
                "category": a.category,
                "severity": a.severity,
                "description": a.description,
                "suggested_action": a.suggested_action,
                "evidence": json.loads(a.evidence) if a.evidence else {},
                "detection_method": a.detection_method,
                "confidence": a.confidence,
                "resolved": a.resolved,
                "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
                "notes": a.notes,
                **_detector_fields(a),
            })
        return StreamingResponse(
            io.BytesIO(json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=anomalies_{hours}h.json"}
        )

    # CSV. Texto livre começando com =, +, - ou @ vira fórmula em planilhas; o
    # apóstrofo inicial faz a planilha tratá-lo como texto.
    def _cell(text):
        return f"'{text}" if text and text[0] in "=+-@\t\r" else (text or "")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "timestamp", "node_id", "network_id", "network_name",
        "anomaly_type", "category", "severity", "description",
        "suggested_action", "detection_method", "confidence",
        "resolved", "resolved_at", "notes", "evidence", *DETECTOR_EXPORT_FIELDS,
        *SYNTHETIC_EXPORT_FIELDS, "neighbor_address",
    ])

    for a in anomalies:
        net_id, net_name = get_node_network(a.node_id)
        writer.writerow([
            a.id, a.timestamp.isoformat(), a.node_id, net_id or "", _cell(net_name),
            a.anomaly_type, a.category, a.severity, _cell(a.description), _cell(a.suggested_action),
            a.detection_method, a.confidence, a.resolved,
            a.resolved_at.isoformat() if a.resolved_at else "",
            _cell(a.notes), a.evidence or "",
            *("" if v is None else v for v in _detector_fields(a).values()),
            a.neighbor_address or "",
        ])
    
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=anomalies_{hours}h.csv"}
    )



@app.get("/")
async def root():
    """Rota raiz (pública: não expõe configuração)."""
    return {"message": "LoRa Mesh Network Monitor API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check para monitoramento externo (sem autenticação).

    ``degraded`` quando o broker está desconectado ou o detector não subiu: a API
    responder não basta para dizer que a ingestão e a detecção estão rodando.
    Expõe só estados, sem endereço do broker nem mensagens de erro.
    """
    from mqtt_service_sync import service_state
    mqtt_ok = bool(service_state["mqtt_connected"])
    detector = service_state["detector"]
    return {
        "status": "ok" if mqtt_ok and detector == "running" else "degraded",
        "mqtt_connected": mqtt_ok,
        "detector": detector,
        "timestamp": datetime.utcnow().isoformat(),
    }


app.include_router(protected)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.API_HOST, port=settings.API_PORT, reload=False, log_level="info")
