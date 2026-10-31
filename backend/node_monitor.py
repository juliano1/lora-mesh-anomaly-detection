"""
Serviço de Monitoramento de Nós - Health Check
Verifica periodicamente se os nós estão ativos ou inativos
"""
import asyncio
import logging
from datetime import datetime, timedelta
from sqlmodel import Session, select
import json

from database import engine
from models import Node
from config import settings

logger = logging.getLogger(__name__)

class NodeMonitor:
    """Monitor de saúde dos nós da rede"""

    def __init__(self):
        self.running = False
        self.mqtt_publish_func = None
        self.loop = None

    def set_mqtt_publisher(self, publish_func):
        """Define a função para publicar no MQTT (compatível com paho-mqtt)"""
        self.mqtt_publish_func = publish_func

    async def start(self, mqtt_client_placeholder=None):
        """Inicia o monitoramento periódico"""
        self.running = True
        self.loop = asyncio.get_running_loop()
        logger.info(f"🩺 Node Monitor iniciado (timeout: {settings.NODE_TIMEOUT_SECONDS}s)")
        
        while self.running:
            try:
                await self.check_nodes_health()
                await asyncio.sleep(settings.HEALTH_CHECK_INTERVAL)
            except Exception as e:
                logger.error(f"Erro no health check: {e}")
                await asyncio.sleep(settings.HEALTH_CHECK_INTERVAL)
    
    def stop(self):
        """Para o monitoramento"""
        self.running = False
        logger.info("🛑 Node Monitor parado")
    
    async def check_nodes_health(self):
        """Verifica o status de todos os nós"""
        timeout_threshold = datetime.utcnow() - timedelta(seconds=settings.NODE_TIMEOUT_SECONDS)
        
        # Executar operação de banco (bloqueante) em uma thread separada
        await asyncio.to_thread(self._sync_check_nodes, timeout_threshold)

    def _sync_check_nodes(self, timeout_threshold):
        """Lógica síncrona de verificação de banco"""
        with Session(engine) as session:
            # Buscar todos os nós
            all_nodes = session.exec(select(Node)).all()
            
            changed_nodes = []
            
            for node in all_nodes:
                old_status = node.status
                
                # Verificar se o nó está offline (sem dados recentes)
                if node.last_seen < timeout_threshold:
                    new_status = "offline"
                else:
                    new_status = "online"
                
                # Se o status mudou, atualizar
                if old_status != new_status:
                    node.status = new_status
                    session.add(node)
                    changed_nodes.append({
                        "node": node,
                        "old_status": old_status,
                        "new_status": new_status,
                        "network_id": node.network_id,
                    })

                    logger.warning(
                        f"🔄 Node {node.node_id} ({node.name}): {old_status} → {new_status} "
                        f"(last_seen: {node.last_seen})"
                    )
            
            # Salvar mudanças no banco
            if changed_nodes:
                session.commit()
                
                # IMPORTANTE: Extrair dados dos nós antes de sair do contexto da sessão
                # para evitar "DetachedInstanceError" ao acessar os objetos em outra thread
                changes_to_notify = []
                for change in changed_nodes:
                    node = change["node"]
                    changes_to_notify.append({
                        "node_id": node.node_id,
                        "node_name": node.name,
                        "old_status": change["old_status"],
                        "new_status": change["new_status"],
                        "last_seen": node.last_seen.isoformat(),
                        "network_id": change["network_id"],
                        "battery_level": node.battery_level,
                    })
                
                # Publicar no MQTT (precisa voltar para o loop async)
                for change_data in changes_to_notify:
                    asyncio.run_coroutine_threadsafe(
                        self.notify_status_change(change_data),
                        self.loop
                    )
    
    async def notify_status_change(self, change_data: dict):
        """Publica a mudança de status no tópico MQTT de status."""
        # Recebe dados simples (dict), não objetos SQLModel, o que evita
        # "DetachedInstanceError"
        
        # Preparar mensagem de status
        status_message = {
            "type": "node_status_change",
            "node_id": change_data["node_id"],
            "node_name": change_data["node_name"],
            "status": change_data["new_status"],
            "last_seen": change_data["last_seen"],
            "network_id": change_data.get("network_id"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Publicar no MQTT (se disponível)
        if self.mqtt_publish_func:
            try:
                mqtt_payload = json.dumps(status_message)
                self.mqtt_publish_func(
                    settings.MQTT_TOPIC_STATUS,
                    mqtt_payload
                )
                logger.info(f"📡 MQTT: Publicado status do Node {change_data['node_id']}")
            except Exception as e:
                logger.error(f"Erro ao publicar status no MQTT: {e}")

# Instância global do monitor
node_monitor = NodeMonitor()
