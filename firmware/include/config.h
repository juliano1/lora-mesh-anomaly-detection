/*******************************************************************************
 * CONFIG.H - Configuração de rede e MQTT
 *
 * Objetivo: centraliza a identificação da rede e os tópicos MQTT.
 *           O NETWORK_ID deve ser configurado antes de gravar o firmware
 *           em cada gateway.
 *
 * Suporte a múltiplas redes LoRa Mesh no mesmo servidor.
 *           Cada gateway publica em "lab/{NETWORK_ID}/uplink".
 *           O backend faz wildcard subscribe em "lab/+/uplink".
 *
 * Configuração:
 *   - Opção A (recomendada): definir NETWORK_ID via platformio.ini:
 *       build_flags = -D NETWORK_ID=1
 *   - Opção B: alterar o padrão abaixo antes de compilar. Ele só vale se
 *     NETWORK_ID não vier dos build_flags, e o platformio.ini deste projeto
 *     já define NETWORK_ID=1.
 *
 * Ao cadastrar uma nova rede no backend (POST /api/networks), o ID gerado
 * deve ser informado aqui antes de gravar o gateway dessa rede.
 ******************************************************************************/

#ifndef CONFIG_H
#define CONFIG_H

// ---------------------------------------------------------------------------
// Macro auxiliar para converter número em string literal em tempo de compilação
// Necessário para concatenar NETWORK_ID nos tópicos MQTT.
// Exemplo: STRINGIFY(1) → "1"
// ---------------------------------------------------------------------------
#define _STRINGIFY(x) #x
#define STRINGIFY(x)  _STRINGIFY(x)

// ---------------------------------------------------------------------------
// NETWORK_ID — Identificador da rede LoRa Mesh
//
// Este valor deve corresponder ao ID gerado pelo backend ao cadastrar a rede.
// Se não definido via build_flags, usa 1 como padrão.
// ---------------------------------------------------------------------------
#ifndef NETWORK_ID
    #define NETWORK_ID 1
#endif

// ---------------------------------------------------------------------------
// Tópicos MQTT derivados do NETWORK_ID
//
// MQTT_TOPIC_UPLINK  : Gateway → Backend (telemetria dos nós mesh)
// MQTT_TOPIC_COMMAND : Backend → Gateway (comandos para nós mesh)
// MQTT_TOPIC_RESULT  : Gateway → Backend (resultado de envios)
// ---------------------------------------------------------------------------
#define MQTT_TOPIC_UPLINK   "lab/"        STRINGIFY(NETWORK_ID) "/uplink"
#define MQTT_TOPIC_COMMAND  "lora/mesh/"  STRINGIFY(NETWORK_ID) "/command"
#define MQTT_TOPIC_RESULT   "lab/"        STRINGIFY(NETWORK_ID) "/send_result"

#endif // CONFIG_H
