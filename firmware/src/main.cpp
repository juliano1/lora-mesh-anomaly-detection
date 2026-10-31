/*******************************************************************************
 * LORAMESH LABORATORY - Projeto de estudo de rede mesh com ESP32
 * 
 * Descrição:
 *   Firmware didático para estudar o comportamento da biblioteca LoraMesher.
 *   Atende tanto nós comuns quanto o gateway (ponte).
 * 
 * Hardware suportado:
 *   - Heltec LoRa 32 V2 (SX1276)
 * 
 * Configuração (platformio.ini):
 *   NODE_ID      : identificador único do nó na rede
 *   IS_GATEWAY   : ativa as funções de gateway (ponte Wi-Fi + MQTT)
 *   HELTEC_V2    : seleção de hardware
 * 
 ******************************************************************************/

#include <Arduino.h>
#include <WiFi.h>
#include <Preferences.h>
#include <ArduinoJson.h>
#include "hal_conf.h"
#include "hal/hal_battery.h"
#include "hal/hal_radio.h"
#include "DisplayManager.h"
#include "BatteryManager.h"

// Includes da LoraMesher
#include <LoraMesher.h>

#ifdef IS_GATEWAY
    #include <PubSubClient.h>  // cliente MQTT
    #include "credentials.h"   // credenciais de Wi-Fi e MQTT (fora do controle de versão)
    #include "config.h"        // NETWORK_ID e tópicos MQTT por rede
#endif

/*******************************************************************************
 * CONSTANTES DE CONFIGURAÇÃO
 ******************************************************************************/

// Identidade do nó
#ifndef NODE_ID
    #define NODE_ID 1  // ID padrão se não for definido no build
#endif

String NODE_NAME = String("NODE_") + String(NODE_ID);

// Versão do firmware
#define FW_VERSION "1.9.2"

// Temporização
#define IDENTITY_BROADCAST_INTERVAL     20000   // 20 s (o backend considera o nó offline após 240 s sem dados)
#define ROUTING_TABLE_PRINT_INTERVAL    15000   // 15 s
#define LED_BLINK_DURATION              100     // 100 ms

// Gerenciamento de energia
#define POWER_BUTTON_PIN                0       // botão PRG (botão de boot)
#define SHUTDOWN_HOLD_TIME              3000    // segurar 3 s para desligar

// Hibernação por bateria baixa
#define BATTERY_SLEEP_PCT       10              // Dorme se bateria ≤ 10%
#define BATTERY_SLEEP_VOLTAGE   3.30f           // ou tensão ≤ 3.30V
#define BATTERY_WAKE_PCT        20              // Retoma só se bateria ≥ 20%
#define BATTERY_WAKE_VOLTAGE    3.55f           // ou tensão ≥ 3.55V
#define BATTERY_CONFIRM_READS   2               // Leituras consecutivas exigidas (histerese)
#define BATTERY_SLEEP_RETRY_US  (5ULL * 60ULL * 1000000ULL)  // Reavalia a cada 5 min

// Estimativa de distância (modelo log-distance de perda de percurso)
// d = d0 * 10^((RSSI0 - RSSI) / (10 * n))
// Calibrado com medições reais em ambiente interno (07/02/2026)
//   N2 a 2 m com visada: RSSI=-60 → est. 1.8m ✓  |  N3 a 7 m + parede: RSSI=-78 → est. 7.4m ✓
#define RSSI_REF_1M         -52     // Calibrado: Heltec V2 em ambiente interno a 1 m
#define PATH_LOSS_EXPONENT  3.0f    // n: 3.0 = ambiente interno (calibrado)
#define DISTANCE_REF_M      1.0f    // d0: distância de referência (metros)

// Configuração do gateway (credenciais em credentials.h)
#ifdef IS_GATEWAY
    #define WIFI_TIMEOUT_MS     20000                   // 20 s
    
    // MQTT Client ID — inclui NETWORK_ID para evitar colisão no broker 
    String MQTT_CLIENT_ID = String("LoRaGateway_") + String(NETWORK_ID) + "_" + String(NODE_ID);
    
    // Tópicos MQTT: definidos em config.h a partir de NETWORK_ID
#endif

/*******************************************************************************
 * OBJETOS GLOBAIS
 ******************************************************************************/

// Instância do LoraMesher
LoraMesher& radio = LoraMesher::getInstance();

// Gerenciador do display
DisplayManager display(OLED_SDA, OLED_SCL, OLED_RST);

// Gerenciador de bateria (ADC lido a cada 60 s; a carga varia ~0.1%/min, não precisa ser mais rápido)
BatteryManager battery(60000);

#ifdef IS_GATEWAY
    WiFiClient wifiClient;
    PubSubClient mqttClient(wifiClient);

    // PubSubClient não é seguro entre tarefas: loop() e a tarefa processReceivedPackets
    // usam o mesmo cliente. Mutex recursivo: publishToMQTT também é chamado de dentro do loop().
    SemaphoreHandle_t mqttMutex = nullptr;
    struct MqttLock {
        MqttLock()  { xSemaphoreTakeRecursive(mqttMutex, portMAX_DELAY); }
        ~MqttLock() { xSemaphoreGiveRecursive(mqttMutex); }
    };

    // Nova tentativa de Wi-Fi/MQTT no máximo a cada 5 s: connect() bloqueia o loop.
    #define LINK_RETRY_MS 5000
    unsigned long lastLinkAttempt = 0;
#endif

/*******************************************************************************
 * VARIÁVEIS DE ESTADO
 ******************************************************************************/

unsigned long lastIdentityBroadcast = 0;
unsigned long lastRoutingTablePrint = 0;
unsigned long ledBlinkStart = 0;
bool ledBlinking = false;
bool ledManualState = false;  // Estado persistente — true=aceso por LED_ON
uint32_t sequenceNumber = 0;
uint32_t txPacketCount = 0;  // Total de pacotes transmitidos

// Envio condicional da bateria: só transmite quando o valor muda ou o heartbeat expira.
// Motivo: a bateria varia ~1% a cada 5-10 min; enviá-la em todo beacon é desperdício.
int8_t lastSentBatteryPct = -99;           // Último % de bateria incluído no beacon
unsigned long lastBatterySendTime = 0;     // millis() do último beacon com bateria
#define BATTERY_SEND_INTERVAL   300000     // Heartbeat: envia pelo menos a cada 5 minutos
#define BATTERY_CHANGE_THRESHOLD 2         // Envia se mudou >= 2% desde o último envio
#define BATTERY_CRITICAL_LEVEL   20        // Neste nível ou abaixo: sempre envia (urgente)

// Estado do modo de simulação
bool     simModeActive   = false;
char     simAnomalyType  = 0;       // '1'..'7'
uint8_t  simStep         = 0;       // Contador incrementado a cada beacon
unsigned long simEndTime = 0;       // instante de expiração em millis()
int8_t   simBatteryPct   = -99;     // Para A3: bateria simulada decrescente

// Temporização da atualização do display
unsigned long lastDisplayUpdate = 0;
#define DISPLAY_UPDATE_INTERVAL 1000  // Atualiza a cada 1 segundo

// Estado do botão de energia
unsigned long powerButtonPressStart = 0;

// Estado da hibernação por bateria baixa — a memória RTC persiste no deep sleep
RTC_DATA_ATTR bool  lowBatterySleepMode    = false;  // true = dormindo para preservar a bateria
RTC_DATA_ATTR uint8_t lowBatteryWakeConfirm = 0;     // leituras consecutivas de despertar acima do limiar
static uint8_t  lowBatteryCount    = 0;              // leituras baixas consecutivas no loop normal
static unsigned long lastLowBatteryCheck = 0;        // instante da última verificação de bateria no loop

// Mantém a trava de bateria baixa entre cold boots (a perda de energia zera a memória RTC).
static const char* POWER_PREF_NS = "power";
static const char* LOW_BATT_LOCK_KEY = "low_batt_lock";

static void setLowBatteryPersistentLock(bool enabled) {
    Preferences prefs;
    if (prefs.begin(POWER_PREF_NS, false)) {
        prefs.putBool(LOW_BATT_LOCK_KEY, enabled);
        prefs.end();
    }
}

static bool getLowBatteryPersistentLock() {
    Preferences prefs;
    bool enabled = false;
    if (prefs.begin(POWER_PREF_NS, true)) {
        enabled = prefs.getBool(LOW_BATT_LOCK_KEY, false);
        prefs.end();
    }
    return enabled;
}

// Handle da tarefa de pacotes recebidos
TaskHandle_t receiveLoRaMessage_Handle = NULL;

/*******************************************************************************
 * PROTÓTIPOS DE FUNÇÃO
 ******************************************************************************/

void setupLoRaMesh();
void printRoutingTable();
void sendIdentityBroadcast();
void processReceivedPackets(void*);
void createReceiveMessages();
void blinkLED();
void checkPowerButton();
void enterLowBatterySleep();    // Chamada do loop quando a bateria baixa é confirmada

// Índice de qualidade do enlace (0-10), heurística RSSI/SNR; não é PDR medida
uint8_t calculateLinkQualityIndex(int rssi, float snr);

// Sobrescritas do modo de simulação
struct NeighborInfo;  // declaração antecipada
void applySimulationOverrides(NeighborInfo* nbrs, int count);

#ifdef IS_GATEWAY
    void setupWiFi();
    void setupMQTT();
    void mqttCallback(char* topic, byte* payload, unsigned int length);
    void mqttReconnect();
    void publishToMQTT(const String& data, float rssi, float snr, uint32_t timeOnAir, uint16_t parentId);
    void publishSendResult(int msgId, const char* dest, const char* status, int rttMs = -1, const char* error = nullptr);
    void handleMQTTDownlink(const String& jsonStr);
#endif

/*******************************************************************************
 * HIBERNAÇÃO POR BATERIA BAIXA — funções auxiliares
 * Fazem uma leitura mínima do ADC pelo HAL antes de BatteryManager::begin(),
 * sem a suavização (EMA) do BatteryManager. quickBatteryPct() repete a curva de
 * hal_battery_to_pct() (hal/hal_battery.h); as duas precisam continuar iguais.
 ******************************************************************************/

static float quickReadBatteryVoltage() {
    /* Usa HAL de bateria — implementação correta por variante de hardware.
       hal_battery_init() é idempotente: seguro chamar antes de BatteryManager::begin(). */
    hal_battery_init();
    uint32_t mv = hal_battery_read_mv();
    return mv / 1000.0f;
}

static uint8_t quickBatteryPct(float v) {
    if (v >= 4.2f) return 100;
    if (v >= 4.0f) return 90  + (uint8_t)((v - 4.0f) / 0.2f * 10.0f);
    if (v >= 3.7f) return 50  + (uint8_t)((v - 3.7f) / 0.3f * 40.0f);
    if (v >= 3.4f) return 20  + (uint8_t)((v - 3.4f) / 0.3f * 30.0f);
    if (v >= 3.0f) return (uint8_t)((v - 3.0f) / 0.4f * 20.0f);
    return 0;
}

// Chamada de setup() ao despertar pelo timer em lowBatterySleepMode.
// Verifica se a bateria se recuperou o bastante para o boot completo; senão, dorme de novo.
static void handleLowBatteryWakeup() {
    float    v   = quickReadBatteryVoltage();
    uint8_t  pct = quickBatteryPct(v);
    Serial.printf("[POWER] Low-battery wakeup — Battery: %.2fV (%d%%)\n", v, pct);

    if (v >= BATTERY_WAKE_VOLTAGE || pct >= BATTERY_WAKE_PCT) {
        lowBatteryWakeConfirm++;
        Serial.printf("[POWER] Leitura de recuperação %d/%d\n",
                      lowBatteryWakeConfirm, BATTERY_CONFIRM_READS);
        if (lowBatteryWakeConfirm >= BATTERY_CONFIRM_READS) {
            // BATTERY_CONFIRM_READS leituras seguidas acima do limiar — boot seguro
            lowBatterySleepMode    = false;
            lowBatteryWakeConfirm  = 0;
            setLowBatteryPersistentLock(false);
            Serial.println("[POWER] ✓ Bateria recuperada — boot normal");
            return;  // Segue com o setup() completo
        }
    } else {
        lowBatteryWakeConfirm = 0;  // Zera se esta leitura ainda estava baixa
        Serial.printf("[POWER] Bateria ainda baixa (%.2fV, %d%%) — dormindo 5min\n", v, pct);
    }

    // Ainda não recuperou — dorme de novo
    Serial.flush();
    esp_sleep_enable_timer_wakeup(BATTERY_SLEEP_RETRY_US);
    esp_sleep_enable_ext0_wakeup((gpio_num_t)POWER_BUTTON_PIN, LOW);
    esp_deep_sleep_start();
    // Nunca chega aqui
}

// Chamada do loop() após BATTERY_CONFIRM_READS leituras baixas consecutivas.
void enterLowBatterySleep() {
    // Não hiberna durante nenhuma simulação (a A3 simula descarga de bateria)
    if (simModeActive) return;

    float   v   = battery.getVoltage();
    uint8_t pct = battery.getPercentage();

    // O texto abaixo repete BATTERY_SLEEP_RETRY_US, BATTERY_WAKE_PCT e
    // BATTERY_WAKE_VOLTAGE; atualizar junto se as constantes mudarem.
    Serial.println("\n╔═══════════════════════════════════════════════════════╗");
    Serial.println("║       🔋 BATERIA BAIXA — MODO ECONOMIA ATIVADO        ║");
    Serial.printf( "║  Tensão: %.2fV | Nível: %3d%% %-23s║\n", v, pct, "");
    Serial.println("║  Dormindo 5 min — acorda se bateria ≥ 20% (≥ 3.55V)  ║");
    Serial.println("║  Pressione PRG para forçar boot imediato              ║");
    Serial.println("╚═══════════════════════════════════════════════════════╝");
    Serial.flush();

    lowBatterySleepMode   = true;
    lowBatteryWakeConfirm = 0;
    setLowBatteryPersistentLock(true);

    display.powerOff();
    delay(100);

    esp_sleep_enable_timer_wakeup(BATTERY_SLEEP_RETRY_US);
    esp_sleep_enable_ext0_wakeup((gpio_num_t)POWER_BUTTON_PIN, LOW);
    esp_deep_sleep_start();
    // Nunca chega aqui
}

/*******************************************************************************
 * FUNÇÃO SETUP
 ******************************************************************************/

void setup() {
    // Inicializa a Serial
    Serial.begin(115200);
    while (!Serial && millis() < 3000);  // Espera até 3 s pela Serial
    
    delay(1000);
    
    Serial.println("\n\n");
    Serial.println("╔══════════════════════════════════════════════════════════╗");
    Serial.printf( "║      LoraMesher Laboratory  v%-27s  ║\n", FW_VERSION);
    Serial.println("╚══════════════════════════════════════════════════════════╝");
    
    // Detecta a causa do despertar e trata a hibernação por bateria baixa ANTES da
    // inicialização pesada. Precisa rodar antes de HAL_Init() para evitar loop de boot
    // (o pico de corrente do LoRa + display pode causar brownout com bateria baixa)
    esp_sleep_wakeup_cause_t wakeup_reason = esp_sleep_get_wakeup_cause();

    // Recupera a trava de bateria baixa no cold boot (as variáveis RTC se perdem ao desligar de vez).
    if (!lowBatterySleepMode && getLowBatteryPersistentLock()) {
        lowBatterySleepMode = true;
        lowBatteryWakeConfirm = 0;
        Serial.println("[POWER] Lock persistente de bateria baixa restaurado");
    }

    if (lowBatterySleepMode) {
        if (wakeup_reason == ESP_SLEEP_WAKEUP_EXT0) {
            // Usuário pressionou o botão — boot forçado, qualquer que seja o nível da bateria
            Serial.println("[POWER] Boot forçado via botão — saindo do modo bateria baixa");
            lowBatterySleepMode   = false;
            lowBatteryWakeConfirm = 0;
            setLowBatteryPersistentLock(false);
        } else if (wakeup_reason == ESP_SLEEP_WAKEUP_UNDEFINED) {
            // Cold boot (a energia acabou e voltou, ex.: recarga solar).
            // A bateria precisa ter tensão suficiente para alimentar o ESP32; verifica se
            // está acima do limiar de despertar. Se sim, boot normal. Se não, dorme com
            // timer para tentar depois, sem a inicialização pesada, para evitar loop de
            // brownout.
            float    v   = quickReadBatteryVoltage();
            uint8_t  pct = quickBatteryPct(v);
            Serial.printf("[POWER] Cold boot com lock de bateria baixa — Battery: %.2fV (%d%%)\n", v, pct);
            if (v >= BATTERY_WAKE_VOLTAGE || pct >= BATTERY_WAKE_PCT) {
                // Bateria recuperada — limpa a trava e segue o boot normal
                Serial.println("[POWER] ✓ Bateria recuperada no cold boot — boot normal");
                lowBatterySleepMode   = false;
                lowBatteryWakeConfirm = 0;
                setLowBatteryPersistentLock(false);
            } else {
                // Ainda baixa — dorme de novo sem a inicialização pesada (evita brownout)
                Serial.printf("[POWER] Bateria ainda baixa no cold boot (%.2fV, %d%%) — dormindo 5min\n", v, pct);
                Serial.flush();
                esp_sleep_enable_timer_wakeup(BATTERY_SLEEP_RETRY_US);
                esp_sleep_enable_ext0_wakeup((gpio_num_t)POWER_BUTTON_PIN, LOW);
                esp_deep_sleep_start();
                // Nunca chega aqui
            }
        } else {
            // Despertar pelo timer — verifica se a bateria se recuperou; pode voltar a dormir
            handleLowBatteryWakeup();
            // Só chega aqui com a bateria recuperada (BATTERY_CONFIRM_READS leituras acima de BATTERY_WAKE_*)
        }
    } else if (wakeup_reason == ESP_SLEEP_WAKEUP_EXT0) {
        Serial.println("⚡ Woke up from power button press!");
    }

    // Inicializa a camada de abstração de hardware (HAL)
    HAL_Init();

    // Configura o botão de energia (fonte de despertar)
    pinMode(POWER_BUTTON_PIN, INPUT_PULLUP);
    esp_sleep_enable_ext0_wakeup((gpio_num_t)POWER_BUTTON_PIN, 0); // Desperta com o botão em LOW
    
    // Imprime a configuração do nó
    Serial.println("\n┌─── Node Configuration ────────────────────────────────┐");
    Serial.printf(  "│ Node ID:       %-39s │\n", NODE_NAME.c_str());
    #ifdef IS_GATEWAY
        Serial.println("│ Mode:          GATEWAY (WiFi + MQTT Bridge)           │");
        Serial.printf( "│ Network ID:    %-39d │\n", NETWORK_ID);
    #else
        Serial.println("│ Mode:          REGULAR NODE (Mesh Only)               │");
    #endif
    Serial.println("└───────────────────────────────────────────────────────┘");
    
    // Inicializa a rede LoRa Mesh
    setupLoRaMesh();
    
    // Inicializa o display
    Serial.println("\n┌─── Initializing Display ──────────────────────────────┐");
    if (display.begin()) {
        Serial.println("│ ✓ OLED Display initialized                            │");
        Serial.println("└───────────────────────────────────────────────────────┘");
        
        // Versão antes do splash, que a exibe
        display.setFirmwareVersion(FW_VERSION);
        display.showSplash(3000);  // 3 segundos
        
        // Configura o display com os dados do nó
        display.setNodeInfo(NODE_NAME, radio.getLocalAddress());
        display.setLoRaConfig(LORA_SPREADING, LORA_BANDWIDTH, FREQ_BAND);
        
        #ifdef IS_GATEWAY
            display.setGatewayStatus(false, "", false);  // Atualizado depois do Wi-Fi/MQTT
        #endif
    } else {
        Serial.println("│ ✗ Display initialization failed                       │");
        Serial.println("└───────────────────────────────────────────────────────┘");
    }
    
    // Inicializa o monitor de bateria
    Serial.println("\n┌─── Initializing Battery Monitor ──────────────────────┐");
    battery.begin();
    Serial.println("└───────────────────────────────────────────────────────┘");
    
    #ifdef IS_GATEWAY
        // Inicialização específica do gateway
        mqttMutex = xSemaphoreCreateRecursiveMutex();
        setupWiFi();
        setupMQTT();
        
        // Atualiza o display com o estado do Wi-Fi/MQTT
        display.setGatewayStatus(
            WiFi.status() == WL_CONNECTED,
            WiFi.localIP().toString(),
            mqttClient.connected()
        );
    #endif
    
    Serial.println("\n✓ System Ready! Starting main loop...\n");
    LED_ON();
    delay(500);
    LED_OFF();
}

/*******************************************************************************
 * LOOP PRINCIPAL
 ******************************************************************************/

void loop() {
    unsigned long currentMillis = millis();
    
    // Tarefa 1: beacon de identidade (a cada IDENTITY_BROADCAST_INTERVAL)
    if (currentMillis - lastIdentityBroadcast >= IDENTITY_BROADCAST_INTERVAL) {
        lastIdentityBroadcast = currentMillis;
        sendIdentityBroadcast();
    }
    
    // Tarefa 2: imprime a tabela de rotas (a cada ROUTING_TABLE_PRINT_INTERVAL)
    if (currentMillis - lastRoutingTablePrint >= ROUTING_TABLE_PRINT_INTERVAL) {
        lastRoutingTablePrint = currentMillis;
        printRoutingTable();
    }

    // Tarefa 3: monitor de bateria baixa
    // Verifica a cada 60 s (alinhado ao intervalo de atualização do BatteryManager).
    // Exige BATTERY_CONFIRM_READS leituras consecutivas para evitar disparos falsos.
    // Ignorado com alimentação USB (sem consumo de bateria).
    if (!battery.isUSB() && currentMillis - lastLowBatteryCheck >= 60000) {
        lastLowBatteryCheck = currentMillis;
        if (battery.isValid()) {
            bool isLow = (battery.getVoltage()    <= BATTERY_SLEEP_VOLTAGE ||
                          battery.getPercentage() <= BATTERY_SLEEP_PCT);
            if (isLow) {
                lowBatteryCount++;
                Serial.printf("[POWER] Leitura baixa %d/%d — %.2fV (%d%%)\n",
                              lowBatteryCount, BATTERY_CONFIRM_READS,
                              battery.getVoltage(), battery.getPercentage());
                if (lowBatteryCount >= BATTERY_CONFIRM_READS) {
                    enterLowBatterySleep();  // Não retorna
                }
            } else {
                if (lowBatteryCount > 0) {
                    Serial.println("[POWER] Bateria OK — contador baixo resetado");
                }
                lowBatteryCount = 0;
            }
        }
    }

    // Tarefa 4: atualiza o display (a cada DISPLAY_UPDATE_INTERVAL)
    if (currentMillis - lastDisplayUpdate >= DISPLAY_UPDATE_INTERVAL) {
        lastDisplayUpdate = currentMillis;
        
        // Atualiza o nível da bateria
        battery.update();
        display.setBatteryLevel(
            battery.getPercentage(),
            battery.getVoltage(),
            battery.isValid()
        );
        
        // Repassa o estado de USB ao display (com USB o display fica ligado)
        display.setUSBPowered(battery.isUSB());
        
        // Repassa a contagem de TX ao display
        display.setTxCount(txPacketCount);
        
        // Atualiza a contagem de rotas
        LM_LinkedList<RouteNode>* routes = radio.routingTableListCopy();
        uint8_t routeCount = routes ? routes->getLength() : 0;
        display.setRouteCount(routeCount);
        
        // Monta a lista de vizinhos para o display
        {
            NeighborDisplayInfo dispNeighbors[10];
            uint8_t dispCount = 0;
            uint16_t parentAddr = 0;
            int lowestMetric = 999;
            
            // Primeiro, encontra o parent
            if (routes) {
                for (int i = 0; i < routes->getLength(); i++) {
                    RouteNode* r = (*routes)[i];
                    if (r && r->networkNode.metric < lowestMetric) {
                        lowestMetric = r->networkNode.metric;
                        parentAddr = r->via;
                    }
                }
                // Extrai os vizinhos diretos (metric == 1)
                for (int i = 0; i < routes->getLength() && dispCount < 10; i++) {
                    RouteNode* r = (*routes)[i];
                    if (r && r->networkNode.metric == 1) {
                        dispNeighbors[dispCount].address = r->networkNode.address;
                        dispNeighbors[dispCount].rssi = (int)r->receivedRSSI;
                        dispNeighbors[dispCount].snr = (float)r->receivedSNR;
                        dispNeighbors[dispCount].hops = r->networkNode.metric;
                        dispNeighbors[dispCount].isParent = 
                            (r->via == parentAddr && r->networkNode.address == r->via);
                        dispNeighbors[dispCount].linkQuality = calculateLinkQualityIndex((int)r->receivedRSSI, (float)r->receivedSNR);
                        dispCount++;
                    }
                }
            }
            display.setNeighbors(dispNeighbors, dispCount);
        }
        
        if (routes) delete routes;
        
        #ifdef IS_GATEWAY
            // Atualiza o estado do gateway
            display.setGatewayStatus(
                WiFi.status() == WL_CONNECTED,
                WiFi.localIP().toString(),
                mqttClient.connected()
            );
        #endif
        
        // Desenha o display
        display.update();
    }
    
    // Controle do pisca do LED
    // Ao terminar o blink, restaura o estado manual (não apaga incondicionalmente)
    if (ledBlinking && (currentMillis - ledBlinkStart >= LED_BLINK_DURATION)) {
        digitalWrite(LED_BUILTIN, ledManualState ? HIGH : LOW);
        ledBlinking = false;
    }
    
    // Troca manual de tela (botão pressionado por 1 segundo)
    if (display.checkButtonPress(POWER_BUTTON_PIN, 1000)) {
        display.nextScreen();
    }
    
    // Botão de energia (desliga com pressão longa de SHUTDOWN_HOLD_TIME)
    checkPowerButton();
    
    #ifdef IS_GATEWAY
        // Tarefas do gateway: gerenciamento da conexão Wi-Fi/MQTT
        {
            MqttLock lock;
            if (!mqttClient.connected() && currentMillis - lastLinkAttempt >= LINK_RETRY_MS) {
                lastLinkAttempt = currentMillis;
                if (WiFi.status() != WL_CONNECTED) {
                    WiFi.reconnect();      // roteador ausente no boot ou queda depois
                } else {
                    mqttReconnect();
                }
            }
            mqttClient.loop();  // Processa as mensagens MQTT
        }
    #endif
    
    // Pequena pausa para evitar problemas com o watchdog
    delay(10);
}

/*******************************************************************************
 * CONFIGURAÇÃO DO LORAMESH
 ******************************************************************************/

void setupLoRaMesh() {
    Serial.println("\n┌─── Initializing LoraMesher ───────────────────────────┐");
    
    // Obtém e configura o LoraMesher
    LoraMesher::LoraMesherConfig config = LoraMesher::LoraMesherConfig();
    
    // Preenche pinos e módulo a partir do HAL (específico da placa)
    hal_radio_fill_config(config);

    // Parâmetros de RF — definidos em hal_conf.h.
    // Sem isso a lib usa LM_BAND=869.9 MHz (EU868) e outros defaults incorretos.
    config.freq           = FREQ_BAND;        // frequência (MHz)
    config.bw             = LORA_BANDWIDTH;   // largura de banda (kHz)
    config.sf             = LORA_SPREADING;   // spreading factor
    config.cr             = LORA_CODING_RATE; // coding rate 4/x (padrão da lib: CR4/7)
    config.syncWord       = LORA_SYNC_WORD;   // sync word (padrão da lib: 0x13)
    config.power          = LORA_TX_POWER;    // potência de TX em dBm (padrão da lib: 6 dBm)
    config.preambleLength = LORA_PREAMBLE;    // preâmbulo (símbolos)

    // Inicializa o LoraMesher
    radio.begin(config);
    
    // Cria e registra a tarefa de recepção
    createReceiveMessages();
    radio.setReceiveAppDataTaskHandle(receiveLoRaMessage_Handle);
    
    // Inicia o LoraMesher
    radio.start();
    
    // Ativa o papel de gateway no protocolo LoRaMesher.
    // O papel ROLE_GATEWAY é propagado nos pacotes Hello, o que permite:
    // - aos nós descobrir o gateway via getClosestGateway()
    // - descobrir nós por papel (getBestNodeByRole)
    #ifdef IS_GATEWAY
        radio.addGatewayRole();
        Serial.printf(  "│ ✓ Gateway Role activated (ROLE_GATEWAY)               │\n");
    #endif
    
    Serial.printf(  "│ ✓ Radio initialized successfully                     │\n");
    Serial.printf(  "│ ✓ Local Address: 0x%04X %-28s│\n", radio.getLocalAddress(), "");
    Serial.printf(  "│ ✓ Module:    %-39s │\n", RADIO_CHIP);
    Serial.printf(  "│ ✓ Frequency: %.1f MHz (AU915 SB%d ch%d)%-10s│\n",
                    FREQ_BAND, LORA_AU915_SUBBAND, LORA_AU915_ABS_CHANNEL, "");
    Serial.printf(  "│ ✓ SF%d / BW%.0f kHz / CR4/%d / Pwr %d dBm %-11s│\n",
                    LORA_SPREADING, LORA_BANDWIDTH, LORA_CODING_RATE, LORA_TX_POWER, "");
    Serial.printf(  "│ ✓ SyncWord: 0x%02X %-36s│\n", LORA_SYNC_WORD, "");
    Serial.println("└───────────────────────────────────────────────────────┘");
}

/*******************************************************************************
 * CRIAÇÃO DA TAREFA DE RECEPÇÃO
 ******************************************************************************/

void createReceiveMessages() {
    int res = xTaskCreate(
        processReceivedPackets,
        "Receive App Task",
        8192,   /* no gateway, publishToMQTT usa um StaticJsonDocument<1536> na pilha */
        (void*) 1,
        2,
        &receiveLoRaMessage_Handle);
    
    if (res != pdPASS) {
        Serial.printf("✗ Error: Receive App Task creation gave error: %d\n", res);
    }
}

/*******************************************************************************
 * PROCESSAMENTO DE PACOTES RECEBIDOS
 * Tarefa FreeRTOS que trata os pacotes que chegam da malha
 ******************************************************************************/

void processReceivedPackets(void*) {
    for (;;) {
        // Aguarda notificação
        ulTaskNotifyTake(pdPASS, portMAX_DELAY);
        
        display.flashActivity();  // Indicador visual no display

        // Processa todos os pacotes da fila
        while (radio.getReceivedQueueSize() > 0) {
            // Obtém o pacote como bytes brutos
            AppPacket<uint8_t>* rawPacket = radio.getNextAppPacket<uint8_t>();

            if (!rawPacket) continue;

            // Extrai os dados do pacote
            uint16_t senderId = rawPacket->src;
            uint16_t dst = rawPacket->dst;
            size_t payloadSize = rawPacket->payloadSize;
            if (payloadSize == 0 || payloadSize > LM_MAX_PACKET_SIZE - 9) {
                radio.deletePacket(rawPacket);
                continue;
            }

            // RSSI e SNR deste remetente vêm da tabela de rotas: são os do último Hello
            // recebido dele (a LoRaMesher com patch guarda receivedRSSI por vizinho), não deste pacote
            float rssi = -85.0;  // Padrão se o remetente não está na tabela de rotas
            float snr = 0.0;
            {
                LM_LinkedList<RouteNode>* rtCopy = radio.routingTableListCopy();
                if (rtCopy) {
                    for (int ri = 0; ri < rtCopy->getLength(); ri++) {
                        RouteNode* rn = (*rtCopy)[ri];
                        if (rn && rn->networkNode.address == senderId) {
                            snr = (float)rn->receivedSNR;
                            rssi = (float)rn->receivedRSSI;  // RSSI medido pelo rádio (último Hello)
                            break;
                        }
                    }
                    delete rtCopy;
                }
            }

            // Converte o payload em String
            String payload = "";
            if (payloadSize > 0 && rawPacket->payload) {
                // O payload é um buffer de bytes, sem terminador NUL garantido.
                if (!payload.concat(reinterpret_cast<const char*>(rawPacket->payload), payloadSize)) {
                    radio.deletePacket(rawPacket);
                    continue;
                }
            }

            // Comandos de simulação (SIM_START / SIM_STOP)
            if (payload.length() >= 12 && payload.startsWith("SIM_START:A")
                && payload[11] >= '1' && payload[11] <= '7') {
                if (simModeActive) {
                    Serial.println("| SIM already active — send SIM_STOP first            |");
                } else {
                    simAnomalyType = payload[11];  // '1'..'7'
                    simStep = 0;
                    simBatteryPct = battery.isValid() ? (int8_t)battery.getPercentage() : 80;

                    // Duração: padrão de 300 s, ou o terceiro parâmetro, se houver
                    unsigned long duration = 300000;
                    if (payload.length() > 13 && payload[12] == ':') {
                        duration = (unsigned long)payload.substring(13).toInt() * 1000;
                        if (duration < 60000)   duration = 60000;    // mín. 1 min
                        if (duration > 1800000) duration = 1800000;  // máx. 30 min
                    }
                    simEndTime = millis() + duration;

                    // Guard A6/A7 — pré-validar vizinhos disponíveis
                    bool simGuardPassed = true;
                    if (simAnomalyType == '6' || simAnomalyType == '7') {
                        int activeNeighbors = 0;
                        LM_LinkedList<RouteNode>* rtGuard = radio.routingTableListCopy();
                        if (rtGuard) {
                            for (int gi = 0; gi < rtGuard->getLength(); gi++) {
                                RouteNode* rg = (*rtGuard)[gi];
                                if (rg && rg->networkNode.metric == 1) activeNeighbors++;
                            }
                            delete rtGuard;
                        }
                        if (simAnomalyType == '7' && activeNeighbors < 2) {
                            Serial.println("| SIM REJECT A7: <2 vizinhos ativos                   |");
                            simGuardPassed = false;
                        } else if (simAnomalyType == '6' && activeNeighbors < 1) {
                            Serial.println("| SIM REJECT A6: 0 vizinhos ativos                    |");
                            simGuardPassed = false;
                        }
                    }

                    if (simGuardPassed) {
                        simModeActive = true;
                        Serial.printf("| SIM ACTIVATED: A%c  duration=%lus                     |\n",
                                       simAnomalyType, duration / 1000);
                    }
                }
            }
            else if (payload == "SIM_STOP") {
                simModeActive = false;
                simStep = 0;
                Serial.println("| SIM DEACTIVATED                                      |");
            }

            // Interpretar comandos de LED — executa diretamente sem piscar.
            // Durante a simulação os comandos de LED são ignorados, para não confundir o operador.
            // Demais pacotes piscam normalmente (feedback visual de atividade de rede).
            if (payload == "LED_ON" || payload == "LED_OFF") {
                if (simModeActive) {
                    Serial.println("| LED cmd ignored — node in simulation mode            |");
                } else {
                    ledManualState = (payload == "LED_ON");
                    digitalWrite(LED_BUILTIN, ledManualState ? HIGH : LOW);
                }
            } else {
                blinkLED();
            }
            
            // Tempo no ar estimado pelo tamanho do payload: ~2 ms por byte, aproximação
            // para SF7/BW125 no SX1276 que não considera os parâmetros de rádio configurados.
            uint32_t timeOnAir = payloadSize * 2; // ~2ms por byte a SF7/BW125
            
            // Atualiza o display com os dados do último pacote
            display.setLastPacket(senderId, rssi, snr, payload, senderId);  // sequence = senderId: o display só usa esse valor para detectar mudança
            
            // Imprime o pacote recebido com qualidade de sinal e ToA.
            // "NODE_%d" mostra o endereço LoRa em decimal, não o NODE_ID do remetente.
            Serial.println("┌─── Packet Received ───────────────────────────────────┐");
            Serial.printf(  "│ From:    0x%04X (NODE_%d) %-28s│\n", senderId, senderId, "");
            Serial.printf(  "│ To:      0x%04X %-39s│\n", dst, dst == 0xFFFF ? "(BROADCAST)" : "");
            Serial.printf(  "│ Size:    %-4zu bytes %-32s│\n", payloadSize, "");
            Serial.printf(  "│ Payload: %-45s │\n", payload.c_str());
            Serial.printf(  "│ RSSI:    %.1f dBm %-35s│\n", rssi, "");
            Serial.printf(  "│ SNR:     %.1f dB %-36s│\n", snr, "");
            Serial.printf(  "│ ToA:     ~%lu ms (est.) %-29s│\n", timeOnAir, "");
            
            #ifdef IS_GATEWAY
                // Nó parent (via) a partir da tabela de rotas: parent_id é o próximo salto
                // da rota até o remetente, usado como estimativa de quem encaminhou a mensagem
                uint16_t parentId = 0;
                LM_LinkedList<RouteNode>* routes = radio.routingTableListCopy();
                if (routes) {
                    for (int i = 0; i < routes->getLength(); i++) {
                        RouteNode* route = (*routes)[i];
                        if (route && route->networkNode.address == senderId) {
                            // route->via é o próximo salto PARA ALCANÇAR esse nó.
                            // Se via == senderId, a conexão é direta (metric 1)
                            // e parentId fica 0.
                            if (route->via == senderId) {
                                // Conexão direta - sem nó intermediário
                                parentId = 0;
                                Serial.printf("│ Parent:  Direct (no relay) %-24s│\n", "");
                            } else {
                                // Mensagem retransmitida por outro nó
                                parentId = route->via;
                                Serial.printf("│ Parent:  0x%04X (via) %-31s│\n", parentId, "");
                            }
                            break;
                        }
                    }
                    delete routes; // Libera a memória
                }

                // Encaminha ao MQTT (gateway)
                if (payloadSize > 0) {
                    publishToMQTT(payload, rssi, snr, timeOnAir, parentId);
                }
            #endif
            Serial.println("└───────────────────────────────────────────────────────┘");
            
            // IMPORTANTE: apagar o pacote para liberar a memória
            radio.deletePacket(rawPacket);
        }
    }
}

/*******************************************************************************
 * LISTA DE VIZINHOS
 * Extrai da tabela de rotas todos os vizinhos diretos (metric == 1).
 * Retorna um vetor com endereço, RSSI, SNR e se o vizinho é o parent.
 ******************************************************************************/

struct NeighborInfo {
    uint16_t address;
    int rssi;
    float snr;
    bool is_parent;
    int hops;          // número de saltos (metric) da tabela de rotas
    float distance_m;  // distância estimada em metros (modelo log-distance)
};

/*******************************************************************************
 * ÍNDICE DE QUALIDADE DO ENLACE - heurística baseada em RSSI/SNR
 *
 * Os cortes e penalidades foram escolhidos para este projeto; não vêm de uma
 * referência publicada nem foram calibrados contra entrega medida. O índice não
 * é taxa de entrega (PDR): a perda real é inferida no backend pelas lacunas do
 * contador de sequência.
 ******************************************************************************/

/**
 * Calcula o índice de qualidade do enlace a partir do RSSI e do SNR do vizinho.
 * @param rssi RSSI do vizinho em dBm (ex: -85)
 * @param snr  SNR do vizinho em dB  (ex: 8.5)
 * @return índice 0-10 (10 = sinal forte); não é porcentagem de entrega
 */
uint8_t calculateLinkQualityIndex(int rssi, float snr) {
    int q = 10;

    // Penalidades por RSSI fraco
    if      (rssi < -120) { q = 0; }
    else if (rssi < -115) { q -= 7; }
    else if (rssi < -110) { q -= 5; }
    else if (rssi < -105) { q -= 4; }
    else if (rssi < -100) { q -= 3; }
    else if (rssi <  -95) { q -= 2; }
    else if (rssi <  -90) { q -= 1; }

    // Penalidades por SNR baixo
    if      (snr < -7) { q = 0; }
    else if (snr < -5) { q -= 5; }
    else if (snr < -3) { q -= 4; }
    else if (snr <  0) { q -= 3; }
    else if (snr <  3) { q -= 2; }
    else if (snr <  5) { q -= 1; }

    if (q < 0) q = 0;
    if (q > 10) q = 10;
    return (uint8_t)q;
}

// Estima a distância pelo modelo log-distance de perda de percurso
float estimateDistance(int rssi) {
    if (rssi >= 0 || rssi < -120) return -1.0f;  // RSSI inválido
    float exponent = (float)(RSSI_REF_1M - rssi) / (10.0f * PATH_LOSS_EXPONENT);
    return DISTANCE_REF_M * pow(10.0f, exponent);
}

#define MAX_NEIGHBORS 10

// Máximo de vizinhos no payload JSON do beacon
// LM_MAX_PACKET_SIZE=230, cabeçalho de 9 bytes → payload máximo de 221 bytes
// Cada vizinho ≈ 37 bytes (a,r,s,p,h,l). Envelope com nearest+u+q+b ≈ 70 bytes.
//   2 vizinhos: ~145 bytes (folga de ~75 bytes)
//   3 vizinhos: ~182-222 bytes (pode passar de 221 bytes)
// Medido em laboratório denso (5 nós próximos): com 3 vizinhos, ~25% dos beacons
// passam do limite; o excesso leva à remontagem com menos vizinhos ou ao descarte
// do beacon (ver sendIdentityBroadcast).
// Prioridade: parent primeiro, depois melhor RSSI
#define MAX_NEIGHBORS_IN_BEACON 2

int getNeighborsList(NeighborInfo neighbors[], int maxNeighbors) {
    int count = 0;
    
    // Copia a tabela de rotas
    LM_LinkedList<RouteNode>* routes = radio.routingTableListCopy();
    
    if (routes == nullptr || routes->getLength() == 0) {
        if (routes) delete routes;
        return 0;
    }
    
    // Encontra o parent (menor métrica; no empate, o primeiro da lista)
    uint16_t parentAddress = 0;
    int lowestMetric = 999;
    
    for (int i = 0; i < routes->getLength(); i++) {
        RouteNode* route = (*routes)[i];
        if (route && route->networkNode.metric < lowestMetric) {
            lowestMetric = route->networkNode.metric;
            parentAddress = route->via;
        }
    }
    
    // Extrai os vizinhos diretos (metric == 1)
    for (int i = 0; i < routes->getLength() && count < maxNeighbors; i++) {
        RouteNode* route = (*routes)[i];
        
        if (route && route->networkNode.metric == 1) {
            neighbors[count].address = route->networkNode.address;
            
            // SNR real dos pacotes Hello da LoRaMesher (preenchido no RouteNode)
            int8_t realSNR = route->receivedSNR;
            neighbors[count].snr = (float)realSNR;
            
            // RSSI real dos pacotes Hello (lib com patch para guardar o RSSI por vizinho)
            neighbors[count].rssi = (int)route->receivedRSSI;
            
            // Verifica se é o parent (via == endereço em conexão direta)
            neighbors[count].is_parent = (route->via == parentAddress && 
                                         route->networkNode.address == route->via);
            
            // Número de saltos e distância estimada
            neighbors[count].hops = route->networkNode.metric;
            neighbors[count].distance_m = estimateDistance(neighbors[count].rssi);
            
            count++;
        }
    }
    
    // Limpeza
    delete routes;
    
    return count;
}

/*******************************************************************************
 * MODO DE SIMULAÇÃO — sobrescreve valores do beacon conforme o perfil de anomalia
 * Cada tipo altera campos específicos de NeighborInfo para gerar padrões de sinal
 * sintéticos que acionam o pipeline de detecção (ML/regras).
 ******************************************************************************/

void applySimulationOverrides(NeighborInfo* nbrs, int count) {
    switch (simAnomalyType) {
    case '1':  // A1 — Degradação de sinal: RSSI cai progressivamente, SNR baixo
        for (int i = 0; i < count; i++) {
            nbrs[i].rssi = -112 - (simStep % 15);   // -112 a -126 dBm
            nbrs[i].snr  = -2.0f + (float)(simStep % 5) * 0.5f;  // -2.0 a 0.0 dB
        }
        break;

    case '2':  // A2 — Perda de pacotes: salto na sequência + índice de qualidade baixo
        sequenceNumber += 7;  // Pula 7 números por beacon — o backend vê packet_loss
        // RSSI intacto — sinal ok, mas a entrega colapsou
        // O índice de qualidade é forçado em sendIdentityBroadcast (neighbor["l"] = 1)
        break;

    case '3':  // A3 — Descarga de bateria: queda rápida a partir do valor real
        simBatteryPct = max((int8_t)10, (int8_t)(simBatteryPct - 5));
        break;

    case '4':  // A4 — Enlace instável: RSSI e SNR oscilam muito
        for (int i = 0; i < count; i++) {
            nbrs[i].rssi = (simStep % 2 == 0) ? -55 : -95;     // diferença de 40 dB
            nbrs[i].snr  = (simStep % 2 == 0) ? 8.0f : -3.0f;  // diferença de 11 dB
        }
        break;

    case '5':  // A5 — Jamming: RSSI estável, SNR oscilante, índice de qualidade baixo
        for (int i = 0; i < count; i++) {
            nbrs[i].rssi = -82;  // Estável — a divergência vem do SNR ruim
            nbrs[i].snr  = (simStep % 2 == 0) ? 1.0f : 12.0f;
        }
        break;

    case '6':  // A6 — Degradação do hardware de TX: RSSI inflado cria assimetria
        for (int i = 0; i < count; i++) {
            nbrs[i].rssi = -50;  // Falso: "recebe muito bem" — o vizinho real reporta o valor verdadeiro
        }
        break;

    case '7':  // A7 — Route flapping: alterna o parent entre os dois primeiros vizinhos
        if (count >= 2) {
            nbrs[0].is_parent = (simStep % 2 == 0);
            nbrs[1].is_parent = (simStep % 2 != 0);
        } else {
            Serial.println("| SIM A7: need 2+ neighbors — skipping override       |");
        }
        break;
    }
}

/*******************************************************************************
 * BEACON DE IDENTIDADE
 * Envia o beacon periódico que anuncia o nó na malha
 ******************************************************************************/

void sendIdentityBroadcast() {
    // Obtém a lista de vizinhos
    NeighborInfo neighbors[MAX_NEIGHBORS];
    int neighborCount = getNeighborsList(neighbors, MAX_NEIGHBORS);
    
    // Ordena os vizinhos — parent primeiro, depois pelo melhor RSSI (maior = mais perto de 0)
    // Assim os vizinhos mais importantes entram quando o limite é atingido
    for (int i = 0; i < neighborCount - 1; i++) {
        for (int j = i + 1; j < neighborCount; j++) {
            bool swapNeeded = false;
            // O parent sempre vem primeiro
            if (neighbors[j].is_parent && !neighbors[i].is_parent) {
                swapNeeded = true;
            } else if (!neighbors[j].is_parent && !neighbors[i].is_parent) {
                // Entre os demais, ordena por RSSI (maior = melhor)
                if (neighbors[j].rssi > neighbors[i].rssi) {
                    swapNeeded = true;
                }
            }
            if (swapNeeded) {
                NeighborInfo temp = neighbors[i];
                neighbors[i] = neighbors[j];
                neighbors[j] = temp;
            }
        }
    }
    
    // Limita os vizinhos no beacon para controlar o tamanho do payload
    int serializeCount = neighborCount < MAX_NEIGHBORS_IN_BEACON ? neighborCount : MAX_NEIGHBORS_IN_BEACON;
    
    // Aplica as sobrescritas de simulação antes da serialização
    if (simModeActive) {
        if (millis() >= simEndTime) {
            simModeActive = false;
            simStep = 0;
            Serial.println("| SIM expired — returning to normal                    |");
        } else {
            applySimulationOverrides(neighbors, serializeCount);
        }
    }

    // Tamanho do documento: a raiz tem até 9 chaves (i,a,hw,n,nearest,u,q,b,sim; sim só
    // durante a simulação) e cada vizinho 6 (a,r,s,p,h,l). JSON_OBJECT_SIZE(8) conta 8
    // chaves da raiz; os 128 bytes extras cobrem a restante e as strings copiadas.
    const size_t docSize =
        JSON_OBJECT_SIZE(8) +
        JSON_ARRAY_SIZE(MAX_NEIGHBORS_IN_BEACON) +
        MAX_NEIGHBORS_IN_BEACON * JSON_OBJECT_SIZE(6) +
        128;
    DynamicJsonDocument doc(docSize);
    
    // Identificação do nó (formato compacto)
    String nodeIdStr = "N" + String(NODE_ID);
    doc["i"] = nodeIdStr;  // ID em formato compacto

    // Endereço LoRa local (sem o prefixo "0x", por compactação)
    char addrHex[5];
    snprintf(addrHex, sizeof(addrHex), "%04X", radio.getLocalAddress());
    doc["a"] = addrHex;

    // Identificação da placa usada na telemetria.
    doc["hw"] = HW_VERSION;
    
    // Vetor de vizinhos (um item por enlace)
    JsonArray neighborsArray = doc.createNestedArray("n");
    
    // Vizinho mais próximo (melhor RSSI)
    int bestRSSI = -999;
    uint16_t nearestAddress = 0;
    
    for (int i = 0; i < serializeCount; i++) {
        JsonObject neighbor = neighborsArray.createNestedObject();
        
        // Endereço do vizinho (hex sem "0x")
        char neighAddrHex[5];
        snprintf(neighAddrHex, sizeof(neighAddrHex), "%04X", neighbors[i].address);
        neighbor["a"] = neighAddrHex;
        
        // RSSI e SNR
        neighbor["r"] = neighbors[i].rssi;
        neighbor["s"] = neighbors[i].snr;
        
        // Indica se é o parent
        neighbor["p"] = neighbors[i].is_parent;
        
        // Saltos (métrica da tabela de rotas)
        neighbor["h"] = neighbors[i].hops;
        
        // Índice de qualidade 0-10 — heurística RSSI/SNR
        // Índice forçado em 1 para A2 (perda de pacotes) e A5 (jamming)
        if (simModeActive && (simAnomalyType == '2' || simAnomalyType == '5')) {
            neighbor["l"] = 1;
        } else {
            // A4: índice de qualidade sintético definido pelo perfil de simulação;
            // não é taxa de entrega medida nem calibração de falha física.
            neighbor["l"] = (simModeActive && simAnomalyType == '4')
                ? 3
                : calculateLinkQualityIndex(neighbors[i].rssi, neighbors[i].snr);
        }
        
        // A distância estimada não vai no beacon (economiza ~12 bytes por vizinho);
        // o backend a calcula a partir do RSSI pelo modelo log-distance
        
        // Vizinho mais próximo por RSSI (campo "nearest")
        if (neighbors[i].rssi > bestRSSI) {
            bestRSSI = neighbors[i].rssi;
            nearestAddress = neighbors[i].address;
        }
    }
    
    // Vizinho mais próximo por RSSI
    if (nearestAddress != 0) {
        char nearestHex[5];
        snprintf(nearestHex, sizeof(nearestHex), "%04X", nearestAddress);
        doc["nearest"] = nearestHex;
    }
    
    // Uptime em minutos (não segundos, por compactação)
    doc["u"] = millis() / 60000;
    
    // Número de sequência
    doc["q"] = sequenceNumber++;
    
    // Envio condicional da bateria, para economizar energia.
    // A bateria varia devagar (~1% a cada 5-10 min). Só entra no beacon quando:
    //   - alimentação USB (sem custo) ou bateria crítica (≤ BATTERY_CRITICAL_LEVEL): sempre
    //   - mudou >= BATTERY_CHANGE_THRESHOLD desde o último envio
    //   - venceu o heartbeat (BATTERY_SEND_INTERVAL): garante dado recente no backend
    // Sem o campo, o beacon fica ~5-8 bytes menor → menos tempo de TX → menos energia.
    // A simulação A3 força o campo em todo beacon, com valor decrescente.
    if (simModeActive && simAnomalyType == '3') {
        doc["b"] = simBatteryPct;  // Forçado em todo beacon, com descarga simulada
    } else if (battery.isValid()) {
        int8_t currentPct = battery.isUSB() ? -1 : (int8_t)battery.getPercentage();
        bool shouldSend = false;

        if (currentPct == -1) {
            shouldSend = true;  // USB: sempre envia (sem custo de bateria)
        } else if (currentPct <= BATTERY_CRITICAL_LEVEL) {
            shouldSend = true;  // Bateria crítica: sempre envia, para monitoramento
        } else if (abs(currentPct - lastSentBatteryPct) >= BATTERY_CHANGE_THRESHOLD) {
            shouldSend = true;  // Mudança significativa
        } else if (millis() - lastBatterySendTime >= BATTERY_SEND_INTERVAL) {
            shouldSend = true;  // Heartbeat: pelo menos a cada BATTERY_SEND_INTERVAL
        }

        if (shouldSend) {
            doc["b"] = currentPct;
            lastSentBatteryPct = currentPct;
            lastBatterySendTime = millis();
        }
        // senão: campo "b" omitido → beacon menor; o backend mantém o último valor conhecido
    } else {
        doc["b"] = -2;  // Sensor inválido: envia para diagnóstico
    }

    // Campo indicador de simulação + incrementa o contador de passos
    if (simModeActive) {
        String simField = String("A") + String(simAnomalyType);
        doc["sim"] = simField;
        simStep++;
    }

    String jsonString;
    serializeJson(doc, jsonString);

    // Tempo no ar estimado: só informativo (serial e gw_toa); não controla duty cycle
    size_t packetSize = jsonString.length();
    uint32_t timeOnAir = packetSize * 2; // Estimativa grosseira para SF7 (~2 ms/byte)

    // Valida o tamanho do payload antes de enviar, para evitar truncamento silencioso
    const size_t MAX_LORA_PAYLOAD = LM_MAX_PACKET_SIZE - 9;  // cabeçalho LoRaMesher de 9 bytes
    if (jsonString.length() > MAX_LORA_PAYLOAD) {
        Serial.println("┌─── Beacon Payload Size Warning ───────────────────┐");
        Serial.printf(  "│ ⚠️  WARN: Beacon %d bytes exceeds max %d bytes    │\n",
                        jsonString.length(), MAX_LORA_PAYLOAD);
        Serial.println("│ Action: Removing last neighbor and retrying       │");
        Serial.println("└───────────────────────────────────────────────────┘");

        // Reduz o número de vizinhos e remonta
        if (serializeCount > 1) {
            serializeCount--;

            // Remonta o JSON com menos vizinhos
            doc.clear();
            doc["i"]  = nodeIdStr;
            char addrHex[5];
            snprintf(addrHex, sizeof(addrHex), "%04X", radio.getLocalAddress());
            doc["a"]  = addrHex;
            doc["hw"] = HW_VERSION;

            JsonArray neighborsArray = doc.createNestedArray("n");
            for (int i = 0; i < serializeCount; i++) {
                JsonObject neighbor = neighborsArray.createNestedObject();
                char neighAddrHex[5];
                snprintf(neighAddrHex, sizeof(neighAddrHex), "%04X", neighbors[i].address);
                neighbor["a"] = neighAddrHex;
                neighbor["r"] = neighbors[i].rssi;
                neighbor["s"] = neighbors[i].snr;
                neighbor["p"] = neighbors[i].is_parent;
                neighbor["h"] = neighbors[i].hops;
                // Mantém o perfil A4 ao remontar um beacon menor.
                neighbor["l"] = (simModeActive && simAnomalyType == '4')
                    ? 3
                    : calculateLinkQualityIndex(neighbors[i].rssi, neighbors[i].snr);
            }

            doc["nearest"] = (nearestAddress != 0) ? String(nearestAddress) : String("");
            doc["u"] = millis() / 60000;
            doc["q"] = sequenceNumber++;

            if (battery.isValid()) {
                int8_t currentPct = battery.isUSB() ? -1 : (int8_t)battery.getPercentage();
                if (currentPct == -1 || currentPct <= BATTERY_CRITICAL_LEVEL ||
                    abs(currentPct - lastSentBatteryPct) >= BATTERY_CHANGE_THRESHOLD ||
                    millis() - lastBatterySendTime >= BATTERY_SEND_INTERVAL) {
                    doc["b"] = currentPct;
                    lastSentBatteryPct = currentPct;
                    lastBatterySendTime = millis();
                }
            }

            if (simModeActive) {
                String simField = String("A") + String(simAnomalyType);
                doc["sim"] = simField;
                simStep++;
            }

            serializeJson(doc, jsonString);
            packetSize = jsonString.length();

            if (jsonString.length() > MAX_LORA_PAYLOAD) {
                Serial.println("⚠️  Still too large after retry — dropping beacon!");
                return;
            }
        } else {
            Serial.println("⚠️  Cannot reduce further — dropping beacon!");
            return;
        }
    }

    // Envia em broadcast (descoberta local de vizinhos)
    radio.createPacketAndSend(BROADCAST_ADDR, (uint8_t*)jsonString.c_str(), jsonString.length());
    txPacketCount++;  // Incrementa o contador de TX
    
    // Gateway: publica o próprio beacon no MQTT
    #ifdef IS_GATEWAY
    if (WiFi.status() == WL_CONNECTED && mqttClient.connected()) {
        publishToMQTT(jsonString, 0, 0, timeOnAir, 0);
    }
    #endif
    
    // Nós sensores também enviam o beacon em UNICAST ao gateway.
    // A LoRaMesher roteia pela malha (multi-hop se preciso), para que os dados de
    // vizinhos cheguem ao backend via MQTT; o envio é best-effort, sem ACK.
    #ifndef IS_GATEWAY
    {
        RouteNode* gwRoute = radio.getClosestGateway();
        if (gwRoute != nullptr) {
            uint16_t gwAddr = gwRoute->networkNode.address;
            if (gwAddr != 0 && gwAddr != radio.getLocalAddress()) {
                radio.createPacketAndSend(gwAddr, (uint8_t*)jsonString.c_str(), jsonString.length());
                Serial.printf("│ >> Unicast to GW 0x%04X %-30s│\n", gwAddr, "");
            }
        } else {
            // Gateway fora da tabela de rotas: o beacon saiu só em broadcast.
            // A mensagem cita o PRG, mas o botão só acorda o display, troca a tela ou
            // desliga o nó; a próxima tentativa é no próximo beacon.
            Serial.println("│ ⚠️  No gateway in routing table — beacon as broadcast  │");
            Serial.printf("│ Retrying at next beacon (%d sec) or press PRG     │\n",
                          IDENTITY_BROADCAST_INTERVAL / 1000);
        }
    }
    #endif
    
    Serial.println("┌─── Identity Broadcast ────────────────────────────────┐");
    Serial.printf(  "│ Node:     %-44s │\n", nodeIdStr.c_str());
    Serial.printf(  "│ Address:  0x%-41s │\n", addrHex);
    Serial.printf(  "│ Neighbors: %-3d (sent %-2d of %-2d) %-22s│\n", neighborCount, serializeCount, neighborCount, "");
    
    // Mostra o nível da bateria
    if (battery.isValid()) {
        Serial.printf("│ Battery:  %d%% (%.2fV) %-31s│\n", battery.getPercentage(), battery.getVoltage(), "");
    }
    
    // Lista os vizinhos
    for (int i = 0; i < neighborCount && i < 3; i++) {  // Mostra os 3 primeiros
        char neighAddrHex[5];
        snprintf(neighAddrHex, sizeof(neighAddrHex), "%04X", neighbors[i].address);
        Serial.printf("│   - 0x%s: %ddBm, SNR %.1f, ~%.1fm %s%-10s│\n", 
                     neighAddrHex, neighbors[i].rssi, neighbors[i].snr,
                     neighbors[i].distance_m,
                     neighbors[i].is_parent ? "(parent) " : "",
                     "");
    }
    if (neighborCount > 3) {
        Serial.printf(  "│   ... and %d more %-32s│\n", neighborCount - 3, "");
    }
    
    Serial.printf(  "│ Payload:  %-44s │\n", jsonString.c_str());
    Serial.printf(  "│ Size:     %-4d bytes %-33s│\n", jsonString.length(), "");
    Serial.printf(  "│ ToA:      ~%lu ms (SF7 est.) %-26s│\n", timeOnAir, "");
    
    // Aviso de tamanho do payload (deveria ficar abaixo de 120 bytes em SF9)
    if (jsonString.length() > 120) {
        Serial.println("│ ⚠️  WARNING: Payload > 120 bytes (may not work on SF9+) │");
    }
    
    Serial.println("└───────────────────────────────────────────────────────┘");
    
    blinkLED();
}

/*******************************************************************************
 * MONITOR DA TABELA DE ROTAS
 * Imprime a tabela de rotas interna para estudo da topologia
 ******************************************************************************/

void printRoutingTable() {
    Serial.println("\n╔═══════════════════════════════════════════════════════════════╗");
    Serial.println("║                    ROUTING TABLE SNAPSHOT                     ║");
    Serial.println("╠═══════════════════════════════════════════════════════════════╣");
    Serial.printf( "║ Local Address: 0x%04X                                         ║\n", radio.getLocalAddress());
    Serial.printf( "║ Timestamp: %-11lu ms                                     ║\n", millis());
    Serial.println("╟───────────────────────────────────────────────────────────────╢");
    Serial.println("║  Destination  │  Next Hop  │  Metric                         ║");
    Serial.println("╠═══════════════╪════════════╪═════════════════════════════════╣");
    
    // Copia a tabela de rotas
    LM_LinkedList<RouteNode>* routes = radio.routingTableListCopy();
    int routeCount = 0;
    
    if (routes == nullptr || routes->getLength() == 0) {
        Serial.println("║              No routes in routing table                       ║");
    } else {
        for (int i = 0; i < routes->getLength(); i++) {
            RouteNode* route = (*routes)[i];
            if (route) {
                Serial.printf("║   0x%04X      │  0x%04X    │   %3d                         ║\n",
                    route->networkNode.address,
                    route->via,
                    route->networkNode.metric
                );
                routeCount++;
            }
        }
    }
    
    Serial.println("╠═══════════════════════════════════════════════════════════════╣");
    Serial.printf( "║ Total Routes: %-3d                                            ║\n", routeCount);
    Serial.println("╚═══════════════════════════════════════════════════════════════╝\n");
    
    // Limpeza
    if (routes) delete routes;
}

/*******************************************************************************
 * AUXILIAR DO PISCA DO LED
 ******************************************************************************/

void blinkLED() {
    LED_ON();
    ledBlinking = true;
    ledBlinkStart = millis();
}

/*******************************************************************************
 * VERIFICAÇÃO DO BOTÃO DE ENERGIA
 * Monitora pressão longa no PRG para desligar o nó em deep sleep
 ******************************************************************************/

void checkPowerButton() {
    // Lê o botão (ativo em LOW - pressionado quando 0)
    if (digitalRead(POWER_BUTTON_PIN) == LOW) {
        // Botão pressionado
        if (powerButtonPressStart == 0) {
            // Primeira detecção - inicia o temporizador
            powerButtonPressStart = millis();
        } else {
            // Botão ainda pressionado - verifica se atingiu o limiar
            unsigned long pressDuration = millis() - powerButtonPressStart;
            
            if (pressDuration >= SHUTDOWN_HOLD_TIME) {
                // Sequência de desligamento
                Serial.println("\n╔═══════════════════════════════════════════════════════╗");
                Serial.println("║          🔴 SHUTDOWN REQUESTED - POWER OFF           ║");
                Serial.println("╚═══════════════════════════════════════════════════════╝");
                Serial.println("Entering deep sleep mode...");
                Serial.println("Press PRG button to wake up.\n");
                Serial.flush();
                
                // Contagem regressiva no display (5 segundos)
                for (int i = 5; i > 0; i--) {
                    display.showSleepCountdown(i);
                    LED_TOGGLE();
                    delay(1000);
                }
                LED_OFF();
                
                // Desliga o display
                display.powerOff();
                delay(100);
                
                // Fonte de despertar: EXT0 no GPIO 0 (POWER_BUTTON_PIN)
                // Desperta quando o botão é pressionado (nível LOW)
                esp_sleep_enable_ext0_wakeup((gpio_num_t)POWER_BUTTON_PIN, LOW);
                
                // Entra em deep sleep
                esp_deep_sleep_start();
            }
        }
    } else {
        // Botão solto - zera o temporizador
        if (powerButtonPressStart != 0) {
            powerButtonPressStart = 0;
        }
    }
}

/*******************************************************************************
 * FUNÇÕES DO GATEWAY (ponte Wi-Fi + MQTT)
 ******************************************************************************/

#ifdef IS_GATEWAY

/*******************************************************************************
 * Configuração do Wi-Fi
 ******************************************************************************/

void setupWiFi() {
    Serial.println("\n┌─── Connecting to WiFi ────────────────────────────────┐");
    Serial.printf(  "│ SSID: %-48s │\n", WIFI_SSID);
    
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    
    unsigned long startAttempt = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - startAttempt < WIFI_TIMEOUT_MS) {
        delay(500);
        Serial.print(".");
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n│ ✓ WiFi Connected!                                     │");
        Serial.printf(  "│ IP Address: %-42s │\n", WiFi.localIP().toString().c_str());
        Serial.printf(  "│ RSSI: %-4d dBm %-39s│\n", WiFi.RSSI(), "");
        Serial.println("└───────────────────────────────────────────────────────┘");
    } else {
        Serial.println("\n│ ✗ WiFi Connection Failed!                             │");
        Serial.println("│   Gateway will operate in Mesh-Only mode              │");
        Serial.println("└───────────────────────────────────────────────────────┘");
    }
}

/*******************************************************************************
 * Configuração do MQTT
 ******************************************************************************/

void setupMQTT() {
    // Servidor, callback e buffer sempre configurados: sem Wi-Fi no boot, o loop()
    // conecta depois.
    mqttClient.setBufferSize(2048);  // buffer de envio e recepção: comporta o beacon com os campos gw_*
    mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
    mqttClient.setCallback(mqttCallback);

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("⚠ MQTT adiado: Wi-Fi não conectado; o loop() tenta de novo");
        return;
    }
    
    Serial.println("\n┌─── Connecting to MQTT Broker ─────────────────────────┐");
    Serial.printf(  "│ Broker: %-46s │\n", MQTT_BROKER);
    Serial.printf(  "│ Port:   %-46d │\n", MQTT_PORT);
    
    mqttReconnect();
}

/*******************************************************************************
 * Reconexão MQTT
 ******************************************************************************/

void mqttReconnect() {
    if (WiFi.status() != WL_CONNECTED) return;
    MqttLock lock;
    
    if (!mqttClient.connected()) {
        Serial.print("│ Attempting MQTT connection... ");
        
        bool connected;
        if (strlen(MQTT_USER) > 0) {
            connected = mqttClient.connect(MQTT_CLIENT_ID.c_str(), MQTT_USER, MQTT_PASSWORD);
        } else {
            connected = mqttClient.connect(MQTT_CLIENT_ID.c_str());
        }
        
        if (connected) {
            Serial.println("Connected! │");
            
            // Assina o tópico de downlink
            mqttClient.subscribe(MQTT_TOPIC_COMMAND);
            Serial.printf("│ ✓ Subscribed to: %-37s │\n", MQTT_TOPIC_COMMAND);
            Serial.println("└───────────────────────────────────────────────────────┘");
        } else {
            Serial.printf("Failed (rc=%d) │\n", mqttClient.state());
        }
    }
}

/*******************************************************************************
 * Callback MQTT (mensagens de downlink)
 ******************************************************************************/

void mqttCallback(char* topic, byte* payload, unsigned int length) {
    if (strcmp(topic, MQTT_TOPIC_COMMAND) != 0 || length == 0 || length > 512) return;
    String message;
    if (!message.concat(reinterpret_cast<const char*>(payload), length)) return;
    
    Serial.println("┌─── MQTT Downlink Received ────────────────────────────┐");
    Serial.printf(  "│ Topic:   %-45s │\n", topic);
    Serial.printf(  "│ Message: %-45s │\n", message.c_str());
    Serial.println("└───────────────────────────────────────────────────────┘");
    
    // Repassa para a malha
    handleMQTTDownlink(message);
}

/*******************************************************************************
 * Publicação no MQTT (uplink: malha → MQTT)
 ******************************************************************************/

void publishToMQTT(const String& data, float rssi, float snr, uint32_t timeOnAir, uint16_t parentId) {
    MqttLock lock;
    if (!mqttClient.connected()) return;
    
    // Faz o parse do JSON recebido e acrescenta os metadados do gateway.
    // 1536 bytes comportam beacons com até ~10 vizinhos; se o parse falhar,
    // o payload segue inteiro em "raw_data".
    StaticJsonDocument<1536> doc;
    DeserializationError error = deserializeJson(doc, data);
    
    if (error) {
        // JSON inválido: envia como dado bruto
        doc["raw_data"] = data;
    }

    // parent_id: nó que encaminhou a mensagem
    // parentId 0 significa conexão direta com o gateway
    if (parentId != 0) {
        doc["parent_id"] = parentId;
    }
    // Com parentId == 0, o campo parent_id é omitido
    
    // Metadados do gateway
    doc["gw_id"]  = NODE_NAME;
    doc["gw_hw"]  = HW_VERSION;
    doc["gw_rssi"] = rssi;
    doc["gw_snr"]  = snr;
    doc["gw_toa"]  = timeOnAir;   // tempo no ar estimado, em milissegundos
    doc["gw_timestamp"] = millis();
    
    String jsonOutput;
    serializeJson(doc, jsonOutput);
    
    // Publica no MQTT
    bool published = mqttClient.publish(MQTT_TOPIC_UPLINK, jsonOutput.c_str());
    
    Serial.println("┌─── MQTT Uplink Published ─────────────────────────────┐");
    Serial.printf(  "│ Topic:   %-45s │\n", MQTT_TOPIC_UPLINK);
    Serial.printf(  "│ Payload: %-45s │\n", jsonOutput.c_str());
    Serial.printf(  "│ Status:  %-45s │\n", published ? "✓ Success" : "✗ Failed");
    Serial.println("└───────────────────────────────────────────────────────┘");
}

/*******************************************************************************
 * Publicação do resultado de envio no MQTT
 * Informa ao backend o status de cada envio, para acompanhamento
 ******************************************************************************/

void publishSendResult(int msgId, const char* dest, const char* status, int rttMs, const char* error) {
    MqttLock lock;
    if (!mqttClient.connected()) return;
    
    StaticJsonDocument<256> doc;
    doc["type"] = "send_result";
    doc["msg_id"] = msgId;
    doc["dest"] = dest;
    doc["status"] = status;
    if (rttMs >= 0) doc["rtt_ms"] = rttMs;
    if (error) doc["error"] = error;
    
    String jsonOutput;
    serializeJson(doc, jsonOutput);
    
    mqttClient.publish(MQTT_TOPIC_RESULT, jsonOutput.c_str());
    
    Serial.printf("📬 Send result: msg_id=%d status=%s rtt=%dms\n", msgId, status, rttMs);
}

/*******************************************************************************
 * Downlink MQTT (MQTT → malha)
 * 
 * Comandos aceitos:
 *   cmd="send"      → unicast para um nó (opcionalmente confiável, com ACK)
 *   cmd="broadcast"  → broadcast para todos os nós
 *   cmd="ping"       → teste rápido de conectividade com um nó
 * 
 * Formato JSON:
 *   { "cmd": "send", "dest": "0x0002", "msg": "LED_ON", "reliable": true, "msg_id": 42 }
 *   { "cmd": "broadcast", "msg": "STATUS_REQUEST", "msg_id": 43 }
 *   { "cmd": "ping", "dest": "0x0002", "msg_id": 44 }
 * 
 * Mensagens sem "cmd" ou com comando desconhecido são recusadas (status "failed").
 ******************************************************************************/

// Endereço LoRa de destino: 1–4 dígitos hexadecimais, prefixo 0x opcional,
// sem 0x0000 (inválido) nem 0xFFFF (broadcast, que tem comando próprio).
static bool parseUnicastAddress(const String& text, uint16_t& out) {
    String digits = text;
    if (digits.startsWith("0x") || digits.startsWith("0X")) digits = digits.substring(2);
    if (digits.length() == 0 || digits.length() > 4) return false;
    for (size_t i = 0; i < digits.length(); i++) {
        if (!isxdigit((unsigned char)digits[i])) return false;
    }
    unsigned long value = strtoul(digits.c_str(), NULL, 16);
    if (value == 0 || value == BROADCAST_ADDR) return false;
    out = (uint16_t)value;
    return true;
}

void handleMQTTDownlink(const String& jsonStr) {
    StaticJsonDocument<512> doc;
    DeserializationError error = deserializeJson(doc, jsonStr);
    
    if (error) {
        Serial.println("⚠ Invalid JSON in downlink message");
        return;
    }
    
    // Campos comuns
    String cmd = doc["cmd"] | "";
    String msg = doc["msg"] | "";
    int msgId = doc["msg_id"] | 0;
    
    Serial.println("┌─── MQTT Downlink Command ─────────────────────────────┐");
    Serial.printf(  "│ Command: %-45s │\n", cmd.c_str());
    
    // ── CMD: SEND (unicast para um nó) ─────────────────────────────
    if (cmd == "send") {
        String dest = doc["dest"] | "";
        bool reliable = doc["reliable"] | false;
        
        if (dest.length() == 0) {
            Serial.println("│ ⚠ ERROR: No destination specified                    │");
            Serial.println("└───────────────────────────────────────────────────────┘");
            if (msgId > 0) publishSendResult(msgId, "", "failed", -1, "no_destination");
            return;
        }
        
        uint16_t destAddress = 0;
        if (!parseUnicastAddress(dest, destAddress)) {
            Serial.println("│ ⚠ ERROR: Invalid destination address                 │");
            Serial.println("└───────────────────────────────────────────────────────┘");
            if (msgId > 0) publishSendResult(msgId, dest.c_str(), "failed", -1, "invalid_address");
            return;
        }
        
        // Payload: o texto de msg
        String payload = msg;
        if (payload.length() == 0) {
            // msg vazia: envia um JSON com "cmd" igual a msg (string vazia) e,
            // se houver, o campo "value" do comando
            StaticJsonDocument<256> payloadDoc;
            payloadDoc["cmd"] = msg;
            if (doc.containsKey("value")) payloadDoc["value"] = doc["value"];
            serializeJson(payloadDoc, payload);
        }

        // Valida o tamanho do payload antes de enviar
        const size_t MAX_DOWNLINK_PAYLOAD = LM_MAX_PACKET_SIZE - 9;  // cabeçalho de 9 bytes
        if (payload.length() > MAX_DOWNLINK_PAYLOAD) {
            Serial.printf(  "│ ⚠️  ERROR: Payload %d bytes exceeds max %d bytes   │\n",
                            payload.length(), MAX_DOWNLINK_PAYLOAD);
            Serial.println("└───────────────────────────────────────────────────────┘");
            if (msgId > 0) publishSendResult(msgId, dest.c_str(), "failed", -1, "payload_too_large");
            return;
        }

        Serial.printf(  "│ Dest:    0x%04X %-38s│\n", destAddress, "");
        Serial.printf(  "│ Message: %-45s │\n", payload.c_str());
        Serial.printf(  "│ Mode:    %-45s │\n", reliable ? "RELIABLE (ACK + retry)" : "STANDARD (best-effort)");
        Serial.printf(  "│ Size:    %d bytes (%.1f%% of max) %-23s│\n", payload.length(),
                        (payload.length() * 100.0f / MAX_DOWNLINK_PAYLOAD), "");

        if (reliable) {
            // sendReliable só enfileira (ACK e retentativas correm em segundo plano) e
            // descarta em silêncio destino fora da tabela de rotas: não há confirmação
            // de entrega a relatar, apenas o envio.
            radio.sendReliable(destAddress, (uint8_t*)payload.c_str(), payload.length());

            Serial.println("│ Result:  Reliable packet queued                       │");

            if (msgId > 0) publishSendResult(msgId, dest.c_str(), "sent");
        } else {
            // Envio padrão: best-effort, sem ACK
            // A LoRaMesher também faz o roteamento multi-hop via getNextHop()
            radio.createPacketAndSend(destAddress, (uint8_t*)payload.c_str(), payload.length());
            
            Serial.println("│ Result:  Packet queued for sending                    │");
            
            // Reporta "sent" (não há ACK para confirmar)
            if (msgId > 0) publishSendResult(msgId, dest.c_str(), "sent");
        }
    }
    
    // ── CMD: BROADCAST (todos os nós) ──────────────────────────────
    else if (cmd == "broadcast") {
        String payload = msg;

        // Valida o tamanho do payload também no broadcast
        const size_t MAX_BROADCAST_PAYLOAD = LM_MAX_PACKET_SIZE - 9;
        if (payload.length() > MAX_BROADCAST_PAYLOAD) {
            Serial.printf(  "│ ⚠️  ERROR: Broadcast payload %d bytes exceeds max %d  │\n",
                            payload.length(), MAX_BROADCAST_PAYLOAD);
            Serial.println("└───────────────────────────────────────────────────────┘");
            if (msgId > 0) publishSendResult(msgId, "BROADCAST", "failed", -1, "payload_too_large");
            return;
        }

        Serial.printf(  "│ Dest:    BROADCAST (0xFFFF)                           │\n");
        Serial.printf(  "│ Message: %-45s │\n", payload.c_str());
        Serial.printf(  "│ Size:    %d bytes (%.1f%% of max) %-23s│\n", payload.length(),
                        (payload.length() * 100.0f / MAX_BROADCAST_PAYLOAD), "");

        radio.createPacketAndSend(BROADCAST_ADDR, (uint8_t*)payload.c_str(), payload.length());

        Serial.println("│ Result:  Broadcast sent                               │");

        if (msgId > 0) publishSendResult(msgId, "BROADCAST", "sent");
    }
    
    // ── CMD: PING (teste rápido de conectividade) ──────────────────
    else if (cmd == "ping") {
        String dest = doc["dest"] | "";
        uint16_t destAddress = 0;
        if (!parseUnicastAddress(dest, destAddress)) {
            Serial.println("│ ⚠ ERROR: Invalid destination address                 │");
            Serial.println("└───────────────────────────────────────────────────────┘");
            if (msgId > 0) publishSendResult(msgId, dest.c_str(), "failed", -1, "invalid_address");
            return;
        }

        String pingPayload = "{\"ping\":1}";
        
        Serial.printf(  "│ Ping:    0x%04X %-38s│\n", destAddress, "");
        
        // Como no envio confiável, o ping só é enfileirado; não há RTT a medir aqui.
        radio.sendReliable(destAddress, (uint8_t*)pingPayload.c_str(), pingPayload.length());

        if (msgId > 0) publishSendResult(msgId, dest.c_str(), "sent");
    }
    
    // ── Comando ausente ou desconhecido: não é retransmitido para a malha ──
    else {
        Serial.println("│ ⚠ ERROR: Unknown command                             │");
        Serial.println("└───────────────────────────────────────────────────────┘");
        if (msgId > 0) publishSendResult(msgId, "", "failed", -1, "unknown_command");
        return;
    }
    
    Serial.println("└───────────────────────────────────────────────────────┘");
    blinkLED();
}

#endif // IS_GATEWAY
