/******************************************************************************
 * DisplayManager.h - Controle do display OLED do LoRaMesher Laboratory
 *
 * Gerencia o OLED SSD1306 (128x64), que mostra o estado da rede mesh,
 * métricas de qualidade de sinal e informações do gateway.
 *
 * Recursos:
 * - Tela de splash no boot (Rede Mesh LoRa + versão do firmware)
 * - Rotação de telas (automática a cada 5 s ou manual pelo botão)
 * - Tela 1: estado do nó (ID, rotas, bateria, uptime, LoRa, TX/vizinhos ou Wi-Fi/MQTT)
 * - Tela 2: malha (vizinhos, parent com RSSI/SNR, índice de qualidade)
 * - Tela 3: Rede Mesh LoRa + versão do firmware
 * - Contagem regressiva antes do deep sleep
 ******************************************************************************/

#ifndef DISPLAYMANAGER_H
#define DISPLAYMANAGER_H

#include <Arduino.h>
#include <U8g2lib.h>
#include <Wire.h>

// Telas do display
enum DisplayScreen {
    SCREEN_SPLASH = 0,      // Splash do boot
    SCREEN_STATUS = 1,      // Estado do nó e da rede
    SCREEN_NEIGHBORS = 2,   // Malha (vizinhos, parent, RSSI/SNR, índice de qualidade)
    SCREEN_ABOUT = 3,       // Rede Mesh LoRa + versão do firmware
    SCREEN_TOTAL            // Número total de telas (= 4)
};

// Dados de vizinho para desenhar no display
struct NeighborDisplayInfo {
    uint16_t address;       // endereço LoRa
    int rssi;               // RSSI (dBm)
    float snr;              // SNR (dB)
    int hops;               // saltos (metric)
    bool isParent;          // é o nó parent
    uint8_t linkQuality;    // índice de qualidade 0-10 (não é PDR)
};

class DisplayManager {
public:
    /**
     * @brief Construtor
     * @param sda pino de dados I2C
     * @param scl pino de clock I2C
     * @param rst pino de reset do display
     */
    DisplayManager(uint8_t sda, uint8_t scl, uint8_t rst);
    
    /**
     * @brief Inicializa o display OLED
     * @return true se deu certo; false se o display não existe nesta placa
     */
    bool begin();
    
    /**
     * @brief Atualiza o conteúdo do display (chamar periodicamente do loop)
     */
    void update();
    
    /**
     * @brief Mostra o splash (Rede Mesh LoRa + versão do firmware)
     * @param durationMs Tempo de exibição (padrão 3000 ms)
     */
    void showSplash(uint16_t durationMs = 3000);
    
    /**
     * @brief Define os dados do nó
     * @param nodeName Identificador do nó (ex.: "NODE_1")
     * @param loraAddress Endereço LoRa na rede (hex)
     */
    void setNodeInfo(const String& nodeName, uint16_t loraAddress);
    
    /**
     * @brief Define a versão do firmware exibida
     * @param version Versão (ex.: "1.0.0")
     */
    void setFirmwareVersion(const String& version);
    
    /**
     * @brief Atualiza a contagem da tabela de rotas
     * @param routeCount Número de nós alcançáveis
     */
    void setRouteCount(uint8_t routeCount);
    
    /**
     * @brief Atualiza os dados do último pacote recebido
     * @param senderId Endereço do nó de origem
     * @param rssi Intensidade do sinal (dBm)
     * @param snr Relação sinal-ruído (dB)
     * @param payload Conteúdo da mensagem
     * @param sequence Valor usado só para detectar mudança e redesenhar
     */
    void setLastPacket(uint16_t senderId, float rssi, float snr, 
                       const String& payload, uint32_t sequence);
    
    /**
     * @brief Atualiza o estado do gateway (Wi-Fi e MQTT)
     * @param wifiConnected Estado da conexão Wi-Fi
     * @param wifiIP Endereço IP (vazio se desconectado)
     * @param mqttConnected Estado da conexão com o broker MQTT
     */
    void setGatewayStatus(bool wifiConnected, const String& wifiIP, 
                          bool mqttConnected);
    
    /**
     * @brief Define a configuração do rádio LoRa
     * @param sf Spreading factor
     * @param bw Largura de banda (kHz)
     * @param freq Frequência (MHz)
     */
    void setLoRaConfig(uint8_t sf, float bw, float freq);
    
    /**
     * @brief Define o nível da bateria
     * @param percentage Percentual da bateria (0-100)
     * @param voltage Tensão da bateria
     * @param isValid Leitura da bateria é válida
     */
    void setBatteryLevel(uint8_t percentage, float voltage, bool isValid);
    
    /**
     * @brief Liga o indicador de atividade ao receber pacote
     * Não acorda o display; só o botão o acorda
     */
    void flashActivity();
    
    /**
     * @brief Define o estado de alimentação USB (com USB o display fica ligado)
     * @param usb true se há alimentação USB
     */
    void setUSBPowered(bool usb);
    
    /**
     * @brief Define a lista de vizinhos exibida
     * @param neighbors Vetor de dados dos vizinhos
     * @param count Número de vizinhos
     */
    void setNeighbors(NeighborDisplayInfo* neighbors, uint8_t count);
    
    /**
     * @brief Define o contador de pacotes TX
     * @param count Total de pacotes transmitidos
     */
    void setTxCount(uint32_t count);
    
    /**
     * @brief Verifica o botão para trocar de tela ou acordar o display
     * @param buttonPin GPIO do botão
     * @param pressedDurationMs Tempo de pressão para trocar de tela (padrão 1000 ms)
     * @return true se o botão foi mantido pelo tempo exigido (troca de tela)
     */
    bool checkButtonPress(uint8_t buttonPin, uint16_t pressedDurationMs = 1000);
    
    /**
     * @brief Acorda o display do modo de economia
     */
    void wakeUp();
    
    /**
     * @brief Indica se o display está dormindo
     * @return true se o display está em modo de economia
     */
    bool isAsleep() const { return displayAsleep; }
    
    /**
     * @brief Avança para a próxima tela
     * @param manual true se veio do botão (zera o temporizador de sono)
     */
    void nextScreen(bool manual = true);
    
    /**
     * @brief Mostra a contagem regressiva antes do deep sleep
     * @param secondsRemaining Segundos até dormir
     */
    void showSleepCountdown(uint8_t secondsRemaining);
    
    /**
     * @brief Desliga o display (antes do deep sleep)
     */
    void powerOff();
    
    /**
     * @brief Retorna a tela atual
     * @return Tela atual
     */
    DisplayScreen getCurrentScreen() const { return currentScreen; }

private:
    bool initialized;

    // Objeto U8g2 do display (SSD1306 128x64 via I2C)
    U8G2_SSD1306_128X64_NONAME_F_HW_I2C display;
    
    // Pinos do display
    uint8_t pinSDA;
    uint8_t pinSCL;
    uint8_t pinRST;
    
    // Estado da tela atual
    DisplayScreen currentScreen;
    unsigned long lastScreenChange;
    unsigned long splashEndTime;
    bool splashActive;
    
    // Rotação de telas (5 segundos)
    static const uint16_t SCREEN_ROTATION_MS = 5000;
    
    // Indicador de atividade
    bool activityIndicator;
    unsigned long activityTime;
    static const uint16_t ACTIVITY_DURATION_MS = 200;
    
    // Debounce do botão
    unsigned long buttonPressStart;
    bool buttonPressed;
    
    // Gerenciamento de energia do display
    bool displayAsleep;
    unsigned long lastUserInteraction;
    static const uint32_t DISPLAY_TIMEOUT_MS = 120000; // 2 minutos
    unsigned long lastUpdateTime;
    static const uint16_t UPDATE_INTERVAL_MS = 2000;   // Atualiza a cada 2 segundos
    bool usbPowered;  // USB detectado — display fica ligado
    
    // Estado anterior, para detectar mudanças
    uint8_t prevRouteCount;
    bool prevHasReceivedPacket;
    uint32_t prevLastSequence;
    
    // Dados do nó
    String nodeName;
    uint16_t loraAddress;
    uint8_t routeCount;
    
    // Dados do último pacote
    uint16_t lastSenderId;
    float lastRSSI;
    float lastSNR;
    String lastPayload;
    uint32_t lastSequence;
    bool hasReceivedPacket;
    
    // Estado do gateway
    bool isGateway;
    bool wifiConnected;
    String wifiIP;
    bool mqttConnected;
    
    // Configuração LoRa
    uint8_t loraSF;
    float loraBW;
    float loraFreq;
    
    // Dados da bateria
    uint8_t batteryPercentage;
    float batteryVoltage;
    bool batteryValid;
    
    // Versão do firmware
    String fwVersion;
    
    // Vizinhos para exibição
    static const uint8_t MAX_DISPLAY_NEIGHBORS = 10;
    NeighborDisplayInfo displayNeighbors[10];
    uint8_t displayNeighborCount;
    
    // Contador de pacotes TX
    uint32_t txCount;
    
    // Métodos internos de desenho
    void renderSplash();
    void renderStatusScreen();
    void renderNeighborsScreen();  // rede + vizinhos
    void renderAbout();     // Rede Mesh LoRa + versão (boot + loop)
    
    // Métodos auxiliares (drawSignalBar, getSignalQuality, getNodeName e truncateString não são usados pelas telas)
    void drawSignalBar(uint8_t x, uint8_t y, uint8_t width, uint8_t height, 
                       float rssi, float snr);
    String getSignalQuality(float rssi, float snr);
    String getNodeName(uint16_t nodeId);
    String formatUptime();
    String truncateString(const String& str, uint8_t maxLen);
    
    // Limiares de qualidade de sinal
    static constexpr float RSSI_EXCELLENT = -50.0;
    static constexpr float RSSI_GOOD = -70.0;
    static constexpr float RSSI_FAIR = -90.0;
    static constexpr float RSSI_POOR = -110.0;
    
    static constexpr float SNR_EXCELLENT = 10.0;
    static constexpr float SNR_GOOD = 5.0;
    static constexpr float SNR_FAIR = 0.0;
    static constexpr float SNR_POOR = -5.0;
};

#endif // DISPLAYMANAGER_H
