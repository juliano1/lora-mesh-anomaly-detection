/******************************************************************************
 * DisplayManager.cpp - Implementação do controle do display OLED
 ******************************************************************************/

#include "DisplayManager.h"

DisplayManager::DisplayManager(uint8_t sda, uint8_t scl, uint8_t rst)
    : initialized(sda != 0xFF && scl != 0xFF && rst != 0xFF),
      display(U8G2_R0, rst, scl, sda),
      pinSDA(sda),
      pinSCL(scl),
      pinRST(rst),
      currentScreen(SCREEN_SPLASH),
      lastScreenChange(0),
      splashEndTime(0),
      splashActive(false),
      activityIndicator(false),
      activityTime(0),
      buttonPressStart(0),
      buttonPressed(false),
      displayAsleep(false),
      lastUserInteraction(0),
      lastUpdateTime(0),
      prevRouteCount(0),
      prevHasReceivedPacket(false),
      prevLastSequence(0),
      nodeName("NODE_?"),
      loraAddress(0),
      routeCount(0),
      lastSenderId(0),
      lastRSSI(0),
      lastSNR(0),
      lastPayload(""),
      lastSequence(0),
      hasReceivedPacket(false),
      isGateway(false),
      wifiConnected(false),
      wifiIP(""),
      mqttConnected(false),
      loraSF(7),
      loraBW(125.0),
      loraFreq(915.0),
      batteryPercentage(0),
      batteryVoltage(0.0f),
      batteryValid(false),
      usbPowered(false),
      displayNeighborCount(0),
      txCount(0)
{
}

bool DisplayManager::begin() {
    if (!initialized) {
        Serial.println("[DISPLAY] Sem OLED neste board — display desabilitado");
        return false;
    }
    #ifdef VEXT_CTRL
        // Liga a alimentação do display (Vext: LOW = ligado nas placas Heltec)
        pinMode(VEXT_CTRL, OUTPUT);
        digitalWrite(VEXT_CTRL, LOW);
        delay(100);  // Aguarda a alimentação do display estabilizar
        Serial.println("[DISPLAY] Display power enabled (Vext ON)");
    #endif
    
    // Inicializa o I2C
    Wire.begin(pinSDA, pinSCL);
    
    // Inicializa o display
    display.begin();
    display.setContrast(80);  // Brilho reduzido para economizar bateria (0-255)
    display.clearBuffer();
    display.sendBuffer();
    
    lastUserInteraction = millis();
    lastUpdateTime = millis();
    
    Serial.println("[DISPLAY] OLED SSD1306 initialized (Low Power Mode)");
    Serial.println("[DISPLAY] Contrast: 80/255 | Timeout: 120s (battery) | Update: 2s");
    return true;
}

void DisplayManager::update() {
    if (!initialized) return;
    unsigned long currentMillis = millis();
    static uint8_t prevRenderedScreen = 255; // Rastrear qual tela foi desenhada por último
    
    // Temporização do splash
    if (splashActive) {
        if (currentMillis >= splashEndTime) {
            splashActive = false;
            currentScreen = SCREEN_STATUS;
            lastScreenChange = currentMillis;
            lastUserInteraction = currentMillis;
        }
        return;  // Não atualiza as outras telas durante o splash
    }
    
    // Display timeout — desabilitado se DISPLAY_ALWAYS_ON definido em build_flags.
    // Sem a flag: dorme após DISPLAY_TIMEOUT_MS em modo bateria (USB = sempre ligado).
    #ifndef DISPLAY_ALWAYS_ON
    if (!displayAsleep && !usbPowered &&
        currentMillis - lastUserInteraction >= DISPLAY_TIMEOUT_MS) {
        displayAsleep = true;
        display.setPowerSave(1);
        Serial.println("[DISPLAY] Display sleeping (battery timeout 2min)");
        return;
    }
    #endif
    
    // Não atualiza com o display dormindo
    if (displayAsleep) {
        return;
    }
    
    // Limita as atualizações a cada UPDATE_INTERVAL_MS (economia de bateria)
    if (currentMillis - lastUpdateTime < UPDATE_INTERVAL_MS) {
        return;
    }
    lastUpdateTime = currentMillis;
    
    // Rotação automática a cada SCREEN_ROTATION_MS (exceto no splash)
    if (currentScreen != SCREEN_SPLASH && 
        currentMillis - lastScreenChange >= SCREEN_ROTATION_MS) {
        nextScreen(false);  // a rotação automática NÃO zera o temporizador de sono
    }
    
    // Expira o indicador de atividade
    if (activityIndicator && currentMillis - activityTime >= ACTIVITY_DURATION_MS) {
        activityIndicator = false;
    }
    
    // Verifica se há mudanças reais a mostrar
    bool hasChanges = (routeCount != prevRouteCount) || 
                      (hasReceivedPacket != prevHasReceivedPacket) ||
                      (lastSequence != prevLastSequence) ||
                      activityIndicator ||
                      (currentScreen != prevRenderedScreen); // Força update se a tela mudou (rotação)
    
    // Atualiza só se houver mudanças ou se a última troca de tela foi há 10 s ou mais
    // Também força atualização se houve interação recente (ex: acabou de acordar)
    bool userActive = (currentMillis - lastUserInteraction < 1000);
    if (!hasChanges && !userActive && (currentMillis - lastScreenChange < 10000)) {
        return;
    }
    
    // Guarda o estado anterior
    prevRouteCount = routeCount;
    prevHasReceivedPacket = hasReceivedPacket;
    prevLastSequence = lastSequence;
    prevRenderedScreen = currentScreen;
    
    // Desenha a tela atual
    display.clearBuffer();
    
    switch (currentScreen) {
        case SCREEN_SPLASH:
            renderSplash();
            break;
        case SCREEN_STATUS:
            renderStatusScreen();
            break;
        case SCREEN_NEIGHBORS:
            renderNeighborsScreen();
            break;
        case SCREEN_ABOUT:
            renderAbout();
            break;
        default:
            break;
    }
    
    display.sendBuffer();
}

void DisplayManager::showSplash(uint16_t durationMs) {
    if (!initialized) return;
    splashActive = true;
    splashEndTime = millis() + durationMs;
    currentScreen = SCREEN_SPLASH;
    lastUserInteraction = millis();
    displayAsleep = false;
    
    display.setPowerSave(0);  // Garante o display ligado
    display.clearBuffer();
    renderSplash();
    display.sendBuffer();
    
    Serial.println("[DISPLAY] Showing splash screen");
}

void DisplayManager::setNodeInfo(const String& nodeName, uint16_t loraAddress) {
    if (!initialized) return;
    this->nodeName = nodeName;
    this->loraAddress = loraAddress;
}

void DisplayManager::setFirmwareVersion(const String& version) {
    if (!initialized) return;
    this->fwVersion = version;
}

void DisplayManager::setRouteCount(uint8_t routeCount) {
    if (!initialized) return;
    this->routeCount = routeCount;
}

void DisplayManager::setLastPacket(uint16_t senderId, float rssi, float snr,
                                   const String& payload, uint32_t sequence) {
    if (!initialized) return;
    this->lastSenderId = senderId;
    this->lastRSSI = rssi;
    this->lastSNR = snr;
    this->lastPayload = payload;
    this->lastSequence = sequence;
    this->hasReceivedPacket = true;
}

void DisplayManager::setGatewayStatus(bool wifiConnected, const String& wifiIP,
                                      bool mqttConnected) {
    if (!initialized) return;
    this->isGateway = true;
    this->wifiConnected = wifiConnected;
    this->wifiIP = wifiIP;
    this->mqttConnected = mqttConnected;
}

void DisplayManager::setLoRaConfig(uint8_t sf, float bw, float freq) {
    if (!initialized) return;
    this->loraSF = sf;
    this->loraBW = bw;
    this->loraFreq = freq;
}

void DisplayManager::setBatteryLevel(uint8_t percentage, float voltage, bool isValid) {
    if (!initialized) return;
    this->batteryPercentage = percentage;
    this->batteryVoltage = voltage;
    this->batteryValid = isValid;
}

// Estado de alimentação USB, usado no gerenciamento do display
void DisplayManager::setUSBPowered(bool usb) {
    if (!initialized) return;
    this->usbPowered = usb;
}

// Lista de vizinhos para SCREEN_NEIGHBORS
void DisplayManager::setNeighbors(NeighborDisplayInfo* neighbors, uint8_t count) {
    if (!initialized) return;
    displayNeighborCount = count < MAX_DISPLAY_NEIGHBORS ? count : MAX_DISPLAY_NEIGHBORS;
    for (uint8_t i = 0; i < displayNeighborCount; i++) {
        displayNeighbors[i] = neighbors[i];
    }
}

// Contador de pacotes TX
void DisplayManager::setTxCount(uint32_t count) {
    if (!initialized) return;
    this->txCount = count;
}

void DisplayManager::flashActivity() {
    if (!initialized) return;
    activityIndicator = true;
    activityTime = millis();
    
    // O display NÃO acorda ao receber pacote
    // nem zera o timeout de sono.
    // Só o botão acorda o display ou zera o timeout (economiza ~15mA na bateria)
}

bool DisplayManager::checkButtonPress(uint8_t buttonPin, uint16_t pressedDurationMs) {
    if (!initialized) return false;
    bool currentState = (digitalRead(buttonPin) == LOW);  // Ativo em LOW
    
    if (currentState && !buttonPressed) {
        // Botão acabou de ser pressionado
        buttonPressed = true;
        buttonPressStart = millis();
        
        // Se o display está dormindo, qualquer pressão o acorda
        if (displayAsleep) {
            wakeUp();
            return false;  // A pressão que acorda não troca de tela
        }
        
    } else if (!currentState && buttonPressed) {
        // Botão solto
        buttonPressed = false;
        
        // Toque curto (< 500 ms) com o display acordado: só zera o timeout
        unsigned long pressDuration = millis() - buttonPressStart;
        
        if (pressDuration < 500 && !displayAsleep) {
            lastUserInteraction = millis();  // Zera o timeout de sono
            Serial.println("[DISPLAY] Quick tap - timeout reset");
            return false;
        }
        
        // Verifica se foi mantido o suficiente para trocar de tela
        if (pressDuration >= pressedDurationMs) {
            Serial.println("[DISPLAY] Button held, switching screen");
            lastUserInteraction = millis();  // Zera o timeout de sono
            return true;
        }
    }
    
    return false;
}

void DisplayManager::nextScreen(bool manual) {
    if (!initialized) return;
    if (currentScreen == SCREEN_SPLASH) {
        currentScreen = SCREEN_STATUS;
    } else if (currentScreen == SCREEN_STATUS) {
        currentScreen = SCREEN_NEIGHBORS;
    } else if (currentScreen == SCREEN_NEIGHBORS) {
        currentScreen = SCREEN_ABOUT;
    } else if (currentScreen == SCREEN_ABOUT) {
        currentScreen = SCREEN_STATUS;
    }

    lastScreenChange = millis();
    if (manual) {
        lastUserInteraction = millis();
    }
    Serial.printf("[DISPLAY] Switched to screen %d\n", currentScreen);
}

void DisplayManager::wakeUp() {
    if (!initialized) return;
    if (displayAsleep) {
        displayAsleep = false;
        display.setPowerSave(0);  // Acorda o display
        lastUserInteraction = millis();
        lastUpdateTime = 0;  // Força atualização imediata
        Serial.println("[DISPLAY] Display waking up!");
    }
}

void DisplayManager::showSleepCountdown(uint8_t secondsRemaining) {
    if (!initialized) return;
    display.clearBuffer();
    
    // Título
    display.setFont(u8g2_font_helvB10_tr);
    display.drawStr(10, 15, "DEEP SLEEP");
    
    // Contagem regressiva
    display.setFont(u8g2_font_logisoso32_tn);
    String countdown = String(secondsRemaining);
    uint8_t width = display.getStrWidth(countdown.c_str());
    display.drawStr((128 - width) / 2, 50, countdown.c_str());
    
    // Rótulo
    display.setFont(u8g2_font_6x10_tr);
    display.drawStr(40, 63, "segundos");
    
    display.sendBuffer();
}

void DisplayManager::powerOff() {
    if (!initialized) return;
    display.clearBuffer();
    display.sendBuffer();
    display.setPowerSave(1);  // Desliga o display
    Serial.println("[DISPLAY] Display powered off");
}

/******************************************************************************
 * MÉTODOS PRIVADOS DE DESENHO
 ******************************************************************************/

void DisplayManager::renderSplash() {
    renderAbout();
}

void DisplayManager::renderStatusScreen() {
    uint8_t y = 10;
    
    // Cabeçalho: nome do nó + endereço
    display.setFont(u8g2_font_helvB08_tr);
    String header = nodeName + " [0x" + String(loraAddress, HEX) + "]";
    display.drawStr(0, y, header.c_str());
    y += 2;
    
    // Linha separadora
    display.drawLine(0, y, 127, y);
    y += 10;
    
    // Fonte do corpo
    display.setFont(u8g2_font_6x10_tr);
    
    // Rotas + bateria na mesma linha
    String routes = "Rota:" + String(routeCount);
    display.drawStr(0, y, routes.c_str());
    
    // Bateria na mesma linha, à direita
    String bat = "Bat:";
    if (batteryValid) {
        bat += String(batteryPercentage) + "%";
    } else {
        bat += "--";
    }
    display.drawStr(70, y, bat.c_str());
    y += 12;
    
    // Uptime
    String uptime = "Uptime: " + formatUptime();
    display.drawStr(0, y, uptime.c_str());
    y += 12;
    
    // Configuração LoRa
    String loraConfig = "SF" + String(loraSF) + " " + 
                        String((int)loraBW) + "k " + 
                        String((int)loraFreq) + "M";
    display.drawStr(0, y, loraConfig.c_str());
    y += 12;
    
    // Contagem de TX + vizinhos (nós sensores)
    if (!isGateway) {
        String txInfo = "TX:" + String(txCount) + "  Viz:" + String(displayNeighborCount);
        display.drawStr(0, y, txInfo.c_str());
    }
    
    // Estado do gateway (se habilitado)
    if (isGateway) {
        // Estado do Wi-Fi e do MQTT
        String status = "WiFi:";
        status += wifiConnected ? "OK " : "OFF";
        status += " MQTT:";
        status += mqttConnected ? "OK" : "OFF";
        display.drawStr(0, y, status.c_str());
    }
    
    // Indicador de atividade (canto superior direito)
    if (activityIndicator) {
        display.drawDisc(120, 4, 3);  // Círculo cheio
    } else {
        display.drawCircle(120, 4, 3);  // Círculo vazio
    }
}

void DisplayManager::renderNeighborsScreen() {
    uint8_t y = 10;

    // Cabeçalho
    display.setFont(u8g2_font_helvB08_tr);
    String header = "REDE " + nodeName;
    display.drawStr(0, y, header.c_str());
    y += 2;

    // Linha separadora
    display.drawLine(0, y, 127, y);
    y += 14;

    // Fonte 6x13 no corpo
    display.setFont(u8g2_font_6x13_tr);

    // Número de vizinhos
    String vizStr = "Vizinhos: " + String(displayNeighborCount);
    display.drawStr(0, y, vizStr.c_str());
    y += 16;

    // Parent + RSSI + SNR + índice de qualidade (não é PDR).
    // "N%d" mostra o byte baixo do endereço LoRa, não o NODE_ID do parent.
    bool foundParent = false;
    for (uint8_t i = 0; i < displayNeighborCount && i < MAX_DISPLAY_NEIGHBORS; i++) {
        if (displayNeighbors[i].isParent) {
            char parentLine[32];
            snprintf(parentLine, sizeof(parentLine), "Parent: N%d  %ddBm",
                     displayNeighbors[i].address & 0xFF,
                     displayNeighbors[i].rssi);
            display.drawStr(0, y, parentLine);
            y += 16;

            char metricsLine[28];
            snprintf(metricsLine, sizeof(metricsLine), "SNR:%ddB  Q:%d/10",
                     (int)displayNeighbors[i].snr, displayNeighbors[i].linkQuality);
            display.drawStr(0, y, metricsLine);

            foundParent = true;
            break;
        }
    }

    if (!foundParent) {
        display.drawStr(0, y, "Parent: --");
    }

    // Indicador de atividade
    if (activityIndicator) {
        display.drawDisc(120, 4, 3);
    } else {
        display.drawCircle(120, 4, 3);
    }
}

void DisplayManager::renderAbout() {
    // "Rede Mesh LoRa" — fonte média, centrado
    display.setFont(u8g2_font_8x13B_tr);
    const char* title = "Rede Mesh LoRa";
    uint8_t titleW = display.getStrWidth(title);
    display.drawStr((128 - titleW) / 2, 30, title);

    // Versão do firmware — fonte pequena, centrado
    display.setFont(u8g2_font_6x10_tr);
    String ver = String("v") + fwVersion;
    uint8_t verW = display.getStrWidth(ver.c_str());
    display.drawStr((128 - verW) / 2, 48, ver.c_str());
}

/******************************************************************************
 * MÉTODOS AUXILIARES
 ******************************************************************************/

void DisplayManager::drawSignalBar(uint8_t x, uint8_t y, uint8_t width, 
                                   uint8_t height, float rssi, float snr) {
    // Contorno
    display.drawFrame(x, y, width, height);
    
    // Nível de preenchimento (0-100%)
    uint8_t fillPercent = 0;
    
    // RSSI como indicador principal
    if (rssi >= RSSI_EXCELLENT) {
        fillPercent = 100;
    } else if (rssi >= RSSI_GOOD) {
        fillPercent = 75;
    } else if (rssi >= RSSI_FAIR) {
        fillPercent = 50;
    } else if (rssi >= RSSI_POOR) {
        fillPercent = 25;
    } else {
        fillPercent = 10;
    }
    
    // Ajuste pelo SNR (±10%)
    if (snr >= SNR_EXCELLENT) {
        fillPercent = min(100, fillPercent + 10);
    } else if (snr < SNR_POOR) {
        fillPercent = max(0, fillPercent - 10);
    }
    
    // Parte preenchida (de baixo para cima)
    uint8_t fillHeight = (height - 2) * fillPercent / 100;
    if (fillHeight > 0) {
        display.drawBox(x + 1, y + height - 1 - fillHeight, width - 2, fillHeight);
    }
}

String DisplayManager::getSignalQuality(float rssi, float snr) {
    // Avaliação combinada da qualidade
    if (rssi >= RSSI_EXCELLENT && snr >= SNR_EXCELLENT) {
        return "Excelente";
    } else if (rssi >= RSSI_GOOD && snr >= SNR_GOOD) {
        return "Boa";
    } else if (rssi >= RSSI_FAIR && snr >= SNR_FAIR) {
        return "Regular";
    } else if (rssi >= RSSI_POOR && snr >= SNR_POOR) {
        return "Fraca";
    } else {
        return "Critica";
    }
}

String DisplayManager::getNodeName(uint16_t nodeId) {
    // Endereços de 1 a 99 viram "NO <n>"
    if (nodeId >= 1 && nodeId <= 99) {
        return "NO " + String(nodeId);
    }
    // Os demais aparecem em hexadecimal
    return "0x" + String(nodeId, HEX);
}

String DisplayManager::formatUptime() {
    unsigned long totalSeconds = millis() / 1000;
    unsigned long hours = totalSeconds / 3600;
    unsigned long minutes = (totalSeconds % 3600) / 60;
    unsigned long seconds = totalSeconds % 60;
    
    String uptime = "";
    
    if (hours > 0) {
        uptime += String(hours) + "h ";
    }
    if (minutes > 0 || hours > 0) {
        uptime += String(minutes) + "m ";
    }
    uptime += String(seconds) + "s";
    
    return uptime;
}

String DisplayManager::truncateString(const String& str, uint8_t maxLen) {
    if (str.length() <= maxLen) {
        return str;
    }
    return str.substring(0, maxLen - 3) + "...";
}
