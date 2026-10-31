/******************************************************************************
 * BatteryManager.cpp - Implementação do monitor de bateria
 *
 * Leitura de ADC delegada a hal_battery_vX.cpp (selecionado em compile-time).
 * Esta classe é hardware-agnostic: não contém #ifdef de variante.
 ******************************************************************************/

#include "BatteryManager.h"

BatteryManager::BatteryManager(uint32_t updateIntervalMs)
    : updateInterval(updateIntervalMs),
      lastUpdate(0),
      voltage(0.0f),
      percentage(0),
      valid(false),
      usbPowered(false),
      lastVoltageReading(0.0f)
{
}

void BatteryManager::begin() {
    hal_battery_init();  /* loga pino e fator de correção por variante */

    Serial.println("[BATTERY] Monitor inicializado");

    forceUpdate();
}

void BatteryManager::update() {
    if (millis() - lastUpdate >= updateInterval) {
        forceUpdate();
    }
}

void BatteryManager::forceUpdate() {
    float newVoltage = readVoltage();

    voltage = smoothVoltage(newVoltage);
    percentage = hal_battery_to_pct(voltage);

    /* Validação de faixa: USB pode chegar a ~5.2 V pelo divisor */
    if (voltage >= 2.0f && voltage <= 5.5f) {
        valid = true;
        usbPowered = (voltage > 4.3f);
        if (usbPowered) {
            percentage = 100;
            Serial.printf("[BATTERY] USB: %.2fV\n", voltage);
        } else {
            Serial.printf("[BATTERY] Bateria: %.2fV, %d%%\n", voltage, percentage);
        }
    } else {
        valid = false;
        usbPowered = false;
        Serial.printf("[BATTERY] Leitura invalida: %.2fV (fora de 2.0-5.5V)\n", voltage);
    }

    lastUpdate = millis();

    if (isLow() && valid && !usbPowered) {
        Serial.println("[BATTERY] Bateria baixa!");
    }
}

float BatteryManager::readVoltage() {
    uint32_t mv = hal_battery_read_mv();
    return mv / 1000.0f;
}

float BatteryManager::smoothVoltage(float newVoltage) {
    if (lastVoltageReading == 0.0f) {
        lastVoltageReading = newVoltage;
        return newVoltage;
    }
    float smoothed = (SMOOTHING_FACTOR * newVoltage) +
                     ((1.0f - SMOOTHING_FACTOR) * lastVoltageReading);
    lastVoltageReading = smoothed;
    return smoothed;
}
