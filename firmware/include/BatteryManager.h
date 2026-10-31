/******************************************************************************
 * BatteryManager.h - Monitor do nível de bateria do LoRaMesher Laboratory
 *
 * Gerencia leitura de tensão e percentual de carga da bateria.
 * A leitura de ADC é delegada ao HAL:
 *   - V2: hal_battery_v2.cpp (GPIO37 / ADC1_CH1)
 *
 * Recursos:
 *   - Leitura de tensão via HAL (ADC calibrado + média de amostras)
 *   - Cálculo de percentual (curva LiPo 3.7 V nominal)
 *   - Filtro de suavização exponencial (EMA)
 *   - Detecção de alimentação USB
 *   - Detecção de bateria baixa
 ******************************************************************************/

#ifndef BATTERYMANAGER_H
#define BATTERYMANAGER_H

#include <Arduino.h>
#include "hal/hal_battery.h"  /* hal_battery_init(), hal_battery_read_mv() */

/* Limiares de tensão da bateria (LiPo 3.7V nominal) */
#define BATTERY_VOLTAGE_MAX     4.2f    /* Carga completa (100%)     */
#define BATTERY_VOLTAGE_MIN     3.0f    /* Vazia (0%)                */
#define BATTERY_VOLTAGE_LOW     3.3f    /* Aviso de bateria baixa    */

class BatteryManager {
public:
    /**
     * @param updateIntervalMs  Intervalo entre leituras (padrão 60 s)
     */
    explicit BatteryManager(uint32_t updateIntervalMs = 60000);

    /** Inicializa ADC via HAL e faz leitura inicial. */
    void begin();

    /** Atualiza leitura se o intervalo tiver expirado. Chamar do loop(). */
    void update();

    /** Força leitura imediata ignorando o intervalo. */
    void forceUpdate();

    float   getVoltage()    const { return voltage; }
    uint8_t getPercentage() const { return percentage; }
    bool    isLow()         const { return voltage > 0.0f && voltage < BATTERY_VOLTAGE_LOW; }
    bool    isValid()       const { return valid; }
    bool    isUSB()         const { return usbPowered; }

private:
    uint32_t      updateInterval;
    unsigned long lastUpdate;

    float   voltage;
    uint8_t percentage;
    bool    valid;
    bool    usbPowered;
    float   lastVoltageReading;

    float   readVoltage();
    float   smoothVoltage(float newVoltage);

    static constexpr float SMOOTHING_FACTOR = 0.2f;
};

#endif /* BATTERYMANAGER_H */
