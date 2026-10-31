/*******************************************************************************
 * HAL_BATTERY.H — Interface de leitura de bateria (hardware-agnostic)
 *
 * Implementação:
 *   src/hal/hal_battery_v2.cpp  — ESP32 + SX1276, ADC1_CH1 (GPIO37)
 *
 * Uso:
 *   1. Chamar hal_battery_init() uma vez (no início do setup ou quickboot).
 *   2. Chamar hal_battery_read_mv() para obter a tensão em milivolts.
 *   3. Chamar hal_battery_to_pct() para converter para percentual.
 ******************************************************************************/

#pragma once

#include <Arduino.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Inicializa o ADC e estruturas de calibração.
 * Seguro chamar várias vezes (idempotente).
 * No quickboot path, chamado antes de BatteryManager::begin().
 */
void hal_battery_init();

/**
 * Lê a tensão da bateria em milivolts, já multiplicada pelo fator de correção do divisor.
 * Realiza múltiplas amostras internas para reduzir ruído.
 *
 * @return  Tensão em mV (ex.: 3700 = 3,70 V).
 *          Se o ADC ainda não foi inicializado, inicializa antes de ler.
 */
uint32_t hal_battery_read_mv();

#ifdef __cplusplus
} // extern "C"
#endif

/**
 * Converte tensão em volts para percentual de carga (curva LiPo).
 * Função comum a todas as variantes de hardware.
 * Inline para não gerar símbolo duplicado entre unidades de compilação.
 *
 * @param  voltageV  Tensão em volts (ex.: 3.7f)
 * @return Percentual 0-100
 */
inline uint8_t hal_battery_to_pct(float voltageV) {
    if (voltageV >= 4.2f) return 100;
    if (voltageV >= 4.0f) return 90 + (uint8_t)((voltageV - 4.0f) / 0.2f * 10.0f);
    if (voltageV >= 3.7f) return 50 + (uint8_t)((voltageV - 3.7f) / 0.3f * 40.0f);
    if (voltageV >= 3.4f) return 20 + (uint8_t)((voltageV - 3.4f) / 0.3f * 30.0f);
    if (voltageV >= 3.0f) return      (uint8_t)((voltageV - 3.0f) / 0.4f * 20.0f);
    return 0;
}
