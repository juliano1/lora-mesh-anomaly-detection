/*******************************************************************************
 * HAL_BATTERY_V2.CPP — Leitura de bateria para Heltec WiFi LoRa 32 V2
 *
 * Hardware:
 *   - MCU:     ESP32 (Xtensa LX6)
 *   - ADC pin: GPIO37 = ADC1_CHANNEL_1
 *   - Divisor: rede resistiva (fator de correção empírico 3.1x)
 *   - Vext:    GPIO21 LOW = liga divisor de bateria; HIGH = desliga
 *              (o mesmo Vext alimenta o OLED, segundo hal_pins_v2.h)
 *
 * O fator de correção 3.1 foi determinado empiricamente:
 *   ADC calibrado lê ~1.27 V quando bateria está em ~3.94 V (3.94/1.27 ≈ 3.1).
 *   É um fator de calibração único (multiplicativo): não corrige offset nem
 *   não linearidade da leitura.
 *   Se a leitura parecer imprecisa, medir VBAT com multímetro em 3 pontos
 *   (3.3 V, 3.7 V, 4.1 V) e ajustar HAL_BATTERY_V2_CORRECTION_FACTOR.
 *
 * ⚠️  Esta implementação usa esp_adc_cal.h (ESP-IDF 4.x / Arduino core 2.x).
 *     Compatível com platform = espressif32@~6.5.0.
 ******************************************************************************/

#ifdef HELTEC_V2

#include "hal/hal_battery.h"
#include "hal/hal_pins.h"
#include <Arduino.h>
#include <driver/adc.h>
#include <esp_adc_cal.h>

/* Fator de correção empírico (ajustar após calibração com multímetro) */
static constexpr float HAL_BATTERY_V2_CORRECTION_FACTOR = 3.1f;

/* Estrutura de calibração ADC — alocada estaticamente para evitar heap */
static esp_adc_cal_characteristics_t s_adcChar;
static bool s_adcInitialized = false;

void hal_battery_init() {
    if (s_adcInitialized) return;  /* idempotente */

    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC1_CHANNEL_1, ADC_ATTEN_DB_11);

    esp_adc_cal_characterize(
        ADC_UNIT_1,
        ADC_ATTEN_DB_11,
        ADC_WIDTH_BIT_12,
        1100,          /* tensão de referência interna em mV */
        &s_adcChar
    );

    s_adcInitialized = true;
    Serial.printf("[HAL_BAT V2] ADC inicializado — pin GPIO%d, fator %.1fx\n",
                  BATTERY_PIN, HAL_BATTERY_V2_CORRECTION_FACTOR);
}

uint32_t hal_battery_read_mv() {
    if (!s_adcInitialized) hal_battery_init();

    /* Liga divisor de tensão via Vext (necessário no V2 para leitura precisa) */
    pinMode(VEXT_CTRL, OUTPUT);
    digitalWrite(VEXT_CTRL, LOW);
    delay(10);  /* aguarda estabilização */

    /* Média de 10 leituras para reduzir ruído */
    const int kNumReadings = 10;
    uint32_t adcSum = 0;
    for (int i = 0; i < kNumReadings; i++) {
        adcSum += adc1_get_raw(ADC1_CHANNEL_1);
        delay(10);
    }

    /* Desliga Vext para economizar energia; como o Vext também alimenta o OLED
       (hal_pins_v2.h), o efeito disso sobre o display não foi verificado em placa. */
    digitalWrite(VEXT_CTRL, HIGH);

    uint32_t adcRaw = adcSum / kNumReadings;
    uint32_t adcMv  = esp_adc_cal_raw_to_voltage(adcRaw, &s_adcChar);

    Serial.printf("[HAL_BAT V2] ADC raw:%u  ADC mV:%u  Bat mV:%.0f\n",
                  adcRaw, adcMv,
                  adcMv * HAL_BATTERY_V2_CORRECTION_FACTOR);

    return (uint32_t)(adcMv * HAL_BATTERY_V2_CORRECTION_FACTOR);
}

#endif /* HELTEC_V2 */
