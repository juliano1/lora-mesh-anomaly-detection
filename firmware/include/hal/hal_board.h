/*******************************************************************************
 * HAL_BOARD.H — Identidade de hardware em runtime
 *
 * Fornece strings e funções para identificação do hardware (Heltec V2).
 * Depende de hal_pins.h (que define HW_VERSION, HW_VERSION_FULL, RADIO_CHIP).
 ******************************************************************************/

#pragma once

#include "hal_pins.h"
#include <Arduino.h>

/**
 * Imprime tabela de identidade de hardware na Serial.
 * Chamada uma vez no início de HAL_Init().
 * Inclui: variante, rádio, frequência e todos os pinos principais.
 */
inline void hal_board_info() {
    Serial.println("╔════════════════════════════════════════╗");
    Serial.printf( "║ Hardware: %-28s ║\n", HW_VERSION_FULL);
    Serial.printf( "║ Radio:    %-28s ║\n", RADIO_CHIP);
    Serial.printf( "║ Freq:     %.1f MHz %-18s║\n", FREQ_BAND, "");
    Serial.printf( "║ LoRa CS/RST/IRQ/IO1: %2d/%2d/%2d/%2d %-4s║\n",
                   LORA_CS, LORA_RST, LORA_DIO0, LORA_DIO1, "");
    Serial.printf( "║ OLED SDA/SCL/RST:    %2d/%2d/%2d %-7s║\n",
                   OLED_SDA, OLED_SCL, OLED_RST, "");
    Serial.printf( "║ LED: %2d  VEXT: %2d  BAT_ADC: %2d %-5s║\n",
                   LED_BUILTIN, VEXT_CTRL, BATTERY_PIN, "");
    Serial.println("╚════════════════════════════════════════╝");
}
