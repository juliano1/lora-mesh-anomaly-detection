/*******************************************************************************
 * HAL_PINS_V2.H — Pinos da Heltec WiFi LoRa 32 V2
 *
 * MCU:   ESP32 (Xtensa LX6, 240 MHz)
 * Rádio: SX1276 (interrupção por DIO0, sem pino BUSY)
 * Flash: 4 MB
 *
 * Estes defines são a fonte única dos números de GPIO da V2; a exceção é o
 * botão PRG (POWER_BUTTON_PIN = 0), definido em main.cpp.
 ******************************************************************************/

#pragma once

/* ── LoRa SPI / Rádio (SX1276) ─────────────────────────────────── */
#define LORA_SCK        5       /* Clock SPI                         */
#define LORA_MISO       19      /* SPI MISO                          */
#define LORA_MOSI       27      /* SPI MOSI                          */
#define LORA_CS         18      /* Chip Select (NSS)                 */
#define LORA_RST        14      /* Reset do rádio                    */
#define LORA_DIO0       26      /* IRQ / DIO0                        */
#define LORA_DIO1       35      /* DIO1                              */
#define LORA_DIO2       34      /* DIO2 (não usado pela LoRaMesher)  */

/* ── Display OLED (SSD1306 via I2C) ────────────────────────────── */
#define OLED_SDA        4
#define OLED_SCL        15
#define OLED_RST        16

/* ── LED da placa ──────────────────────────────────────────────── */
/* Guard: o board heltec_wifi_lora_32_V2 já define LED_BUILTIN=25   */
#ifndef LED_BUILTIN
    #define LED_BUILTIN 25      /* LED branco                        */
#endif

/* ── Alimentação externa (Vext) ────────────────────────────────── */
#define VEXT_CTRL       21      /* LOW = Vext ligado (alimenta o OLED; também
                                   ligado na leitura da bateria, ver
                                   hal_battery_v2.cpp)               */

/* ── ADC da bateria ────────────────────────────────────────────── */
#define BATTERY_PIN     37      /* GPIO37 = ADC1_CH1                 */

/* ── Identificação da placa ────────────────────────────────────── */
#define HW_VERSION      "v2"
#define HW_VERSION_FULL "Heltec LoRa 32 V2"
#define RADIO_CHIP      "SX1276"
/* FREQ_BAND definida centralmente em hal_conf.h          */
