/*******************************************************************************
 * HAL_RADIO_V2.CPP — Configuração de rádio para Heltec WiFi LoRa 32 V2
 *
 * Módulo: SX1276 (interrupção por DIO0)
 * SPI:    padrão do board heltec_wifi_lora_32_V2 (SCK=5, MISO=19, MOSI=27)
 ******************************************************************************/

#ifdef HELTEC_V2

#include "hal/hal_radio.h"
#include "hal/hal_pins.h"

void hal_radio_fill_config(LoraMesher::LoraMesherConfig& cfg) {
    cfg.loraCs  = LORA_CS;    /* GPIO18 — NSS/CS                      */
    cfg.loraRst = LORA_RST;   /* GPIO14 — Reset                        */
    cfg.loraIrq = LORA_DIO0;  /* GPIO26 — DIO0 (IRQ)                   */
    cfg.loraIo1 = LORA_DIO1;  /* GPIO35 — DIO1                         */
    cfg.module  = LoraMesher::LoraModules::SX1276_MOD;
}

#endif /* HELTEC_V2 */
