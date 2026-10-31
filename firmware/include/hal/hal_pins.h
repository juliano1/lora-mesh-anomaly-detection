/*******************************************************************************
 * HAL_PINS.H — Dispatcher de pinos de hardware
 *
 * Inclui o arquivo de pinos com base na macro definida em platformio.ini
 * (build_flags).
 *
 * Board suportado:
 *   HELTEC_V2        — ESP32   + SX1276
 ******************************************************************************/

#pragma once

/* ── Nenhuma macro definida ───────────────────────────────────────── */
#if !defined(HELTEC_V2)
    #error "Hardware não definido. Adicione -D HELTEC_V2 em platformio.ini."
#endif

/* ── Inclusão do arquivo de pinos ─────────────────────────────────── */
#include "hal_pins_v2.h"
