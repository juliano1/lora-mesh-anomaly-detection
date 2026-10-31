/*******************************************************************************
 * HAL_CONF.H — Configuração de rádio e inicialização da placa
 *
 * Define os parâmetros LoRa, as macros de LED/Vext e HAL_Init(); os pinos
 * vêm do HAL modular em include/hal/.
 *
 * Fonte única de verdade de pinos: include/hal/hal_pins_v2.h
 * Validação de hardware:           include/hal/hal_pins.h
 *
 * NÃO adicionar novos pinos ou #ifdef de hardware aqui.
 * Novo hardware vai em hal_pins_<placa>.h + hal_battery_<placa>.cpp + hal_radio_<placa>.cpp
 ******************************************************************************/

#ifndef HAL_CONF_H
#define HAL_CONF_H

#include <Arduino.h>
#include "hal/hal_pins.h"   /* Valida seleção e inclui pinos corretos */

/* Frequência de laboratório calculada a partir de um índice de canal AU915.
 * O protocolo é LoRaMesher. Escolher um canal não garante ausência de
 * interferência; valide frequência, potência e antena antes do uso em campo.
 * Todos os nós devem usar os mesmos parâmetros. Padrão: 903.9 MHz.
 */

/* Parâmetros do plano AU915 — não alterar (definições do padrão)          */
#define AU915_FREQ_BASE_MHZ     902.3f
#define AU915_FREQ_STEP_MHZ     0.2f
#define AU915_SUBBAND_SIZE      8

/* Sub-banda e canal padrão — sobrescrever via -D no platformio.ini        */
#ifndef LORA_AU915_SUBBAND
  #define LORA_AU915_SUBBAND        2   /* 1–8                            */
#endif
#ifndef LORA_AU915_CHANNEL_IN_SB
  #define LORA_AU915_CHANNEL_IN_SB  0   /* 0–7                            */
#endif

/* Validação em tempo de compilação                                         */
#if (LORA_AU915_SUBBAND < 1) || (LORA_AU915_SUBBAND > 8)
  #error "LORA_AU915_SUBBAND deve estar entre 1 e 8"
#endif
#if (LORA_AU915_CHANNEL_IN_SB < 0) || (LORA_AU915_CHANNEL_IN_SB > 7)
  #error "LORA_AU915_CHANNEL_IN_SB deve estar entre 0 e 7"
#endif

/* Canal absoluto e frequência resultante                                   */
#define LORA_AU915_ABS_CHANNEL \
  (((LORA_AU915_SUBBAND) - 1) * AU915_SUBBAND_SIZE + (LORA_AU915_CHANNEL_IN_SB))

/* Padrão calculado pelo plano AU915; pode ser sobrescrito por -D FREQ_BAND=
 * no platformio.ini.                                                     */
#ifndef FREQ_BAND
  #define FREQ_BAND \
    (AU915_FREQ_BASE_MHZ + AU915_FREQ_STEP_MHZ * (float)(LORA_AU915_ABS_CHANNEL))
#endif
/* Padrão (SB2 offset 0): 902.3 + 0.2*8 = 903.9 MHz                       */

#include "hal/hal_board.h"  /* hal_board_info() — usa FREQ_BAND acima     */

/*******************************************************************************
 * CONFIGURAÇÃO DE RÁDIO COMUM (independe da variante)
 ******************************************************************************/
#define LORA_BANDWIDTH      125.0   /* kHz (125, 250, 500)            */
#define LORA_SPREADING      7       /* Spreading factor (7-12)        */
#define LORA_CODING_RATE    5       /* Coding Rate (5-8 = 4/5..4/8)  */
#define LORA_SYNC_WORD      0x12    /* Sync word de rede privada      */
#define LORA_TX_POWER       20      /* dBm; validar para a instalação e antena */
#define LORA_PREAMBLE       8       /* Comprimento do preâmbulo       */

/*******************************************************************************
 * MACROS AUXILIARES
 ******************************************************************************/
#ifdef LED_ACTIVE_LOW
#define LED_ON()     digitalWrite(LED_BUILTIN, LOW)
#define LED_OFF()    digitalWrite(LED_BUILTIN, HIGH)
#else
#define LED_ON()     digitalWrite(LED_BUILTIN, HIGH)
#define LED_OFF()    digitalWrite(LED_BUILTIN, LOW)
#endif
#define LED_TOGGLE() digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN))

#define VEXT_ON()  do { if ((VEXT_CTRL) >= 0) digitalWrite(VEXT_CTRL, LOW);  } while(0)
#define VEXT_OFF() do { if ((VEXT_CTRL) >= 0) digitalWrite(VEXT_CTRL, HIGH); } while(0)

/*******************************************************************************
 * INICIALIZAÇÃO DO HARDWARE
 * Configura GPIO de LED e Vext, aguarda estabilização e imprime banner.
 ******************************************************************************/
inline void HAL_Init() {
    pinMode(LED_BUILTIN, OUTPUT);
    LED_OFF();

    if (VEXT_CTRL >= 0) {
        pinMode(VEXT_CTRL, OUTPUT);
        VEXT_ON();
    }

    delay(100);  /* Aguarda o hardware estabilizar */

    hal_board_info();
}

#endif /* HAL_CONF_H */
