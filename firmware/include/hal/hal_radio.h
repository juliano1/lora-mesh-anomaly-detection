/*******************************************************************************
 * HAL_RADIO.H — Interface de configuração do rádio LoRa
 *
 * Preenche uma LoraMesherConfig com os pinos e módulo corretos
 * para cada variante de hardware.
 *
 * Implementação:
 *   src/hal/hal_radio_v2.cpp  — SX1276_MOD, pinos V2
 *
 * Uso em setupLoRaMesh():
 *   LoraMesher::LoraMesherConfig config;
 *   hal_radio_fill_config(config);
 *   radio.begin(config);
 ******************************************************************************/

#pragma once

#include <LoraMesher.h>

/**
 * Preenche cfg com pinos e tipo de módulo do hardware compilado.
 * Deve ser chamada antes de radio.begin().
 */
void hal_radio_fill_config(LoraMesher::LoraMesherConfig& cfg);
