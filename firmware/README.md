# Firmware

Firmware PlatformIO para a Heltec WiFi LoRa 32 V2 (ESP32 + SX1276). Os nós formam a
malha com a [LoRaMesher](https://github.com/LoRaMesher/LoRaMesher); o gateway é um nó
que, além disso, publica a telemetria via Wi-Fi/MQTT e repassa comandos para a malha.

## Ambientes

| Ambiente | Papel | Requer |
|---|---|---|
| `heltec_v2` | nó da malha | `NODE_ID` único por placa |
| `heltec_v2_gateway` | gateway (`NODE_ID=1`, `IS_GATEWAY`) | `include/credentials.h` |

`NETWORK_ID` (padrão 1) vale para todos os ambientes e define os tópicos MQTT. Para
uma segunda rede, use outro `NETWORK_ID`, outro usuário no broker e, de preferência,
outro canal de rádio.

## Compilar e gravar

```bash
cp include/credentials.h.example include/credentials.h   # só o gateway usa; edite os valores
pio run -e heltec_v2 -e heltec_v2_gateway                  # compila os dois
pio run -e heltec_v2 -t upload                             # grava um nó
pio run -e heltec_v2_gateway -t upload                     # grava o gateway
pio device monitor                                         # serial a 115200 baud
```

`credentials.h` é ignorado pelo Git. Preencha Wi-Fi, endereço do broker na rede local e
o usuário MQTT do gateway, criado conforme `scripts/mosquitto/config/acl.example`. O MQTT
não usa TLS: use só em rede de laboratório isolada.

As versões das bibliotecas estão fixadas em `platformio.ini`, inclusive a LoRaMesher
(commit `d37dee7`). Antes de cada build, `apply_patch.py` aplica
`patches/loramesher_rssi_fix.patch`, que faz a LoRaMesher registrar RSSI/SNR por
vizinho. O script confere o resultado, não reaplica um patch já aplicado e interrompe
o build se a fonte da biblioteca não corresponder ao esperado.

## Rádio

Os parâmetros ficam em `include/hal_conf.h`:

| Parâmetro | Padrão |
|---|---|
| Frequência | 903,9 MHz (plano AU915, sub-banda 2, canal 0) |
| Largura de banda | 125 kHz |
| Spreading factor | 7 |
| Coding rate | 4/5 |
| Sync word | `0x12` |
| Potência | 20 dBm |

O canal pode ser trocado sem editar código, com `-D LORA_AU915_SUBBAND=<1–8>` e
`-D LORA_AU915_CHANNEL_IN_SB=<0–7>`; todos os nós de uma rede precisam usar o mesmo.
**Antes de qualquer ensaio de transmissão**, confira esta configuração com os requisitos
da Anatel para equipamentos de radiação restrita: frequência, largura de banda, potência
conduzida e EIRP com a antena usada, e ciclo de trabalho. No Brasil, as faixas são
902–907,5 e 915–928 MHz: o padrão de 903,9 MHz fica na primeira, e os canais 26 a 63
(907,5 a 914,9 MHz) ficam fora de ambas. O padrão de 20 dBm também precisa ser conferido.
Fora do Brasil, vale a regulamentação local.

A pinagem da placa está em `include/hal/hal_pins_v2.h`.

## Beacon

Cada nó transmite um beacon a cada 20 s, e o gateway o publica em `lab/<rede>/uplink`
com os campos `gw_*` acrescentados:

| Campo | Conteúdo |
|---|---|
| `i`, `a`, `hw` | nome do nó, endereço LoRa (hex), versão de hardware |
| `n` | até 2 vizinhos: `a` endereço, `r` RSSI (dBm), `s` SNR (dB), `p` é parent, `h` saltos, `l` índice de qualidade 0–10 calculado de RSSI/SNR |
| `nearest` | vizinho de melhor RSSI |
| `u` | uptime (min) |
| `q` | número de sequência (lacunas indicam perda) |
| `b` | bateria (%); `-1` alimentação USB, `-2` sensor inválido. Na bateria, só é enviado quando muda, fica crítico ou a cada 5 min |
| `sim` | presente só durante uma simulação de anomalia |
| `gw_rssi`, `gw_snr` | RSSI/SNR do último Hello do remetente na tabela de rotas do gateway, não do pacote recebido; sem entrada, o firmware envia `-85`/`0` fixos, que não se distinguem de uma medida |
| `gw_toa`, `gw_id` | tempo no ar estimado (ms) e nome do gateway |

O endereço LoRa é atribuído pela LoRaMesher a partir do MAC da placa e aparece no
display e no serial. É esse endereço que se cadastra no backend.

## Comandos

O gateway assina `lora/mesh/<rede>/command` e aceita:

| `cmd` | Campos | Efeito |
|---|---|---|
| `send` | `dest` (hex, 1–4 dígitos), `msg`, `reliable` | Envia `msg` a um nó. |
| `broadcast` | `msg` | Envia `msg` a todos os nós. |
| `ping` | `dest` | Envia um ping confiável. |

Com `msg_id`, o resultado sai em `lab/<rede>/send_result`. `sent` significa apenas que o
pacote entrou na fila de transmissão; a LoRaMesher não informa a entrega. Destino
inválido, payload grande demais e comando ausente ou desconhecido geram `failed`, e o
comando não é transmitido.

Os nós interpretam as mensagens `LED_ON`, `LED_OFF`, `SIM_START:A<n>[:<segundos>]` e
`SIM_STOP`. `SIM_START` altera os valores do beacon para imitar o tipo A1–A7 por 60 a
1800 s (padrão 300 s), com o campo `sim` marcando os dados como simulados. As mensagens
que trafegam pela malha não são autenticadas.

## Estrutura

```
firmware/
├── platformio.ini
├── apply_patch.py              # aplica e verifica o patch RSSI na LoRaMesher
├── patches/loramesher_rssi_fix.patch
├── include/
│   ├── config.h                # tópicos MQTT derivados de NETWORK_ID
│   ├── hal_conf.h              # rádio
│   ├── hal/                    # pinos, rádio e bateria da Heltec V2
│   ├── DisplayManager.h, BatteryManager.h
│   └── credentials.h.example
└── src/
    ├── main.cpp                # malha, beacon, gateway MQTT, comandos e simulação
    ├── DisplayManager.cpp, BatteryManager.cpp
    └── hal/
```

O display mostra o índice de qualidade do parent como `Q:<0–10>/10`; não é taxa de entrega.

No gateway, o cliente MQTT é configurado no boot mesmo sem Wi-Fi. Wi-Fi e MQTT são
reconectados no `loop()` a cada 5 s, no máximo. Um mutex recursivo protege o cliente
MQTT, usado pelo `loop()` e pela tarefa de recepção. O display e o estado da simulação
são compartilhados entre as duas tarefas sem proteção.

O firmware desta versão foi compilado, mas não foi exercitado em placas.
