# LoRa Mesh Anomaly Detection

Monitoramento de uma rede mesh LoRa de laboratório com detecção de anomalias.
Nós ESP32 (Heltec WiFi LoRa 32 V2) formam a malha com a biblioteca
[LoRaMesher](https://github.com/LoRaMesher/LoRaMesher). Um nó gateway publica a
telemetria via MQTT. O backend grava os dados, calcula 11 features por janela de
tempo e aplica um detector de anomalias: o LSTM-AE por padrão, com o Isolation Forest
para comparação. As janelas anômalas são classificadas nos tipos A1–A7.

**Versão experimental.** O código foi testado apenas com dados sintéticos. Ainda não
houve validação com dados reais nem testes nas placas. O autoteste da detecção ainda
reprova em alguns cenários. As limitações conhecidas estão em
[Limitações](#limitações).

| Componente | Pasta | Tecnologia |
|---|---|---|
| Firmware (nó e gateway) | `firmware/` | C++, PlatformIO, Arduino-ESP32, LoRaMesher, RadioLib |
| Backend (API, MQTT, detecção) | `backend/` | Python 3.12+, FastAPI, SQLModel, scikit-learn, PyTorch (opcional) |
| Utilitários | `scripts/` | autoteste, exportação/importação de dados, broker local |
| Detector sobre CSV | `importacao_arquivos_csv/` | análise independente dos dados exportados |

Esta versão não inclui interface web, análise por LLM, Telegram, relatórios PDF nem
implantação de produção.

```
Nó ◄─LoRa─► Nó ◄─LoRa─► Gateway ──Wi-Fi/MQTT (lab/<rede>/uplink)──► Mosquitto ──► Backend ── SQLite/PostgreSQL
                                 ◄── comandos (lora/mesh/<rede>/command) ──────────┘
```

## Detecção

O backend agrega a telemetria de cada nó em janelas e calcula, nesta ordem:
`rssi_mean`, `rssi_std`, `snr_mean`, `snr_std`, `pdr_mean`, `battery_delta`,
`packet_loss_rate`, `rssi_snr_divergence`, `beacon_interval_std`,
`route_change_rate` e `link_asymmetry`. Definições e unidades estão em
`backend/ai/anomaly_definitions.py`. Note que `pdr_mean` é um índice heurístico de
qualidade (0–10) calculado pelo firmware a partir de RSSI/SNR, e não a PDR medida.

Há dois detectores:

- **LSTM autoencoder (padrão experimental).** Avalia a sequência das últimas 12
  subjanelas de 5 min e exige o PyTorch (`ML_LSTM_ENABLED=true`). Sem o PyTorch, o
  backend usa o IF e registra o motivo da troca (`fallback_reason`).
- **Isolation Forest.** Avalia o vetor da janela de 15 min e fica disponível para
  comparação.

O LSTM-AE é o candidato principal para a próxima validação. Foi o único detector a passar
em todos os cenários sintéticos do autoteste, em 4 de 5 sementes; na outra, teve falsos
positivos acima do limite. Isso mostra sensibilidade a desvios nesses cenários, não
desempenho em dados reais nem aprendizado de padrões temporais (veja
[Limitações](#limitações)).

Os modelos rodam isolados ou combinados (`PATCH /api/settings/ml`, campo `online`):

- `single`: só o primário (padrão);
- `shadow`: cada modelo decide e registra os próprios alertas;
- `cascade` (experimental): o alerta exige a concordância dos dois;
- `fusion` (experimental): compara a média ponderada dos postos dos dois scores com um
  limiar τ calibrado fora do treino.

Cascata e fusão ainda não mostraram vantagem sobre o LSTM-AE isolado. A API as marca
como experimentais (`experimental_combos`).

O treino do LSTM-AE divide a referência no tempo, sem embaralhar, com purga nos cortes:

- 60% iniciais para os pesos;
- 20% seguintes só para a parada antecipada;
- 20% finais para o limiar p99 e o τ da fusão.

O IF treina até o início do bloco de calibração. Assim, a escolha da época não reaproveita
os dados que fixam o limiar, e o teste fica posterior aos três blocos. É a separação que
a documentação do scikit-learn recomenda: o limiar de decisão não deve ser ajustado com os
dados que treinaram o modelo, e o teste não deve orientar escolhas sobre o modelo
([ajuste do limiar](https://scikit-learn.org/stable/modules/classification_threshold.html),
[vazamento de dados](https://scikit-learn.org/stable/common_pitfalls.html)).

Em qualquer modo, a tabela `online_inference` guarda, por nó e ciclo, a decisão de cada
modelo separada da decisão da combinação. Depois que o detector sinaliza uma janela,
regras determinísticas atribuem o tipo; elas não detectam nada sozinhas:

| Código | Tipo | Critério principal |
|---|---|---|
| A1 | Degradação de sinal | RSSI médio < −100 dBm, ou variabilidade alta com SNR < 0 |
| A2 | Perda de telemetria inferida | perda por lacunas de sequência > 30% |
| A3 | Queda do indicador de bateria | Δ bateria < −20 p.p. na janela |
| A4 | Possível instabilidade do enlace | variabilidade alta com índice de qualidade baixo |
| A5 | Suspeita de interferência | padrão RSSI/SNR compatível; não comprova jamming |
| A6 | Assimetria bidirecional elevada | \|RSSI ida − volta\| > 15 dB |
| A7 | Alternância do parent reportado | trocas de parent > 30% dos registros |

A6 descreve o enlace, não o nó. O alarme é registrado no nó cujas features foram
avaliadas, mas a evidência lista os enlaces assimétricos (vizinho, diferença em dB, RSSI
nos dois sentidos) e traz `origin = not_determined`: a causa pode estar em qualquer um
dos dois nós.

O score mede o afastamento em relação à referência de treino (isolamento no IF, erro de
reconstrução no LSTM-AE), e não a probabilidade de falha. Janelas sem dados suficientes, com valores
não finitos ou sem modelo treinado recebem estados próprios, distintos de "normal". Uma
janela em que alguma feature não foi observada não chega aos modelos. Exemplos: menos de
dois números de sequência para a perda, ou a bateria sem leitura recente. Para a bateria,
vale a última leitura dos 60 min anteriores, o contexto de uma decisão do LSTM-AE, porque
o firmware só a envia quando muda 2 p.p. ou a cada 5 min.

Cada alarme registra em `evidence.data_origin` quantas linhas sintéticas havia no contexto
da decisão. A exportação e a API mostram isso em `is_synthetic`/`synthetic_basis`.
Registros antigos, sem essa contagem, aparecem como não verificados (`null`).

## Instalação

Requer Python 3.12 ou mais novo (o lock inclui `scipy` 1.18, que não instala em 3.11). As
dependências são separadas em perfis, e `requirements-lock.txt` fixa as versões testadas. O
detector de `importacao_arquivos_csv/` tem requisitos próprios e roda também em 3.11.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt -c requirements-lock.txt          # núcleo
pip install -r backend/requirements-lstm.txt -c requirements-lock.txt     # LSTM-AE (padrão)
```

Sem o perfil LSTM, o backend usa o Isolation Forest e registra o motivo em
`fallback_reason`.

| Perfil | Arquivo | Acrescenta |
|---|---|---|
| Núcleo | `backend/requirements.txt` | API, MQTT, Isolation Forest |
| LSTM | `backend/requirements-lstm.txt` | PyTorch (para `ML_LSTM_ENABLED=true`) |
| PostgreSQL | `backend/requirements-postgres.txt` | driver psycopg2 |
| Scripts | `scripts/requirements.txt` | pandas e pyarrow (exportação e treino offline) |
| Desenvolvimento | `requirements-dev.txt` | todos os perfis, pytest e ruff |

O PyTorch padrão do PyPI no Linux inclui bibliotecas CUDA. Para a versão só de CPU,
acrescente `--extra-index-url https://download.pytorch.org/whl/cpu` à instalação.

## Configuração

1. **Backend.** Copie `.env.example` para `backend/.env`. Defina `JWT_SECRET_KEY`
   (com pelo menos 32 bytes aleatórios) e a senha de `ADMIN100_PASSWORD` (de 12 a 72 bytes).
   Sem senha, a conta não é criada. A API escuta só em `127.0.0.1` por padrão. Os demais
   parâmetros e seus padrões estão em `backend/config.py`.
2. **Broker.** O Mosquitto de `scripts/docker-compose.yml` exige usuário e senha, aplica
   ACL e escuta só em `127.0.0.1:1883`. Copie `scripts/mosquitto/config/acl.example` para
   `acl` e crie `password_file` com os comandos descritos no próprio exemplo. Use um
   usuário por gateway, preso à sua rede. Se outro broker já ocupa a porta 1883, suba com
   `MQTT_HOST_PORT=1884 docker compose up -d` e ajuste `MQTT_PORT`.
3. **Firmware.** Copie `firmware/include/credentials.h.example` para `credentials.h`
   (o arquivo é ignorado pelo Git) e preencha Wi-Fi e MQTT do gateway. Ajuste `NODE_ID` por
   placa e `NETWORK_ID` por rede em `firmware/platformio.ini`. Detalhes em
   [`firmware/README.md`](firmware/README.md).

## Execução

```bash
cd scripts && docker compose up -d && cd ..             # broker local autenticado
cd backend && python start.py                           # API em http://127.0.0.1:8000 (/docs)
```

```bash
# em outro terminal, a partir da raiz do repositório
pio run --project-dir firmware -e heltec_v2 -t upload           # nó (NODE_ID único)
pio run --project-dir firmware -e heltec_v2_gateway -t upload   # gateway (requer credentials.h)
```

Antes de receber dados, cadastre a rede (gateways e nós) pela API: `POST /api/networks`,
com token de administrador obtido em `POST /api/auth/login`. O backend só grava a
telemetria cujo endereço LoRa está cadastrado na rede indicada pelo tópico
(`lab/<rede>/uplink`); as demais mensagens são descartadas. O treino do detector
começa quando há histórico suficiente (veja `ML_MIN_TRAIN_SAMPLES*` em `config.py`).

## Exemplo sintético

O autoteste não usa hardware, broker nem banco:

```bash
python scripts/self_test_detection.py                     # IF, LSTM-AE, cascata e fusão
python scripts/self_test_detection.py --sem-lstm          # só o IF (ambiente sem PyTorch)
python scripts/self_test_detection.py --dados /tmp/autoteste --csv /tmp/decisoes.csv
```

O script gera séries contínuas em CSV no formato do firmware (beacon a cada 20 s) e as
avalia com o código do backend sem alterações (features, sequências, runners, pipeline e
regras), na configuração padrão. Treino, validação, calibração e teste ocupam períodos
separados:

- **Treino:** os 60% iniciais da referência normal de 24 h. Definem os pesos do LSTM.
- **Validação:** os 20% seguintes. Servem só para a parada antecipada do LSTM.
- **Calibração:** os 20% finais. Definem o limiar p99 do LSTM e o τ da fusão. O IF treina
  até o início deste bloco; nenhum modelo treina nele.
- **Teste:** cenários posteriores, com outra semente e os modelos congelados.

Cada cenário tem episódios de 60 min normais seguidos de 60 min do cenário. A decisão
avaliada é a do fim do episódio, quando a janela do IF e a sequência do LSTM já estão
inteiras no cenário.

| Cenário | Esperado | Critério de aceite |
|---|---|---|
| controle normal | normal | falso positivo ≤ 10% |
| RSSI ≈ −106 dBm, SNR ≈ −6 dB | anomalia A1 | detecção ≥ 90%; tipo correto ≥ 90% das detecções |
| perda de ~50% dos beacons | anomalia A2 | idem |
| queda de bateria de 30 p.p. em 15 min | anomalia A3 | idem |
| nó sem vizinhos | dados insuficientes | 100%, nunca "normal" |
| RSSI não finito | entrada inválida | 100%, nunca "normal" |
| modelo não treinado | sem modelo | 100%, nunca "normal" |

O relatório traz esperado × obtido para cada detector (IF, LSTM-AE, cascata e fusão, em
separado), com falsos positivos, anomalias não detectadas e alarmes no período de
calibração. O script termina com 0 se todos passam, 1 se algum reprova e 2 em erro de
execução. **Hoje ele reprova**; veja [Limitações](#limitações).

Para analisar dados exportados, use `scripts/export_dataset.py` e o detector de
`importacao_arquivos_csv/` (veja o README da pasta).

## Fluxo ponta a ponta sintético

`scripts/e2e_sintetico.py` percorre o caminho completo com componentes reais e
descartáveis, todos em `127.0.0.1` e com credenciais aleatórias a cada execução:

- Mosquitto em Docker, com a imagem do compose;
- backend (`backend/start.py`);
- banco SQLite;
- exportação e detector CSV.

Requer Docker, o perfil LSTM instalado e as portas 18830 e 18000 livres (ajustáveis por
opção). Leva cerca de 1 h com os padrões.

```bash
python scripts/e2e_sintetico.py --trabalho /tmp/e2e            # pasta vazia ou inexistente
python scripts/e2e_sintetico.py --trabalho /tmp/e2e --python .venv/bin/python --horas 26 \
  --normal-min 20 --anomalia-min 30
```

1. **Histórico reconstruído.** As 26 h de histórico **não são vividas em tempo real**: os
   beacons de dois nós passam pelo broker e pela ingestão normal em ritmo acelerado
   (cerca de 1,5 min). Com o backend parado, os carimbos são reescritos para o passado no
   ritmo de 20 s do firmware. Só o ritmo e a ordem reproduzem um nó real.
2. **Ao vivo.** LSTM-AE e IF em `shadow`, com beacons em tempo real. Depois do treino e de
   uma fase normal, a recepção do N3 cai para cerca de −106 dBm e −6 dB.
3. **Exportação e detector CSV,** com corte de treino no fim do histórico.

O resultado (`resultado.json`, na pasta de trabalho) separa:

- **Detecção:** se o N3 degradado foi sinalizado;
- **Localização:** se o alarme ficou no N3 e se um eventual A6 no vizinho N2 cita o enlace
  N2↔N3 sem atribuir a causa.

Todo o banco do E2E é sintético, mas fica gravado como `data_origin = real`. A política de
referência e o detector CSV recusam treinar com linhas sintéticas, e o roteiro precisa
exercitar o treino. O pacote leva `--source-id e2e-sintetico`.

Os ciclos da fase anômala, um por minuto, são observações repetidas de **uma única**
degradação. A fração de ciclos sinalizados mede a persistência da detecção nessa
ocorrência, não a sensibilidade sobre um conjunto de eventos independentes.

## Testes

```bash
pip install -r requirements-dev.txt -c requirements-lock.txt
python -m pytest
ruff check backend scripts firmware/apply_patch.py tests conftest.py importacao_arquivos_csv/codigos --select E9,F63,F7,F82
python scripts/check_publication.py
pio run --project-dir firmware -e heltec_v2 -e heltec_v2_gateway   # requer credentials.h (use o exemplo)
```

`check_publication.py` procura segredos, dados pessoais e arquivos que não devem ser
versionados; com `--history`, examina também os blobs do histórico Git. A triagem não
substitui a revisão manual nem um secret scanner.

Os testes de PostgreSQL rodam quando `TEST_POSTGRES_URL` aponta para um banco descartável.
Alguns testes do auditor de exportação exigem um repositório Git com commits e são
pulados, com o motivo indicado, quando não há `HEAD`.

Os identificadores públicos de artefatos usam nomes descritivos, como `dataset-v1`
e `features-v4`. Os leitores aceitam identificadores antigos com prefixo numerado
quando o formato e a versão coincidem. Modelos com outra versão ou ordem de features
continuam sendo recusados; os dados de bancos existentes não são reescritos.

## Limitações

- **Sensibilidade da detecção.** No autoteste com 5 sementes:
  - **LSTM-AE:** passa nos 7 cenários em 4 sementes. Na outra, reprova o controle, com 20%
    de falsos positivos para um limite de 10%. Esses alarmes ficam pouco acima do limiar
    (mediana de 1,13×), e o erro se espalha pelas features.
  - **Isolation Forest:** reprova a queda de bateria nas 5 sementes (3–20% de detecção) e a
    perda de pacotes em uma (87%). O score do IF não cresce quando o desvio passa da faixa
    vista no treino. Quando só uma das 11 features se desvia, ele não chega ao limiar.
  - **Cascata:** herda essas perdas.
  - **Fusão:** detecta 23–60% das quedas de bateria e teve 13% de falsos positivos numa
    semente.

  A referência sintética não tem estrutura temporal. O LSTM-AE reconstrói pouco mais que a
  média, e o bom resultado dele vem do erro crescer com a magnitude do desvio, não de
  padrões temporais. As regras A1–A7 só classificam o que o detector sinalizou. Nenhum
  resultado foi verificado com dados reais ou de hardware.
- **Dados ausentes.** Janelas com feature não observada ficam fora do treino e da
  decisão, com estado próprio. Num nó com perdas frequentes, isso reduz o histórico útil e
  pode atrasar o treino. A tolerância de 60 min para a bateria supõe o ritmo de envio do
  firmware desta versão. Nesse intervalo, uma queda cujos envios se perderam todos só
  aparece na próxima leitura recebida.
- **Localização.** As features de um nó incluem os enlaces com os vizinhos. Quando a
  recepção de um nó piora, o vizinho passa a ver o enlace assimétrico (`link_asymmetry`) e
  pode receber A6, ainda que nada tenha mudado nele. O alarme indica a assimetria e os dois
  nós do enlace, sem apontar a origem. Detecção e localização da origem precisam ser
  avaliadas em separado, como faz o fluxo ponta a ponta.
- **Laboratório.** MQTT sem TLS. Use só em rede isolada, com o broker local
  autenticado e nunca um broker público. Os comandos que trafegam pela malha LoRa
  (simulação, LED) não são autenticados: quem transmitir na mesma frequência e
  sync word pode enviá-los.
- **Frequência.** O padrão é 903,9 MHz (plano AU915, sub-banda 2, canal 0). **Antes de qualquer ensaio de
  transmissão**, confira a configuração de rádio com os requisitos da Anatel para
  equipamentos de radiação restrita. Verifique frequência, largura de banda, potência
  conduzida e EIRP com a antena usada, e ciclo de trabalho. No Brasil, as faixas são
  902–907,5 e 915–928 MHz; canais acima de 907,5 MHz neste plano ficam fora delas. O
  padrão de 20 dBm também precisa ser conferido. Fora do Brasil, vale a regulamentação
  local.
- **Hardware.** Nesta versão, o firmware foi compilado mas não exercitado em placas.
  Só a Heltec WiFi LoRa 32 V2 é suportada. O gateway envia `gw_rssi`/`gw_snr` da tabela
  de rotas e, sem entrada para o remetente, os valores fixos −85/0, que o backend não
  distingue de uma medida.
- **Beacon grande.** Um beacon acima de 221 bytes é descartado: a remontagem com menos
  vizinhos não funciona nesta versão. O Vext (GPIO21) é desligado após cada leitura de
  bateria e também alimenta o OLED; o efeito sobre o display não foi verificado em placa.
- **Concorrência no firmware.** O cliente MQTT do gateway é protegido por mutex. O display
  e o estado da simulação ainda são compartilhados entre tarefas sem proteção.
- **Modelos salvos.** Os pacotes de modelo congelados (`joblib`) e o `if_model.joblib` dos
  scripts executam código ao serem carregados. Carregue apenas arquivos gerados por você;
  nunca use modelos recebidos de terceiros.
- **Backend.** Um único processo: o laço de detecção, o cliente MQTT e o limite de
  tentativas de login vivem na memória dele. O banco ainda contém tabelas e colunas
  legadas de análise por LLM, migradas apenas para manter bancos antigos atualizáveis.
- **Scripts offline.** `extract_features.py`, `train_if.py`, `train_lstm.py`,
  `train_cascade.py` e `inject_anomalies.py` reproduzem um estudo anterior e
  têm defeitos conhecidos, descritos no cabeçalho de cada um. A divisão temporal
  desses scripts não tem intervalo de purga no corte.
- **CSV.** Os CSVs gerados preservam os textos da entrada. Ao abri-los numa
  planilha, desative a interpretação de fórmulas.

Falsos positivos, sensibilidade, localização da origem, calibração e concorrência ficam
para as próximas versões. Antes de qualquer conclusão sobre desempenho, o projeto ainda
precisa ser validado com dados reais e nas placas.

## Licença

MIT. Veja [LICENSE](LICENSE). As atribuições de bibliotecas de terceiros, incluindo o
patch aplicado à LoRaMesher, estão em [NOTICE](NOTICE).
