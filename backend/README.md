# Backend

API FastAPI, cliente MQTT e detecção de anomalias, num único processo que:

1. assina `lab/+/uplink` e `lab/+/send_result` no broker local;
2. valida cada uplink (tamanho, JSON finito, faixas, endereço LoRa cadastrado na rede
   do tópico) e grava nós, telemetria e vizinhos;
3. marca como offline os nós sem dados há `NODE_TIMEOUT_SECONDS`;
4. a cada ciclo, calcula as 11 features por nó e janela, treina ou recarrega o
   detector e registra as anomalias;
5. expõe a API REST com autenticação JWT.

A instalação, o `.env` e o broker estão descritos no [README principal](../README.md).

## Execução

```bash
cd backend
python start.py            # 127.0.0.1:8000; UVICORN_RELOAD=true recarrega ao editar
```

Ao iniciar, o backend:

- valida `JWT_SECRET_KEY` e as senhas de bootstrap;
- cria as tabelas e aplica as migrações de `STARTUP_MIGRATIONS` e `SERVICE_MIGRATIONS`
  (`main.py`). Qualquer falha impede a subida do serviço, com o nome da migração no erro;
- cria as contas `admin100` e `admin200` que tiverem senha definida;
- inicia o cliente MQTT. Se o broker estiver fora, o cliente continua tentando, com espera
  de 1 a 60 s, e refaz as assinaturas a cada conexão.

`GET /health` responde `ok` com o broker conectado e o detector rodando, e `degraded` caso
contrário. Os campos `mqtt_connected` e `detector` dizem qual dos dois falta.

Use um único worker: o laço de detecção, o cliente MQTT e o limite de tentativas de
login ficam na memória do processo.

## Configuração

Todos os parâmetros estão em `config.py`, com o padrão e o significado de cada um.
Os principais:

| Variável | Padrão | Uso |
|---|---|---|
| `JWT_SECRET_KEY` | — | Obrigatória, com pelo menos 32 bytes. |
| `ADMIN100_PASSWORD`, `ADMIN200_PASSWORD` | vazio | Senha de bootstrap (12 a 72 bytes); vazio não cria a conta. |
| `API_HOST`, `API_PORT` | `127.0.0.1`, `8000` | Expor em outra interface exige TLS e controle de acesso à frente. |
| `CORS_ORIGINS` | vazio | Origens de navegador permitidas; vazio desativa o CORS. |
| `MQTT_BROKER`, `MQTT_PORT`, `MQTT_USER`, `MQTT_PASS` | `localhost`, `1883`, vazio | Broker local autenticado. |
| `DATABASE_URL` | SQLite em `backend/lora_mesh.db` | PostgreSQL é opcional (`requirements-postgres.txt`). |
| `NODE_TIMEOUT_SECONDS` | `240` | Equivale a 12 intervalos de beacon de 20 s. |
| `ML_WINDOW_MINUTES` | `15` | Duração da janela das features. |
| `ML_MIN_TRAIN_SAMPLES` | `10` | Janelas de referência exigidas para o primeiro treino. |
| `ML_CONTAMINATION` | `0.03` | Fração esperada de anomalias na referência. |
| `ML_LSTM_ENABLED` | `true` | LSTM-AE é o primário padrão (experimental). Sem o PyTorch, ou com `false`, o pipeline usa o IF e registra `fallback_reason`. |
| `LOGIN_MAX_FAILURES`, `LOGIN_LOCKOUT_SECONDS` | `5`, `900` | Bloqueio temporário por cliente e usuário. |

Os parâmetros do detector também podem ser alterados com o backend rodando, por
`PATCH /api/settings/ml` (tabela `mlsettings`). O campo `online` escolhe os modelos:
`primary` (`lstm_ae`, o padrão, ou `isolation_forest`), `secondary` (o outro, ou nulo) e
`combo` (`single`, `shadow`, `cascade` ou `fusion`). `cascade` e `fusion` são
experimentais: a resposta traz `default_primary`, `experimental_combos` e `experimental`.
Cada ciclo grava em `online_inference` a decisão de cada modelo e a da combinação,
inclusive as normais e as indisponíveis.

## API

A documentação interativa fica em `/docs`. `/`, `/health`, `/api/auth/login`, `/docs`,
`/redoc` e `/openapi.json` são públicos. As demais rotas exigem
`Authorization: Bearer <token>`. As marcadas com **admin** exigem conta de administrador.
O login limita falhas por cliente e usuário, aplica também um limite de 30 tentativas
por cliente a cada 60 segundos e expira o estado em memória. Ao atingir a capacidade
de acompanhamento do processo, novas chaves recebem 429 até haver espaço.

| Rota | Uso |
|---|---|
| `POST /api/auth/login` | Recebe `{"username", "password"}` e devolve o token; responde 429 após falhas repetidas. |
| `GET/POST/PUT/DELETE /api/networks[/{id}]` | Cadastro de redes, gateways e dispositivos (escrita: **admin**). |
| `GET /nodes`, `GET /history/{node_id}`, `GET /topology` | Nós, telemetria e topologia. |
| `GET /api/anomalies`, `/stats`, `/total`, `/export` | Anomalias registradas. |
| `PATCH /api/anomalies/{id}/resolve` | Marca uma anomalia como resolvida. |
| `GET/PATCH /api/settings/ml` | Parâmetros do detector (escrita: **admin**). |
| `GET /api/ml/online/status`, `POST /api/ml/online/retrain` | Estado e retreino do detector online (retreino: **admin**). |
| `POST/GET/DELETE /api/nodes/{id}/simulate`, `GET /api/simulation/...` | Simulação de anomalias no nó por comando LoRa (escrita: **admin**). |

Exemplo de cadastro, com valores fictícios:

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username": "admin100", "password": "<senha definida no .env>"}' \
  | python -c 'import json, sys; print(json.load(sys.stdin)["access_token"])')

curl -s -X POST http://127.0.0.1:8000/api/networks \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d '{
    "name": "laboratorio",
    "gateways": [{"name": "GW", "lora_address": "0x0001", "device_model": "Heltec WiFi LoRa 32 V2"}],
    "devices": [{"name": "N2", "lora_address": "0x0002"}, {"name": "N3", "lora_address": "0x0003"}]
  }'
```

O endereço LoRa (4 dígitos hexadecimais) é o que o nó exibe no display e envia no
campo `a` do beacon. Os campos opcionais de ambiente e bateria estão em `/docs`.

## MQTT

| Tópico | Sentido | Conteúdo |
|---|---|---|
| `lab/<rede>/uplink` | gateway → backend | Beacon do nó, mais RSSI/SNR medidos no gateway (`gw_rssi`, `gw_snr`). |
| `lab/<rede>/send_result` | gateway → backend | `{"msg_id", "dest", "status": "sent" \| "failed", ...}`. |
| `lora/mesh/<rede>/command` | backend → gateway | `{"cmd": "send" \| "broadcast" \| "ping", "dest", "msg", "msg_id", "reliable"}`. |
| `lora/status` | backend → broker | Mudanças online/offline dos nós. |

São descartadas, com registro no log, as mensagens:

- acima de 8 KiB, com NaN/Infinity ou com mais de 16 vizinhos;
- com valores fora das faixas aceitas;
- de endereço não cadastrado na rede do tópico.

Resultados de envio só atualizam comandos unicast cujo nó e endereço de destino
coincidem com o registro e pertencem à rede ativa do tópico. Registros legados sem
destino inequívoco não são atualizados por confirmação do gateway.

Se a publicação MQTT da parada de uma simulação falhar, a API responde 503 e mantém
o estado anterior para permitir nova tentativa. Publicação aceita não comprova que
o firmware já aplicou o comando.

O formato do beacon está em [`firmware/README.md`](../firmware/README.md).

## Estrutura

```
backend/
├── main.py                 # API, ciclo de vida, migrações na inicialização
├── start.py                # uvicorn com um worker
├── config.py               # parâmetros e validação de segurança
├── auth.py, routers/auth.py, schemas/auth.py   # JWT, login e bloqueio
├── database.py, models.py  # engine e modelos SQLModel
├── mqtt_service_sync.py    # cliente MQTT, validação do uplink, laço de detecção
├── node_monitor.py         # online/offline
├── seed_users.py           # contas de bootstrap
├── ai/                     # features, regras A1–A7, detectores e pipeline online
├── dataset_export/         # exportação e importação de pacotes verificáveis
├── migrations/             # migrações da inicialização (SQLite e PostgreSQL) e 8 scripts antigos não executados (7 avulsos em sqlite3)
└── tests/
```

## Testes

Rode `python -m pytest` a partir da raiz do repositório. Os testes de PostgreSQL só
rodam quando `TEST_POSTGRES_URL` aponta para um banco descartável; eles criam e apagam
um schema próprio.
