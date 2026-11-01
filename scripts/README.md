# Scripts

Utilitários de linha de comando, executados a partir da raiz do repositório. Os scripts
que usam o backend precisam de `scripts/requirements.txt`; `train_lstm.py` e
`train_cascade.py` exigem também `backend/requirements-lstm.txt`.

| Script | Uso |
|---|---|
| `self_test_detection.py` | Autoteste da detecção com dados sintéticos em CSV (veja o README principal). |
| `e2e_sintetico.py` | Fluxo ponta a ponta sintético, com broker em Docker, backend, banco SQLite, exportação e detector CSV descartáveis. O histórico é reconstruído por envio acelerado e reescrita dos carimbos (veja o README principal). |
| `export_dataset.py` | Exporta uma rede e um período do banco (ou de uma cópia SQLite) para um pacote ZIP com manifesto e hashes. |
| `import_dataset.py` | Reconstrói um pacote exportado como banco SQLite. |
| `audit_dataset_export.py` | Confere um pacote contra o banco de origem e grava um relatório JSON. |
| `monitor_mqtt.py` | Mostra as mensagens de um tópico do broker (`MQTT_USER`/`MQTT_PASS` do ambiente). |
| `check_publication.py` | Triagem de segredos, dados pessoais e arquivos proibidos (também roda na CI). |
| `extract_features.py`, `train_if.py`, `train_lstm.py`, `train_cascade.py`, `inject_anomalies.py` | Reprodução de um estudo anterior; têm defeitos conhecidos, descritos no cabeçalho de cada um. Não use para novas avaliações. |

## Exportação

```bash
mkdir -p data/exports
python scripts/export_dataset.py --network-id 1 \
  --start 2026-01-01T00:00:00Z --end 2026-01-02T00:00:00Z \
  --kind full --source-id laboratorio --out data/exports/exportacao.zip
```

O pacote contém dados reais da rede: mantenha-o em `data/`, que o `.gitignore` exclui.

- **Origem:** sem `--sqlite`, o script lê o banco de `DATABASE_URL`; com
  `--sqlite <arquivo>`, lê uma cópia aberta somente para leitura.
- **Tipo de pacote (`--kind`):** `raw` traz os registros brutos, `features` traz as
  11 features por nó e janela, e `full` traz os dois.
- **Limites:** o fim do período não pode estar no futuro, e o intervalo máximo é
  `DATASET_EXPORT_MAX_DAYS`.

O pacote `full` alimenta o detector de `importacao_arquivos_csv/`. As demais opções
aparecem com `--help`.

## Broker local

`docker-compose.yml` sobe o Mosquitto em `127.0.0.1:1883`, com autenticação obrigatória
e ACL por usuário. A imagem é fixada por digest. O projeto do compose se chama
`lora-mesh-anomaly-detection` e não fixa nome de contêiner, para não colidir com outros
brokers; se a porta 1883 estiver ocupada, use `MQTT_HOST_PORT=1884 docker compose up -d`.
Antes de subir:

1. copie `mosquitto/config/acl.example` para `mosquitto/config/acl`;
2. crie `mosquitto/config/password_file` com os comandos do início do exemplo.

Os dois arquivos são ignorados pelo Git.

```bash
cd scripts && docker compose up -d && cd ..
python scripts/monitor_mqtt.py --topic 'lab/+/uplink'     # com MQTT_USER/MQTT_PASS do backend
```

Tamanho máximo de pacote MQTT: 9216 bytes. O backend descarta uplinks acima de 8 KiB.
