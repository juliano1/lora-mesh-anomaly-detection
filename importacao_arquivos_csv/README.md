# Análise independente dos CSVs exportados

Esta pasta reúne códigos para detectar outliers nos pacotes gerados por
`scripts/export_dataset.py`. O processamento acontece fora do sistema, usando os
arquivos exportados e bibliotecas conhecidas. Cada execução produz resultados identificáveis e
um relatório que quem for reproduzir o experimento consegue interpretar.

O primeiro detector é o **Isolation Forest do scikit-learn**, identificado como
`detector_001_isolation_forest_sklearn`.

## Estado da implementação

| Item | Estado |
|---|---|
| Script `codigos/detector_001_isolation_forest_sklearn.py` | Implementado, versão 1.0.0. |
| Dependências `codigos/requirements_001_isolation_forest_sklearn.txt` | Versões testadas com Python 3.11.14 e 3.12.3. |
| Testes `codigos/testes/` | 71 testes com dados sintéticos (leitura, contrato, filtros, partições, modelo, saídas e relatório). |

## Organização

```text
importacao_arquivos_csv/
├── README.md
├── .gitignore
├── codigos/
│   ├── detector_001_isolation_forest_sklearn.py
│   ├── requirements_001_isolation_forest_sklearn.txt
│   └── testes/                      # pytest, só com dados sintéticos
├── entrada_csv/
│   └── README.md
└── resultados/
    ├── README.md
    └── MODELO_RELATORIO.md
```

| Local | Finalidade |
|---|---|
| `codigos/` | Scripts de detecção e suas dependências. Cada detector recebe um número e um nome que identifica algoritmo e framework. |
| `entrada_csv/` | Pacotes exportados e extraídos, preservados sem edição. Manter cada exportação em sua própria pasta. |
| `resultados/` | Uma pasta nova para cada execução, com relatório, tabelas, gráficos e metadados de reprodução. |

Os dados de entrada e as execuções geradas ficam fora do Git pelas regras locais
de `.gitignore`. Os documentos e os códigos podem ser versionados. Para que outra
pessoa reproduza um experimento, compartilhe também o pacote de entrada e a pasta da
execução; um clone do repositório não contém esses dados.

## Qual arquivo usar primeiro

Exporte com `--kind features` ou `--kind full` e use o `features.csv`. Cada linha
representa um dispositivo em uma janela de tempo. Preserve junto dele o
`manifest.json` e o `schema.json`, que identificam o pacote, os parâmetros e as
colunas. Para o arquivo científico, prefira guardar o pacote `full` original.

O pacote `raw` traz medições que exigem outro processamento antes de alimentar
este primeiro detector. `coverage.csv` resume a cobertura, e
`simulation_runs.csv` descreve simulações; eles não substituem o `features.csv`.

## Como começar

1. Exporte com `scripts/export_dataset.py` e extraia o pacote em `entrada_csv/`,
   mantendo a pasta interna do ZIP.
2. Instale as dependências num ambiente virtual ([Instalação](#instalação)).
3. Execute o script sobre a pasta que contém `features.csv`, `manifest.json` e
   `schema.json` ([Execução](#execução)). Também é possível indicar diretamente o
   `features.csv`, com proveniência limitada.
4. Abra o relatório HTML da execução e consulte os CSVs quando precisar analisar
   as janelas sinalizadas. O [modelo do relatório](resultados/MODELO_RELATORIO.md)
   define o conteúdo esperado.

## Instalação

Requer Python 3.11 ou mais novo. A partir da raiz do repositório:

```bash
python3 -m venv importacao_arquivos_csv/.venv
importacao_arquivos_csv/.venv/bin/python -m pip install \
  -r importacao_arquivos_csv/codigos/requirements_001_isolation_forest_sklearn.txt
```

No Windows, use `importacao_arquivos_csv\.venv\Scripts\python` no lugar de
`importacao_arquivos_csv/.venv/bin/python`. A pasta `.venv/` fica fora do Git.
O script não usa banco de dados, backend, API nem rede.

## Execução

A partir da raiz do repositório, com o Python do ambiente virtual:

```bash
importacao_arquivos_csv/.venv/bin/python importacao_arquivos_csv/codigos/detector_001_isolation_forest_sklearn.py \
  --entrada importacao_arquivos_csv/entrada_csv/<pasta_do_dataset> \
  --corte-treino 2026-10-16T00:00:00Z
```

A data é só um exemplo de sintaxe. Escolha o corte e, se for o caso, o início da
referência pela cronologia da campanha e pelos eventos conhecidos; o script não
fixa datas da campanha.

| Opção | Padrão | Uso |
|---|---|---|
| `--entrada` | obrigatória | Pasta extraída do pacote ou caminho de um `features.csv`. Se apontar a pasta externa do ZIP, a mensagem indica a pasta interna. |
| `--corte-treino` | — | Instante ISO 8601 com fuso explícito (`Z` ou offset), convertido para UTC. Treino candidato: fim da janela ≤ corte. |
| `--fracao-treino` | `0.70` | Só sem `--corte-treino`: corte no quantil dos fins das janelas `valid`, arredondado ao minuto. Convenção exploratória. |
| `--inicio-treino` | — | Opcional: janelas que começam antes desse instante ficam fora da referência (motivo `treino:antes_inicio_referencia`). Serve para excluir períodos de bancada ou instalação. |
| `--percentil-limiar` | `99` | Percentil dos scores do treino efetivo usado como limiar; `0 < p < 100`. |
| `--seed` | `42` | `random_state` do Isolation Forest. |
| `--n-estimators` | `300` | Número de árvores. |
| `--saida` | `importacao_arquivos_csv/resultados/` | Onde criar a pasta da execução; o padrão é resolvido pelo local do script. |

Códigos de saída: `0` concluída; `1` falha inesperada depois de criada a pasta
(estado `falhou` no `execucao__<run_id>.json`); `2` erro de uso ou de entrada, sem
nada gravado; `3` dados insuficientes, com pasta de diagnóstico sem detecção (só
fica sem pasta quando não há janela `valid` para calcular o corte automático).
Enquanto a análise roda, o JSON da execução fica com estado `em_execucao`.

### O que o script faz

1. Confere o SHA-256 de `features.csv` e de `schema.json` contra o `manifest.json`,
   as 11 features contra `model_input_columns` do schema, o número de linhas, a
   fonte, a janela e o passo; para em qualquer divergência. Sem metadados, usa o
   identificador `csv_<12 primeiros caracteres do SHA-256>` e registra a
   proveniência limitada. Para cada versão dos dados (`feature_schema_version`,
   `extractor_revision`, `extraction_policy`), `versoes_fonte` no JSON da execução
   diz se ela veio do `manifest.json` ou do próprio `features.csv`.
2. Lê datas como ISO 8601 em UTC, os campos JSON com `json` e os booleanos `0`/`1`;
   valores fora do contrato, chaves repetidas `(source_id, node_id,
   window_end_utc)`, mais de uma fonte ou de uma configuração de janela/passo são
   erros de entrada.
3. Uma linha é analisável se `feature_status = valid`, as 11 features são finitas,
   `defaulted_features = []` e `unavailable_features = {}`. Não há imputação.
4. Separa no tempo com um corte único: treino candidato (fim ≤ corte), folga de uma
   janela (corte < fim ≤ corte + W) e teste (fim > corte + W).
5. Treino efetivo: treino candidato analisável com `reference_eligible_by_policy =
   1`, `simulation_overlap = 0` e origem `real` ou `unknown` (com ressalva). O teste
   analisa todas as linhas analisáveis, de qualquer origem.
6. Ordena por (`window_end_utc`, `node_id`), ajusta `RobustScaler` +
   `IsolationForest` só no treino efetivo, calcula `anomaly_score =
   -score_samples` (maior = mais anômalo) e sinaliza o teste com `anomaly_score >
   limiar`, sendo o limiar o percentil configurado dos scores do treino.

### Colunas acrescentadas em `predicoes__<run_id>.csv`

As colunas originais vêm como lidas, na ordem do arquivo; células vazias e texto
vazio saem como nulo. Os CSVs seguem a convenção do exportador: UTF-8, vírgula,
valor não nulo entre aspas, nulo = célula vazia sem aspas.

| Coluna | Conteúdo |
|---|---|
| `numero_linha_original` | Linha de dados no `features.csv` (1 = primeira linha depois do cabeçalho). |
| `run_id`, `detector`, `detector_versao` | Identidade da execução e do detector. |
| `particao_temporal` | `treino_candidato`, `folga` ou `teste`. |
| `estado_analise` | `treino_efetivo`, `excluido_treino`, `folga`, `teste_analisado` ou `excluido_teste`. Numa execução encerrada por dados insuficientes, as linhas que seriam pontuadas saem como `treino_sem_modelo` ou `teste_sem_modelo`, com motivo `execucao:dados_insuficientes`. |
| `motivos_exclusao` | Códigos separados por `;`, com prefixo `treino:`, `teste:` ou `folga:` (ex.: `treino:simulacao_sobreposta`, `teste:default_declarado`). |
| `anomaly_score`, `limiar` | Preenchidos só no treino efetivo (diagnóstico) e no teste analisado. |
| `is_outlier` | `1` ou `0` só no teste analisado; vazio em todas as outras linhas. |

### Colunas de `resumo_por_dispositivo__<run_id>.csv`

Uma linha por `node_id`. As contagens são de janelas (linhas de `features.csv`).

| Coluna | Conteúdo |
|---|---|
| `node_id`, `node_name` | Dispositivo; o nome é o primeiro `node_name` não vazio do dispositivo, na ordem do arquivo. |
| `janelas_entrada`, `janelas_valid` | Todas as janelas do dispositivo e as com `feature_status = valid`. |
| `janelas_analisaveis` | Janelas que passam na qualidade, em qualquer partição (inclui as da folga). |
| `treino_candidato`, `treino_efetivo`, `excluido_treino` | Partição de treino e sua divisão; `treino_efetivo + excluido_treino = treino_candidato`. |
| `folga` | Janelas da folga entre treino e teste, sem score. |
| `teste`, `teste_analisado`, `excluido_teste` | Partição de teste e sua divisão; `teste_analisado + excluido_teste = teste`. |
| `outliers_teste` | Janelas do teste analisado com `anomaly_score > limiar`. |
| `taxa_outliers_teste` | Fração de 0 a 1 (`outliers_teste ÷ teste_analisado`), não percentual; `não aplicável` quando o teste analisado é 0. |
| `teste_analisado_com_simulacao`, `outliers_com_simulacao` | As mesmas contagens restritas a `simulation_overlap = 1`. |
| `score_max_teste` | Maior `anomaly_score` no teste analisado; vazio sem teste analisado. |
| `primeira_janela_fim_utc`, `ultima_janela_fim_utc` | Primeiro e último fim de janela do dispositivo no arquivo. |
| `sem_referencia_propria` | `1` quando o dispositivo não tem nenhuma janela no treino efetivo; o modelo global ainda o pontua. |

Numa execução encerrada por dados insuficientes, `outliers_teste`, `taxa_outliers_teste`,
`outliers_com_simulacao` e `score_max_teste` ficam vazios, as demais contagens
mostram o que seria treino e teste, e não são gravados `outliers__<run_id>.csv`
nem os gráficos.

### Testes

Os testes usam só dados sintéticos, gerados em pastas temporárias:

```bash
importacao_arquivos_csv/.venv/bin/python -m pip install pytest==9.2.0
importacao_arquivos_csv/.venv/bin/python -m pytest importacao_arquivos_csv/codigos/testes -q
```

## Exemplo

Com um banco de teste alimentado por telemetria sintética (três nós, 8 min), a
exportação `full` gera 30 linhas em `features.csv`, das quais 18 `valid`. O detector
termina com código `3` e estado `dados_insuficientes`, porque não sobra janela no
teste depois do corte e da folga. É o comportamento esperado: com pouco histórico, o
resultado é "sem decisão", nunca "normal". Uma análise real precisa de dias de dados e
de um corte escolhido pela cronologia da coleta.

## Identificação dos códigos e das saídas

O primeiro detector é `codigos/detector_001_isolation_forest_sklearn.py`.
Outros algoritmos receberão números distintos; correções no mesmo detector são
identificadas pela versão do código e pelo seu hash, registrados na execução.

Cada execução gera:

```text
resultados/
└── detector_001_isolation_forest_sklearn__<dataset>__<dataUTC>__<idcurto>/
    ├── relatorio__<run_id>.html
    ├── relatorio__<run_id>.md
    ├── predicoes__<run_id>.csv
    ├── outliers__<run_id>.csv
    ├── resumo_por_dispositivo__<run_id>.csv
    ├── execucao__<run_id>.json
    ├── ambiente__<run_id>.txt
    └── graficos/
```

O `run_id` combina detector, identificação segura do dataset, data e hora UTC e
um sufixo único. Os metadados preservam o `dataset_id` original, o SHA-256 do CSV,
a versão do código, os parâmetros, as bibliotecas e o comando executado. Uma nova
execução cria outra pasta e preserva os resultados anteriores.

## Como interpretar a detecção

Uma janela sinalizada é uma **candidata a anomalia estatística**. A confirmação
de falha física exige evidências de campo ou rótulos independentes. O relatório
deve mostrar quantas janelas foram analisadas e excluídas, os motivos, os resultados
por dispositivo e o limiar usado, inclusive quando nenhum outlier for encontrado.

O detector aplica separação temporal com folga entre treino e teste, tratamento
explícito de valores ausentes e defaults e identificação das simulações. Dados
sintéticos podem ser analisados no teste, mas não entram no treino de referência.
Origem `unknown` precisa ser declarada; ela não comprova origem real nem saúde.

A execução ajusta seu próprio modelo a partir do CSV. Comparações com o detector
online dependem também de seus dados de treino, parâmetros e políticas.

## Documentação de apoio

- [Scripts de exportação e importação](../scripts/README.md).
- [Definição das features](../backend/ai/anomaly_definitions.py).
- [Documentação oficial do Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).
