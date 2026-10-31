# Relatório de detecção de outliers — Isolation Forest

**Modelo editorial. Não contém uma execução nem resultados medidos.**
Os campos entre `<...>` devem ser preenchidos pelo programa a partir da execução.

## Identificação

| Campo | Valor |
|---|---|
| Execução | `<run_id>` |
| Detector / versão | `detector_001_isolation_forest_sklearn / <versao>` |
| Dataset | `<dataset_id ou identificador derivado do hash do CSV>` |
| Arquivo analisado / SHA-256 | `<arquivo> / <sha256>` |
| Data da análise | `<data e hora UTC>` |
| Período / dispositivos | `<inicio e fim UTC> / <dispositivos>` |
| Janela / passo | `<W segundos> / <S segundos>` |
| Proveniência | `<metadados presentes e verificações realizadas>` |

## 1. Resumo executivo

Foram recebidas `<N>` janelas de `<D>` dispositivos. Após os filtros de qualidade
e a separação temporal, `<T>` janelas compuseram a referência de treino e `<E>`
foram analisadas no teste. Foram sinalizadas `<A>` janelas, correspondentes a
`<taxa>%` das janelas efetivamente analisadas no teste.

Apresente os dispositivos e períodos que merecem investigação, sem atribuir causa
física aos resultados. Se não houver detecções, registre isso explicitamente.
Janelas excluídas não entram no denominador da taxa de detecção.

## 2. Dados e qualidade

Apresente as contagens de entrada, treino, folga, teste, estados de qualidade,
defaults, valores não finitos, origens e simulações. As exclusões devem ter motivos
e contagens auditáveis; explique quando um registro tiver mais de um motivo.
Mostre cobertura temporal e contagens por dispositivo, incluindo dispositivos
sem janelas analisáveis.

Registre versões do schema e do extrator, identidade da fonte e limites da
proveniência. A origem `unknown` deve aparecer como desconhecida.

## 3. Método e reprodução

Descreva as 11 entradas e sua ordem, a política de qualidade, a seleção da
referência, o corte UTC comum a todos os dispositivos e a folga entre treino e
teste. Informe se o corte foi automático ou fornecido pelo usuário.

Registre o modelo global, o pré-processamento ajustado apenas no treino, os
hiperparâmetros, a semente, as versões das bibliotecas e o comando executado.
Defina o score, sua direção e a regra de decisão com o limiar numérico usado.
O score expressa isolamento estatístico; não é probabilidade de falha.

## 4. Resultados no período de teste

| Dispositivo | Treino | Teste analisado | Janelas sinalizadas | Taxa no teste |
|---|---|---|---|---|
| `<dispositivo>` | `<n>` | `<n>` | `<n>` | `<taxa ou não aplicável>` |

Inclua a evolução temporal dos scores com o limiar e os sinais de outlier e uma
distribuição dos scores separando treino e teste. Identifique janelas com
simulação. Apresente as até 20 janelas de teste com maior score, com dispositivo,
intervalo UTC, origem e informação de simulação.

Mostre os valores observados das features das principais janelas como apoio à
inspeção. Sua diferença em relação ao treino é uma descrição, não uma explicação
causal do Isolation Forest.

## 5. Simulações e evidências de campo

Informe quantas janelas do teste têm sobreposição com simulação e quantas foram
sinalizadas. Deixe explícito se nenhuma simulação ficou no teste. Use esse
cruzamento como análise exploratória da intervenção, sem chamar automaticamente
as demais janelas de normais ou calcular acurácia sem rótulos verificados.

Indique os arquivos de anotações e evidências necessários para investigar os
achados. O `events_template.csv` original é um formulário vazio, não um conjunto
de rótulos já validado.

## 6. Limitações e interpretação

Descreva as limitações efetivamente presentes: defaults e exclusões, origem
desconhecida, amostra de treino pequena, diferenças entre dispositivos, features
constantes, ausência de rótulos e artefatos da coleta documentados nos manuais.
Declare que a elegibilidade de referência não comprova saúde e que o limiar
estimado no próprio treino é uma escolha exploratória, sem validação independente.

Explique que janelas sobrepostas são dependentes; `<A>` janelas sinalizadas não
significam `<A>` falhas distintas. Ausência de detecções não comprova ausência de
falhas. Não apresente precisão, recall, F1 ou diagnóstico físico sem evidência.

## 7. Arquivos e próximos passos

Liste os artefatos da execução e seus nomes. Indique as janelas prioritárias para
conferência com os dados brutos, as anotações de campo e as simulações. Para uso
em publicação, recomende definir previamente uma referência adequada e calibrar o
limiar em um período independente antes da avaliação final.
