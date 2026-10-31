# Resultados das detecções

Cada execução cria uma pasta identificada por detector, dataset, data e hora
UTC e um sufixo único. Os arquivos também levam esse identificador (`run_id`), para
continuarem reconhecíveis quando forem enviados separadamente.

| Saída | Conteúdo |
|---|---|
| `relatorio__<run_id>.html` | Relatório em português, para abrir no navegador, com resumo, metodologia, qualidade, resultados, gráficos e limitações. Funciona sem internet, com os gráficos incorporados. |
| `relatorio__<run_id>.md` | Versão textual do mesmo relatório, com referências aos gráficos da execução. |
| `predicoes__<run_id>.csv` | Todas as linhas da entrada, com identificação, partição temporal, estado da análise, motivo de exclusão e resultados quando calculados. |
| `outliers__<run_id>.csv` | Somente as janelas sinalizadas no teste, em ordem decrescente de score. Continua tendo cabeçalho quando vazio. |
| `resumo_por_dispositivo__<run_id>.csv` | Contagens por dispositivo: entrada, treino, teste analisado, exclusões e outliers; taxa calculada sobre o teste analisado. |
| `execucao__<run_id>.json` | Identidade e hashes da entrada e do código, parâmetros, corte temporal, folga, limiar, contagens e limitações da execução. |
| `ambiente__<run_id>.txt` | Versões do Python e das bibliotecas usadas e comando de reprodução. |
| `graficos/` | Figuras identificadas pela execução, com eixos, unidades e legenda. |

O [modelo de relatório](MODELO_RELATORIO.md) descreve a apresentação esperada.
Ele é um modelo editorial, sem resultados calculados.

Janelas não analisadas têm score e decisão vazios, acompanhados de motivo;
não aparecem como normais. Os resultados principais dizem respeito ao
período de teste, posterior ao treino e à folga. Contagens de janelas não equivalem
a contagens de incidentes independentes, pois as janelas podem se sobrepor.

As pastas geradas são ignoradas pelo Git. Preserve a execução inteira junto do
pacote de entrada antes de compartilhar ou citar os resultados. O HTML pode ser usado
sozinho, porque os gráficos estão incorporados nele; o Markdown usa os arquivos de
`graficos/`.
