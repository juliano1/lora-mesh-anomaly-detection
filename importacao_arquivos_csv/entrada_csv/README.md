# Arquivos de entrada

Coloque aqui os pacotes gerados por `scripts/export_dataset.py`. Para o detector 001,
use `--kind features` ou `--kind full`.

Extraia cada ZIP mantendo sua pasta interna. Exemplo de organização:

```text
entrada_csv/
├── README.md
├── completo_<dataset_id>.zip
└── <dataset_id>/
    ├── features.csv
    ├── manifest.json
    ├── schema.json
    └── ... demais arquivos do pacote ...
```

A entrada preferencial do script é a pasta que contém os três arquivos
acima. Um `features.csv` isolado também pode ser usado, com a limitação de
proveniência registrada no relatório quando os metadados não estão presentes.

Preserve o ZIP e os arquivos extraídos sem edição. Evite salvar o CSV pelo Excel,
pois isso pode alterar datas, separador e casas decimais. Se precisar de outra
seleção, gere uma nova exportação ou registre uma derivação separadamente.

Não misture linhas de exportações diferentes no mesmo CSV. Nesta primeira versão,
cada execução analisa uma exportação, com uma configuração de janela e passo.
Os arquivos são locais e ignorados pelo Git; mantenha sua cópia de arquivo junto
das evidências do experimento.

Veja o [README da pasta](../README.md) e o [README dos scripts](../../scripts/README.md).
