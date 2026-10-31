"""
Detecção de anomalias da rede mesh LoRa.

Detecção contínua (pipeline online em ``ai.online``: Isolation Forest e LSTM-AE,
isolados ou combinados) e regras A1–A7 compartilhadas (``ai.anomaly_definitions``).

As regras A1–A7 só classificam o que um detector já sinalizou (sinal, pacotes, energia,
enlace, interferência, assimetria, topologia); não detectam nada sozinhas.
Os submódulos são importados explicitamente; este pacote não carrega nada na importação.
"""
