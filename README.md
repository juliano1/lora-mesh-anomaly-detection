# lora-mesh-anomaly-detection

Monitoring and anomaly detection for LoRa mesh networks built with [LoRaMesher](https://github.com/LoRaMesher/LoRaMesher) on ESP32 (Heltec WiFi LoRa 32). The system combines machine learning for continuous detection with a large language model (LLM) layer that validates alerts and reduces false positives.

> **Status:** source code is being prepared for publication (credentials and internal data removed). Firmware, backend, web interface and installation guide will be published here shortly.

## Overview

- **Firmware (ESP32 + LoRaMesher):** mesh nodes send periodic beacons with RSSI, SNR, routing table, battery and uptime. A gateway node forwards the data over Wi-Fi/MQTT.
- **Backend (Python, FastAPI, PostgreSQL):** ingests telemetry, builds 15-minute feature windows and runs the detectors.
- **Detection:** Isolation Forest and LSTM autoencoder running continuously, plus on-demand algorithms (LOF, One-Class SVM, ECOD, PCA, LODA). Alerts are classified into a catalog of seven anomaly types (signal degradation, inferred telemetry loss, battery drop, link instability, suspected interference, link asymmetry, parent flapping).
- **LLM validation:** each alert is reviewed by an LLM (Gemini, Anthropic or local Ollama) that returns a verdict: real, warning, false positive or inconclusive. Human feedback is fed back as examples.
- **Web interface (React):** topology map, per-node details, reports, detection dashboards, LLM audit and data export.

## Publication

If you use this work, please cite:

> FREITAS, J. P.; VIEIRA, F. H. T. *Detecção de Anomalias e Falhas em Redes LoRa Mesh: Uma Abordagem com Isolation Forest e LLM para Redução de Falsos Positivos.* In: Workshop de Testes e Tolerância a Falhas (WTF), 44º Simpósio Brasileiro de Redes de Computadores e Sistemas Distribuídos (SBRC). Praia do Forte, Brazil, 2026.


## License

[MIT](LICENSE)

---

*Em português:* este repositório reúne o sistema de monitoramento e detecção de anomalias em redes mesh LoRa desenvolvido com apoio do CERISE. https://cerise.ufg.br/
