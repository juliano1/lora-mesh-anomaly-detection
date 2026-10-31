#!/usr/bin/env python3
"""Fluxo ponta a ponta com dados sintéticos: MQTT → backend → banco → detecção ao vivo
(LSTM-AE e Isolation Forest em shadow) → exportação → detector CSV.

Usa só valores fictícios e componentes descartáveis: um Mosquitto em contêiner Docker
(imagem fixada em ``scripts/docker-compose.yml``), um banco SQLite e o backend real
(``backend/start.py``), todos dentro da pasta de trabalho, ligados só em 127.0.0.1.

Rede sintética: gateway (0x0001) e nós N2 (0x0002) e N3 (0x0003), vizinhos entre si.
Beacons no formato do firmware, a cada 20 s, com a bateria enviada quando muda 2 p.p.
ou a cada 5 min.

Fases:
 1. Histórico reconstruído. ``--horas`` de beacons publicados pelo broker em ritmo
    acelerado (26 h levam cerca de 1,5 min) e gravados pela ingestão normal do backend.
    O firmware não envia horário (o backend carimba na chegada), então, com o backend
    parado, os carimbos são reescritos no banco para o passado com o ritmo de 20 s:
    telemetria pelo contador de sequência; vizinhos pela telemetria da mesma mensagem.
    O histórico não foi vivido em 26 h: só o ritmo e a ordem são os de um nó real. As
    linhas de detecção criadas durante o envio acelerado são descartadas e contadas.
 2. Ao vivo. Backend religado com LSTM-AE (primário) e IF (secundário) em shadow;
    beacons em tempo real. Normal até o pipeline treinar + ``--normal-min``; depois a
    recepção do N3 cai (RSSI ≈ −106 dBm, SNR ≈ −6 dB) por ``--anomalia-min``. O N2
    segue normal e serve de controle.
 3. Exportação ``full`` e detector CSV, com corte de treino no fim do histórico.

Leitura dos resultados:
- os ciclos da fase anômala (um por minuto) são observações repetidas de UMA única
  degradação, não eventos independentes; a taxa por ciclo mede a persistência da
  detecção nessa ocorrência, não a sensibilidade sobre uma população de eventos;
- detecção (o N3 degradado foi sinalizado?) e localização (o alarme ficou no nó de
  origem?) são avaliadas em separado. A degradação do N3 torna o enlace N2↔N3
  assimétrico, e o N2 pode receber A6; esse alarme deve citar o enlace com o N3 e não
  atribuir a origem a nenhum dos dois nós;
- todo o banco é sintético, mas fica gravado como ``data_origin = real``. A política de
  referência do backend e o detector CSV recusam treinar com linhas sintéticas, e o
  roteiro precisa exercitar o treino. A ingestão só marca um beacon como sintético pelo
  campo ``sim``, que é das simulações do firmware. Por isso os alarmes registram 0 linhas
  sintéticas no contexto. O pacote exportado é identificado por ``--source-id
  e2e-sintetico``, e ``resultado.json`` registra a condição.

Requisitos: Docker, as dependências de ``backend/requirements.txt`` e
``backend/requirements-lstm.txt`` no interpretador de ``--python``, e as portas
``--porta-mqtt`` e ``--porta-api`` livres. Duração: histórico + treino (≈ 2 min) +
fases ao vivo (padrão ≈ 55 min).

Uso:
    python scripts/e2e_sintetico.py --trabalho /tmp/e2e [--python .venv/bin/python]
        [--horas 26] [--normal-min 20] [--anomalia-min 30]
        [--porta-mqtt 18830] [--porta-api 18000]

A pasta de trabalho precisa estar vazia ou não existir. Ao final ficam nela
``resultado.json`` (checagens e fatos), o banco, os logs do backend, o pacote exportado
e a saída do detector CSV. Código de saída 0 se todas as checagens passam, 1 se
alguma falha e 2 se a execução não pode começar.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import random
import re
import secrets
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import warnings
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

import paho.mqtt.client as mqtt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BEACON = 20                                  # IDENTITY_BROADCAST_INTERVAL do firmware (s)
BATTERY_HEARTBEAT = 300                      # BATTERY_SEND_INTERVAL (s)
BATTERY_CHANGE = 2                           # BATTERY_CHANGE_THRESHOLD (p.p.)
Q0 = 1000                                    # primeiro contador de sequência
# (nome, endereço, outro nó, deslocamento do beacon em s)
NODES = (("N2", "0002", "0003", 0), ("N3", "0003", "0002", 10))
# Credenciais do broker e do backend descartáveis: aleatórias a cada execução.
GATEWAY_USER = "gateway-rede1"
GATEWAY_PASS = secrets.token_urlsafe(18)
BACKEND_USER = "backend"
BACKEND_PASS = secrets.token_urlsafe(18)
ADMIN_PASS = secrets.token_urlsafe(18)
COMPONENTS = {"primary": "LSTM", "secondary": "IF"}
DB_TS = "%Y-%m-%d %H:%M:%S.%f"

results: dict = {}
facts: dict = {}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def check(name, ok, detail=""):
    results[name] = bool(ok)
    log(("ok    " if ok else "FALHA ") + name + (f" — {detail}" if detail else ""))


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ─── Infraestrutura descartável ──────────────────────────────────────────────

def port_free(port):
    with socket.socket() as s:
        # Como o uvicorn e o Docker: conexões em TIME_WAIT não ocupam a porta.
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def broker_image():
    """Imagem fixada por digest no compose do projeto (fonte única)."""
    text = (ROOT / "scripts" / "docker-compose.yml").read_text(encoding="utf-8")
    m = re.search(r"^\s*image:\s*(\S+@sha256:[0-9a-f]{64})", text, re.MULTILINE)
    if not m:
        raise RuntimeError("imagem do Mosquitto não encontrada em scripts/docker-compose.yml")
    return m.group(1)


class Broker:
    """Mosquitto em contêiner, só em 127.0.0.1, com usuários fictícios e a ACL do projeto."""

    def __init__(self, work: Path, port: int):
        self.dir, self.port, self.cid = work / "broker", port, None
        self.name = f"lora_e2e_broker_{port}"

    def start(self):
        image = broker_image()
        self.dir.mkdir()
        src = ROOT / "scripts" / "mosquitto" / "config"
        shutil.copy(src / "mosquitto.conf", self.dir / "mosquitto.conf")
        shutil.copy(src / "acl.example", self.dir / "acl")
        # O broker roda como o usuário mosquitto: o arquivo de senhas precisa ser dele e
        # não pode ficar legível por outros.
        pw = "/mosquitto/config/password_file"
        subprocess.run(["docker", "run", "--rm", "-v", f"{self.dir}:/mosquitto/config", image, "sh", "-c",
                        f'mosquitto_passwd -b -c {pw} "$1" "$2" && mosquitto_passwd -b {pw} "$3" "$4" '
                        f"&& chown mosquitto {pw} && chmod 0600 {pw}",
                        "sh", GATEWAY_USER, GATEWAY_PASS, BACKEND_USER, BACKEND_PASS],
                       check=True, capture_output=True)
        out = subprocess.run(["docker", "run", "-d", "--name", self.name,
                              "-p", f"127.0.0.1:{self.port}:1883",
                              "-v", f"{self.dir}:/mosquitto/config:ro", image],
                             check=True, capture_output=True, text=True)
        self.cid = out.stdout.strip()
        # A porta publicada abre antes do Mosquitto; só um CONNACK aceito confirma.
        for _ in range(20):
            accepted = threading.Event()
            probe = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            probe.username_pw_set(BACKEND_USER, BACKEND_PASS)
            probe.on_connect = lambda c, u, f, rc, p, ev=accepted: rc == 0 and ev.set()
            try:
                probe.connect("127.0.0.1", self.port)
                probe.loop_start()
                ok = accepted.wait(1)
                probe.loop_stop()
                probe.disconnect()
                if ok:
                    return
            except OSError:
                time.sleep(0.5)
        logs = subprocess.run(["docker", "logs", self.cid], capture_output=True, text=True)
        raise RuntimeError(f"broker não aceitou conexão: {(logs.stdout + logs.stderr)[-300:]}")

    def stop(self):
        if self.cid:                         # só o contêiner criado por este roteiro
            subprocess.run(["docker", "rm", "-f", self.cid], capture_output=True)
            self.cid = None


class Backend:
    def __init__(self, python, work: Path, env: dict, api: str):
        self.python, self.work, self.env, self.api = python, work, env, api
        self.proc = self.logf = None

    def start(self, tag):
        self.logf = open(self.work / f"backend_{tag}.log", "w")
        self.proc = subprocess.Popen([self.python, "start.py"], cwd=ROOT / "backend", env=self.env,
                                     stdout=self.logf, stderr=subprocess.STDOUT)
        for _ in range(240):
            if self.proc.poll() is not None:
                raise RuntimeError(f"backend terminou (código {self.proc.returncode}); veja backend_{tag}.log")
            try:
                if http("GET", f"{self.api}/health", timeout=1)[0] == 200:
                    return
            except OSError:
                pass
            time.sleep(0.5)
        raise RuntimeError("backend não respondeu em /health")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(10)
        if self.logf:
            self.logf.close()
        self.proc = self.logf = None


def http(method, url, body=None, headers=None, timeout=10):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def login(api):
    status, text = http("POST", f"{api}/api/auth/login", {"username": "admin100", "password": ADMIN_PASS})
    if status != 200:
        raise RuntimeError(f"login falhou: {status} {text[:200]}")
    return {"Authorization": f"Bearer {json.loads(text)['access_token']}"}


# ─── Beacons no formato do firmware ──────────────────────────────────────────

def battery_at(k):
    """Indicador de bateria contínuo entre histórico e fase ao vivo (dreno de 0,9 p.p./h)."""
    return 95 - int(k * 0.9 * BEACON / 3600)


class BatteryGate:
    """Envio condicional do firmware: muda 2 p.p. ou passa 5 min desde o último envio."""

    def __init__(self):
        self.last = {}

    def include(self, node, k, pct):
        sent = self.last.get(node)
        if sent is None or abs(pct - sent[1]) >= BATTERY_CHANGE or (k - sent[0]) * BEACON >= BATTERY_HEARTBEAT:
            self.last[node] = (k, pct)
            return True
        return False


def payload(name, addr, other, q, rnd, battery=None, degraded=False):
    base, snr, lq = (-106, -6, 2) if degraded else (-75, 8, 8)
    msg = {"i": name, "a": addr, "u": 100 + q * BEACON // 60, "q": q,
           "n": [{"a": "0001", "r": round(rnd.gauss(base, 1.5)), "s": round(rnd.gauss(snr, 0.8), 1),
                  "p": True, "h": 1, "l": int(max(0, min(10, round(rnd.gauss(lq + 0.5, 0.5)))))},
                 {"a": other, "r": round(rnd.gauss(base - 4, 1.5)), "s": round(rnd.gauss(snr - 1, 0.8), 1),
                  "p": False, "h": 1, "l": int(max(0, min(10, round(rnd.gauss(lq, 0.5)))))}],
           "gw_rssi": round(rnd.gauss(-75, 1.5)), "gw_snr": round(rnd.gauss(8, 0.8), 1)}
    if battery is not None:
        msg["b"] = battery
    return msg


def gateway_client(port):
    gw = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    gw.username_pw_set(GATEWAY_USER, GATEWAY_PASS)
    gw.connect("127.0.0.1", port)
    gw.loop_start()
    time.sleep(1)
    return gw


def count_rows(db, table="telemetry"):
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10) as c:
        return c.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


# ─── Fases ───────────────────────────────────────────────────────────────────

def phase_history(args, backend, db, rnd, gate):
    n_hist = int(args.horas * 3600 / BEACON)
    lost = {name: {k for k in range(n_hist) if rnd.random() < 0.03} for name, *_ in NODES}
    backend.start("fase1")
    try:
        H = login(backend.api)
        status, text = http("POST", f"{backend.api}/api/networks", {
            "name": "rede-sintetica",
            "gateways": [{"name": "GW", "lora_address": "0x0001", "device_model": "Heltec WiFi LoRa 32 V2"}],
            "devices": [{"name": "N2", "lora_address": "0x0002"}, {"name": "N3", "lora_address": "0x0003"}]},
            headers=H)
        check("rede_criada", status == 201, text[:120])
        status, text = http("PATCH", f"{backend.api}/api/settings/ml", {"online": {
            "primary": "lstm_ae", "secondary": "isolation_forest", "combo": "shadow"}}, headers=H)
        check("selecao_lstm_if_shadow", status == 200, text[:200])
        gw = gateway_client(args.porta_mqtt)
        published, t0 = 0, time.time()
        for k in range(n_hist):
            for name, addr, other, _ in NODES:
                if k in lost[name]:
                    continue
                pct = battery_at(k)
                bat = pct if gate.include(name, k, pct) else None
                gw.publish("lab/1/uplink", json.dumps(payload(name, addr, other, Q0 + k, rnd, battery=bat)), qos=1)
                published += 1
            if k == 5:                       # inválidos: não podem gravar nada
                gw.publish("lab/1/uplink", '{"i": "N2", "a": "0002", "gw_snr": NaN}', qos=1)
                gw.publish("lab/1/uplink", "[1, 2, 3]", qos=1)
            if k % 100 == 99:                # não deixa a fila do broker crescer
                while published - count_rows(db) > 300:
                    time.sleep(0.2)
        for _ in range(600):
            if count_rows(db) >= published:
                break
            time.sleep(0.5)
        gw.loop_stop()
        gw.disconnect()
        stored = count_rows(db)
        facts["historico_envio_acelerado_s"] = round(time.time() - t0, 1)
        check("historico_gravado_por_mqtt", stored == published,
              f"{stored} telemetrias de {published} mensagens válidas ({n_hist} beacons/nó; "
              f"{args.horas} h reconstruídas a partir de {facts['historico_envio_acelerado_s']} s de envio)")
    finally:
        backend.stop()
    return n_hist


def rewrite_history(db, n_hist):
    """Backend parado: reescreve os carimbos para o passado no ritmo de 20 s."""
    restart = utcnow() + timedelta(seconds=15)
    h0 = restart - timedelta(seconds=n_hist * BEACON)
    with sqlite3.connect(db) as c:
        ids = {r[0]: r[1] for r in c.execute("SELECT name, node_id FROM node")}
        for name, _addr, _other, offset in NODES:
            nid = ids[name]
            # A ingestão grava os vizinhos e depois a telemetria da mesma mensagem; os ids
            # de tabelas diferentes não se comparam, então o pareamento é pelo carimbo.
            tel = c.execute("SELECT id, sequence, timestamp FROM telemetry WHERE node_id=? ORDER BY timestamp",
                            (nid,)).fetchall()
            nbs = c.execute("SELECT id, timestamp FROM neighbor WHERE node_id=? ORDER BY timestamp",
                            (nid,)).fetchall()
            target = [(ts, h0 + timedelta(seconds=(q - Q0) * BEACON + offset), tid) for tid, q, ts in tel]
            stamps = [t[0] for t in target]
            upd = []
            for nb_id, nb_ts in nbs:
                j = bisect.bisect_left(stamps, nb_ts)
                if j == len(stamps):
                    raise RuntimeError(f"vizinho {nb_id} sem telemetria posterior")
                upd.append((target[j][1].strftime(DB_TS), nb_id))
            c.executemany("UPDATE neighbor SET timestamp=? WHERE id=?", upd)
            c.executemany("UPDATE telemetry SET timestamp=? WHERE id=?",
                          [(t.strftime(DB_TS), tid) for _, t, tid in target])
            c.execute("UPDATE node SET last_seen=? WHERE node_id=?", (restart.strftime(DB_TS), nid))
        discarded = {t: c.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("online_inference", "anomaly")}
        for t in discarded:
            c.execute(f"DELETE FROM {t}")
        span = c.execute("SELECT min(timestamp), max(timestamp) FROM telemetry").fetchone()
    facts["historico_reconstruido"] = {"inicio": span[0], "fim": span[1],
                                       "linhas_do_envio_acelerado_descartadas": discarded}
    log(f"histórico reescrito: {span}; descartadas (carimbo comprimido): {discarded}")
    return ids, h0, restart


def phase_live(args, backend, rnd, gate, n_hist):
    stop = threading.Event()
    state = {"degraded": False, "q": Q0 + n_hist, "error": None}

    def publish():
        try:
            gw = gateway_client(args.porta_mqtt)
            nxt = time.time()
            while not stop.is_set():
                q = state["q"]
                for name, addr, other, _ in NODES:
                    pct = battery_at(q - Q0)
                    bat = pct if gate.include(name, q - Q0, pct) else None
                    gw.publish("lab/1/uplink", json.dumps(payload(
                        name, addr, other, q, rnd, battery=bat, degraded=name == "N3" and state["degraded"])), qos=1)
                state["q"] += 1
                nxt += BEACON
                stop.wait(max(0, nxt - time.time()))
            gw.loop_stop()
            gw.disconnect()
        except Exception as e:  # noqa: BLE001 — registrado e verificado pela thread principal
            state["error"] = repr(e)

    backend.start("fase2")
    publisher = threading.Thread(target=publish, daemon=True)
    live_start = utcnow()
    publisher.start()
    ready_at = onset = live_end = None
    try:
        H = login(backend.api)
        st = {}
        for _ in range(180):                 # até 30 min para treinar
            st = json.loads(http("GET", f"{backend.api}/api/ml/online/status", headers=H)[1])
            nets = st.get("networks") or {}
            pipe = (nets.get("1") or nets.get(1) or {}).get("published") or {}
            if (pipe.get("primary") or {}).get("ready") and (pipe.get("secondary") or {}).get("ready"):
                ready_at = utcnow()
                facts["pipeline_treinado"] = {k: pipe.get(k) for k in ("primary_id", "secondary_id", "combo",
                                                                       "tau", "tau_status")}
                facts["treino"] = {"lstm": pipe["primary"].get("training_summary"),
                                   "if": pipe["secondary"].get("training_summary")}
                break
            if state["error"]:
                break
            time.sleep(10)
        trained = facts.get("pipeline_treinado") or {}
        check("pipeline_lstm_if_treinado", ready_at is not None and trained.get("primary_id") == "lstm_ae"
              and trained.get("secondary_id") == "isolation_forest", json.dumps(trained or st)[:400])
        if ready_at is None:
            return None
        lstm = (facts["treino"] or {}).get("lstm") or {}
        check("lstm_limiar_do_bloco_de_calibracao", lstm.get("threshold_source") == "calibration"
              and lstm.get("n_calib", 0) > 0, json.dumps({k: lstm.get(k) for k in (
                  "n_train", "n_val", "n_calib", "cutoff", "calib_cutoff", "threshold_source")}))
        log(f"treinado em {(ready_at - live_start).total_seconds():.0f} s; fase normal de {args.normal_min} min")
        time.sleep(args.normal_min * 60)
        state["degraded"], onset = True, utcnow()
        log(f"N3 degradado a partir de {onset}; {args.anomalia_min} min")
        time.sleep(args.anomalia_min * 60)
        live_end = utcnow()
        time.sleep(70)                       # mais um ciclo de detecção
    finally:
        stop.set()
        publisher.join(30)
        backend.stop()
    check("publicador_sem_erro", state["error"] is None, state["error"] or "")
    return ready_at, onset, live_end


# ─── Avaliação: detecção e localização em separado ───────────────────────────

def _ts(value):
    return datetime.fromisoformat(str(value).replace(" ", "T").replace("Z", ""))


def evaluate_live(db, ids, ready_at, onset, live_end):
    names = {v: k for k, v in ids.items()}
    with sqlite3.connect(db) as c:
        c.row_factory = sqlite3.Row
        rows = [dict(r) for r in c.execute(
            "SELECT reference_time, node_id, primary_status, primary_is_anomaly, "
            "secondary_status, secondary_is_anomaly FROM online_inference ORDER BY reference_time")]
        anomalies = [dict(r) for r in c.execute(
            "SELECT node_id, anomaly_type, detector_algorithm, neighbor_address, evidence FROM anomaly")]

    def summary(sel):
        out = {}
        for comp, label in COMPONENTS.items():
            st = [r[f"{comp}_status"] for r in sel]
            an = [r[f"{comp}_is_anomaly"] for r in sel if r[f"{comp}_status"] == "valid"]
            out[label] = {"ciclos": len(sel), "validos": len(an), "anomalia": sum(bool(a) for a in an),
                          "indisponivel": {s: st.count(s) for s in set(st) if s != "valid"}}
        return out

    phases = {
        "normal": lambda t: ready_at + timedelta(minutes=2) <= t < onset,
        "anomalia_0_15min": lambda t: onset <= t < onset + timedelta(minutes=15),
        "anomalia_15min_fim": lambda t: onset + timedelta(minutes=15) <= t <= live_end,
    }
    table = {f"{n}:{ph}": summary([r for r in rows if r["node_id"] == ids[n] and pred(_ts(r["reference_time"]))])
             for ph, pred in phases.items() for n in ("N2", "N3")}
    facts["decisoes_por_ciclo"] = table
    for k, v in table.items():
        log(f"{k:24s} {v}")

    norm = [table[f"{n}:normal"] for n in ("N2", "N3")]
    check("fase_normal_decidida", all(s[m]["validos"] >= 0.9 * s[m]["ciclos"] > 0 for s in norm for m in ("IF", "LSTM")),
          "IF e LSTM com decisão válida em ≥ 90% dos ciclos")
    fp = {f"{n}:{m}": round(table[f"{n}:normal"][m]["anomalia"] / max(table[f"{n}:normal"][m]["validos"], 1), 3)
          for n in ("N2", "N3") for m in ("IF", "LSTM")}
    facts["alarmes_fase_normal"] = fp
    check("fase_normal_sem_excesso_de_alarmes", all(v <= 0.10 for v in fp.values()), json.dumps(fp))

    # Detecção: o N3 degradado foi sinalizado? Uma ocorrência, observada a cada ciclo.
    est = table["N3:anomalia_15min_fim"]
    rate = {m: round(est[m]["anomalia"] / max(est[m]["validos"], 1), 3) for m in ("IF", "LSTM")}
    facts["deteccao"] = {
        "ocorrencias": 1,
        "nota": "ciclos são observações repetidas de uma única degradação do N3, não eventos independentes",
        "ciclos_observados": {m: est[m]["validos"] for m in ("IF", "LSTM")},
        "fracao_de_ciclos_sinalizados": rate,
    }
    check("deteccao_N3_LSTM", rate["LSTM"] >= 0.9, json.dumps(est["LSTM"]))
    check("deteccao_N3_IF", rate["IF"] >= 0.9, json.dumps(est["IF"]))

    # Localização: o alarme ficou no N3? O N2 só tem o enlace afetado.
    ctrl = table["N2:anomalia_15min_fim"]
    facts["localizacao"] = {
        m: {"ciclos_com_alarme_no_N3": est[m]["anomalia"], "ciclos_com_alarme_no_N2": ctrl[m]["anomalia"],
            "fracao_dos_alarmes_no_no_de_origem": round(
                est[m]["anomalia"] / max(est[m]["anomalia"] + ctrl[m]["anomalia"], 1), 3)}
        for m in ("IF", "LSTM")}
    persisted = {}
    for a in anomalies:
        key = f"{names.get(a['node_id'], a['node_id'])}:{a['detector_algorithm']}:{a['anomaly_type']}"
        persisted[key] = persisted.get(key, 0) + 1
    facts["anomalias_persistidas"] = persisted
    n3 = [a for a in anomalies if a["node_id"] == ids["N3"]]
    check("anomalias_A1_no_N3_por_modelo",
          all(any(a["detector_algorithm"] == f"shadow:{m}" and a["anomaly_type"] == "A1" for a in n3)
              for m in ("lstm", "if")), json.dumps(persisted))
    a6_n2 = [a for a in anomalies if a["node_id"] == ids["N2"] and a["anomaly_type"] == "A6"]
    bad = []
    for a in a6_n2:
        ev = json.loads(a["evidence"] or "{}")
        links = {int(link["neighbor"], 16) for link in ev.get("asymmetric_links", [])}
        if int(a["neighbor_address"] or "0", 16) != 0x0003 or ev.get("origin") != "not_determined" or 0x0003 not in links:
            bad.append(a)
    check("A6_no_vizinho_cita_o_enlace_sem_culpar_o_no", not bad,
          f"{len(a6_n2)} A6 no N2; {len(bad)} sem o enlace N2↔N3 ou com origem atribuída")


def export_and_detect(args, work, db, ids, h0, restart, onset, live_end):
    names = {v: k for k, v in ids.items()}
    grid_start = (h0 + timedelta(minutes=1)).replace(second=0, microsecond=0)
    grid_end = (live_end + timedelta(minutes=1)).replace(second=0, microsecond=0)
    while utcnow() < grid_end + timedelta(seconds=1):
        time.sleep(1)

    def iso(t):
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")

    zip_path = work / "e2e.zip"
    exp = subprocess.run([args.python, "scripts/export_dataset.py", "--sqlite", str(db), "--network-id", "1",
                          "--start", iso(grid_start), "--end", iso(grid_end), "--kind", "full",
                          "--window-seconds", "900", "--step-seconds", "300",
                          "--source-id", "e2e-sintetico", "--out", str(zip_path)],
                         cwd=ROOT, capture_output=True, text=True)
    check("exportacao_full", exp.returncode == 0, (exp.stderr or exp.stdout)[-300:])
    if exp.returncode != 0:
        return
    pkg = work / "pacote"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(pkg)
    folder = next(p for p in pkg.iterdir() if p.is_dir())
    with (folder / "features.csv").open(encoding="utf-8") as fh:
        feats = list(csv.DictReader(fh))
    valid = [r for r in feats if r["feature_status"] == "valid"]
    facts["features_exportadas"] = {"linhas": len(feats), "validas": len(valid)}
    check("features_exportadas", len(valid) > 200 and all(math.isfinite(float(r["rssi_mean"])) for r in valid),
          json.dumps(facts["features_exportadas"]))

    out = work / "resultados_csv"
    det = subprocess.run([args.python, "importacao_arquivos_csv/codigos/detector_001_isolation_forest_sklearn.py",
                          "--entrada", str(folder), "--saida", str(out), "--corte-treino", iso(restart)],
                         cwd=ROOT, capture_output=True, text=True)
    runs = list(out.rglob("execucao__*.json"))
    run_state = json.loads(runs[0].read_text(encoding="utf-8"))["estado"] if runs else None
    check("detector_csv_concluido", det.returncode == 0 and run_state == "concluida",
          f"código {det.returncode}, estado {run_state}; {det.stderr[-200:]}")
    flags = []
    for p in out.rglob("*.csv"):
        with p.open(encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            if rd.fieldnames and "is_outlier" in rd.fieldnames:
                flags = [r for r in rd if r["is_outlier"] in ("0", "1")]
                break
    if not flags:
        check("detector_csv_sinaliza_N3_degradado", False, "sem CSV com is_outlier")
        return
    end_key = next(k for k in flags[0] if k in ("window_end", "fim_janela", "window_end_utc"))
    node_key = next(k for k in flags[0] if k in ("node_id", "no"))
    groups = {}
    for r in flags:
        t = _ts(r[end_key])
        phase = ("anomalia_estabelecida" if t >= onset + timedelta(minutes=15)
                 else "normal" if t <= onset else "transicao")
        g = f"{names.get(int(r[node_key]), r[node_key])}:{phase}"
        a, b = groups.get(g, (0, 0))
        groups[g] = (a + (r["is_outlier"] == "1"), b + 1)
    facts["detector_csv_sinalizadas"] = {g: f"{a}/{b}" for g, (a, b) in sorted(groups.items())}
    a, b = groups.get("N3:anomalia_estabelecida", (0, 0))
    check("detector_csv_sinaliza_N3_degradado", b > 0 and a == b, json.dumps(facts["detector_csv_sinalizadas"]))


# ─── Execução ────────────────────────────────────────────────────────────────

def _parser():
    p = argparse.ArgumentParser(description="Fluxo ponta a ponta com dados sintéticos e componentes descartáveis.")
    p.add_argument("--trabalho", type=Path, required=True, help="pasta de trabalho (vazia ou inexistente)")
    p.add_argument("--python", default=sys.executable, help="interpretador com as dependências do backend")
    p.add_argument("--horas", type=float, default=26, help="horas de histórico reconstruído (padrão 26)")
    p.add_argument("--normal-min", type=float, default=20, help="fase normal ao vivo após o treino (min)")
    p.add_argument("--anomalia-min", type=float, default=30, help="duração da degradação do N3 (min)")
    p.add_argument("--porta-mqtt", type=int, default=18830)
    p.add_argument("--porta-api", type=int, default=18000)
    p.add_argument("--semente", type=int, default=11)
    return p


def main(argv=None):
    args = _parser().parse_args(argv)
    work = args.trabalho.resolve()
    if work.exists() and any(work.iterdir()):
        print(f"pasta de trabalho não está vazia: {work}", file=sys.stderr)
        return 2
    busy = [p for p in (args.porta_mqtt, args.porta_api) if not port_free(p)]
    if busy:
        print(f"portas ocupadas: {busy}", file=sys.stderr)
        return 2
    if shutil.which("docker") is None:
        print("Docker não encontrado", file=sys.stderr)
        return 2
    work.mkdir(parents=True, exist_ok=True)
    db = work / "e2e.db"
    api = f"http://127.0.0.1:{args.porta_api}"
    env = {**os.environ,
           "DATABASE_URL": f"sqlite:///{db}",
           "JWT_SECRET_KEY": secrets.token_urlsafe(48),
           "ADMIN100_PASSWORD": ADMIN_PASS, "ADMIN200_PASSWORD": "",
           "MQTT_BROKER": "127.0.0.1", "MQTT_PORT": str(args.porta_mqtt),
           "MQTT_USER": BACKEND_USER, "MQTT_PASS": BACKEND_PASS,
           "API_HOST": "127.0.0.1", "API_PORT": str(args.porta_api), "ML_LSTM_ENABLED": "true",
           "ML_ONLINE_BUNDLE_DIR": str(work / "bundles")}
    facts.update({"inicio": utcnow().isoformat(), "dados": (
        "todos sintéticos, gravados como data_origin=real: a política de referência e o detector "
        "CSV recusam treino com linhas sintéticas"), "parametros": {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "python"}})
    rnd, gate = random.Random(args.semente), BatteryGate()
    broker, backend = Broker(work, args.porta_mqtt), Backend(args.python, work, env, api)
    try:
        broker.start()
        n_hist = phase_history(args, backend, db, rnd, gate)
        ids, h0, restart = rewrite_history(db, n_hist)
        live = phase_live(args, backend, rnd, gate, n_hist)
        if live:
            ready_at, onset, live_end = live
            evaluate_live(db, ids, ready_at, onset, live_end)
            export_and_detect(args, work, db, ids, h0, restart, onset, live_end)
    except Exception as e:  # noqa: BLE001 — a falha entra no resultado
        check("execucao_sem_excecao", False, repr(e))
    finally:
        backend.stop()
        broker.stop()
        facts["fim"] = utcnow().isoformat()
        (work / "resultado.json").write_text(json.dumps({"checks": results, "fatos": facts}, ensure_ascii=False,
                                                        indent=2, default=str), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False))
    return 0 if results and all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
