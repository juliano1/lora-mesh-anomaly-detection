"""
REPRODUÇÃO DE UM ESTUDO ANTERIOR — não use para novas avaliações.
Implementação própria com defeitos conhecidos (só a última simulação por nó,
perda após reset, lacunas concatenadas, contexto de calibração da cascata; dado
ausente vira valor fixo plausível, p. ex. pdr_mean=5.0 e as demais features 0.0,
sem marcação). Novas avaliações usam o backend desta versão, que não tem esses
defeitos (ai.sequence_builder, ai.feature_engineering, ai.reference_policy).

Extrai as 11 features do sistema a partir de um SQLite local (cópia do banco do backend).

Replica a lógica de FeatureExtractor (backend/ai/feature_engineering.py) de forma
independente — sem FastAPI, sem SQLModel, sem o backend rodando.

Por padrão, exclui janelas contaminadas por simulação.
Use --include-sim para manter (útil apenas para avaliação com rótulos conhecidos).

Saída: data/features.parquet  (e data/features.csv para inspeção)

Colunas:
    node_id, window_start, window_end,
    rssi_mean, rssi_std, snr_mean, snr_std, pdr_mean,
    battery_delta, packet_loss_rate, rssi_snr_divergence,
    beacon_interval_std, route_change_rate, link_asymmetry

Uso:
    python scripts/extract_features.py [--db backend/lora_mesh.db] [--window 15] [--step 7]
"""

import sqlite3
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "rssi_mean", "rssi_std",
    "snr_mean", "snr_std",
    "pdr_mean",
    "battery_delta",
    "packet_loss_rate",
    "rssi_snr_divergence",
    "beacon_interval_std",
    "route_change_rate",
    "link_asymmetry",
]


def load_node_addresses(con) -> dict[int, str]:
    """node_id → lora_address"""
    cur = con.execute("SELECT node_id, lora_address FROM node WHERE lora_address IS NOT NULL")
    return {row[0]: row[1] for row in cur.fetchall()}


def load_sim_periods(con) -> dict[int, tuple[str, str | None]]:
    """
    Carrega períodos de simulação por nó.
    Retorna: node_id → (sim_started_at, sim_ended_at or None)
    Nós sem sim_started_at não entram no dict.
    """
    cur = con.execute("""
        SELECT node_id, sim_started_at, sim_ended_at
        FROM node
        WHERE sim_started_at IS NOT NULL
    """)
    return {row[0]: (row[1], row[2]) for row in cur.fetchall()}


def window_overlaps_sim(win_start: str, win_end: str,
                        sim_start: str, sim_end: str | None) -> bool:
    """Retorna True se a janela [win_start, win_end] se sobrepõe ao período de simulação."""
    if sim_end is None:
        # Simulação em aberto — contamina qualquer janela após sim_start
        return win_end > sim_start
    return win_start < sim_end and win_end > sim_start


def extract_window(con, node_id: int, lora_address: str,
                   addr_to_id: dict[str, int],
                   win_start: str, win_end: str) -> dict | None:
    """Extrai features para um nó em uma janela temporal."""

    # ── Registros de vizinhos ─────────────────────────────────────────
    nb_rows = con.execute("""
        SELECT neighbor_address, rssi, snr, is_parent, link_quality, timestamp
        FROM neighbor
        WHERE node_id = ? AND timestamp > ? AND timestamp <= ?
        ORDER BY timestamp
    """, (node_id, win_start, win_end)).fetchall()

    if len(nb_rows) < 2:
        return None

    nb_addrs   = [r[0] for r in nb_rows]
    rssi_vals  = [r[1] for r in nb_rows]
    snr_vals   = [r[2] for r in nb_rows]
    is_parent  = [bool(r[3]) for r in nb_rows]
    lq_vals    = [r[4] for r in nb_rows if r[4] is not None]
    nb_ts      = [r[5] for r in nb_rows]

    rssi_mean = float(np.mean(rssi_vals))
    rssi_std  = float(np.std(rssi_vals))
    snr_mean  = float(np.mean(snr_vals))
    snr_std   = float(np.std(snr_vals))
    pdr_mean  = float(np.mean(lq_vals)) if lq_vals else 5.0

    # ── Registros de telemetria ───────────────────────────────────────
    tel_rows = con.execute("""
        SELECT battery_level, sequence, timestamp
        FROM telemetry
        WHERE node_id = ? AND timestamp > ? AND timestamp <= ?
        ORDER BY timestamp
    """, (node_id, win_start, win_end)).fetchall()

    # battery_delta
    bat_vals = [r[0] for r in tel_rows if r[0] is not None and r[0] >= 0]
    battery_delta = float(bat_vals[-1] - bat_vals[0]) if len(bat_vals) >= 2 else 0.0

    # packet_loss_rate (deduplica sequence)
    seq_vals = sorted(set(r[1] for r in tel_rows if r[1] is not None))
    packet_loss_rate = 0.0
    if len(seq_vals) >= 2:
        expected = seq_vals[-1] - seq_vals[0]
        actual   = len(seq_vals) - 1
        if expected > 0 and actual <= expected:
            packet_loss_rate = float(1.0 - actual / expected)

    # ── Divergência temporal RSSI-SNR por enlace ──────────────────────
    link_rssi: defaultdict = defaultdict(list)
    link_snr:  defaultdict = defaultdict(list)
    for addr, r, s in zip(nb_addrs, rssi_vals, snr_vals):
        link_rssi[addr].append(r)
        link_snr[addr].append(s)
    r_stds = [float(np.std(v)) for v in link_rssi.values() if len(v) >= 2]
    s_stds = [float(np.std(v)) for v in link_snr.values()  if len(v) >= 2]
    rssi_snr_div = abs(float(np.mean(s_stds)) - float(np.mean(r_stds))) if (r_stds and s_stds) else 0.0

    # ── Desvio-padrão do intervalo entre beacons ──────────────────────
    beacon_interval_std = 0.0
    if len(tel_rows) >= 3:
        ts_list = [r[2] for r in tel_rows]
        dts = [
            (datetime.fromisoformat(ts_list[i+1]) - datetime.fromisoformat(ts_list[i])).total_seconds()
            for i in range(len(ts_list) - 1)
        ]
        if dts:
            beacon_interval_std = float(np.std(dts))

    # ── Taxa de troca de rota ─────────────────────────────────────────
    parent_seq = [addr for addr, ip in zip(nb_addrs, is_parent) if ip]
    route_change_rate = 0.0
    if len(parent_seq) >= 2:
        changes = sum(1 for i in range(1, len(parent_seq)) if parent_seq[i] != parent_seq[i-1])
        route_change_rate = float(changes) / len(parent_seq)

    # ── Assimetria de enlace ──────────────────────────────────────────
    link_asymmetry = 0.0
    if lora_address:
        unique_nb = set(nb_addrs)
        asym_deltas = []
        for nb_addr in unique_nb:
            nb_node_id = addr_to_id.get(nb_addr)
            if nb_node_id is None:
                continue
            rev = con.execute("""
                SELECT rssi FROM neighbor
                WHERE node_id = ? AND neighbor_address = ?
                  AND timestamp > ? AND timestamp <= ?
            """, (nb_node_id, lora_address, win_start, win_end)).fetchall()
            if not rev:
                continue
            our_rssi  = float(np.mean([r for a, r in zip(nb_addrs, rssi_vals) if a == nb_addr]))
            their_rssi = float(np.mean([r[0] for r in rev]))
            asym_deltas.append(abs(our_rssi - their_rssi))
        if asym_deltas:
            link_asymmetry = float(np.mean(asym_deltas))

    return {
        "rssi_mean":           rssi_mean,
        "rssi_std":            rssi_std,
        "snr_mean":            snr_mean,
        "snr_std":             snr_std,
        "pdr_mean":            pdr_mean,
        "battery_delta":       battery_delta,
        "packet_loss_rate":    packet_loss_rate,
        "rssi_snr_divergence": rssi_snr_div,
        "beacon_interval_std": beacon_interval_std,
        "route_change_rate":   route_change_rate,
        "link_asymmetry":      link_asymmetry,
    }


def extract_all(db_path: Path, window_min: int, step_min: int,
                exclude_sim: bool = True,
                start: str | None = None,
                end: str | None = None,
                exclude_nodes: list[int] | None = None) -> pd.DataFrame:
    con = sqlite3.connect(db_path)

    addr_map   = load_node_addresses(con)          # node_id → lora_address
    addr_to_id = {v: k for k, v in addr_map.items()}  # lora_address → node_id

    sim_periods = load_sim_periods(con) if exclude_sim else {}
    if exclude_sim and sim_periods:
        print(f"excluindo janelas sob simulação para nós {list(sim_periods.keys())}")

    # período: usa start/end se fornecidos, senão min/max do banco
    row = con.execute("SELECT MIN(timestamp), MAX(timestamp) FROM neighbor").fetchone()
    t_min = datetime.fromisoformat(start) if start else datetime.fromisoformat(row[0])
    t_max = datetime.fromisoformat(end)   if end   else datetime.fromisoformat(row[1])

    node_ids = [n for n in addr_map.keys() if n not in (exclude_nodes or [])]
    if exclude_nodes:
        print(f"Nós excluídos: {exclude_nodes}")
    print(f"Nós ativos: {node_ids}")
    print(f"Período: {t_min} → {t_max}")
    print(f"Janela: {window_min}min  Step: {step_min}min")

    records = []
    n_skipped_sim = 0
    win_delta  = timedelta(minutes=window_min)
    step_delta = timedelta(minutes=step_min)

    total_windows = int((t_max - t_min - win_delta) / step_delta) + 1
    done = 0

    win_start = t_min
    while win_start + win_delta <= t_max:
        win_end = win_start + win_delta
        ws = win_start.isoformat(sep=" ")
        we = win_end.isoformat(sep=" ")

        for node_id in node_ids:
            # Filtro de simulação
            if node_id in sim_periods:
                sim_start, sim_end = sim_periods[node_id]
                if window_overlaps_sim(ws, we, sim_start, sim_end):
                    n_skipped_sim += 1
                    continue

            lora_addr = addr_map[node_id]
            feats = extract_window(con, node_id, lora_addr, addr_to_id, ws, we)
            if feats:
                rec = {
                    "node_id":      node_id,
                    "window_start": ws,
                    "window_end":   we,
                }
                rec.update(feats)
                records.append(rec)

        done += 1
        if done % 50 == 0:
            print(f"  {done}/{total_windows} janelas...", end="\r", flush=True)

        win_start += step_delta

    con.close()

    df = pd.DataFrame(records)
    print(f"\nTotal de vetores extraídos: {len(df):,}")
    if n_skipped_sim:
        print(f"Janelas puladas por simulação: {n_skipped_sim:,}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",     default="backend/lora_mesh.db")
    ap.add_argument("--out",    default="data/features.parquet")
    ap.add_argument("--window", type=int, default=15, help="Janela em minutos")
    ap.add_argument("--step",   type=int, default=7,  help="Passo em minutos")
    ap.add_argument("--include-sim", action="store_true",
                    help="Inclui janelas sob simulação. Padrão: exclui.")
    ap.add_argument("--start", default=None, help="Data início (ISO, ex: 2026-05-08)")
    ap.add_argument("--end",   default=None, help="Data fim inclusiva (ISO, ex: 2026-05-13)")
    ap.add_argument("--exclude-nodes", default=None,
                    help="Lista de node_ids a excluir, separados por vírgula (ex: 6)")
    args = ap.parse_args()

    db_path  = Path(args.db)
    out_path = Path(args.out)

    if not db_path.exists():
        print(f"Banco não encontrado: {db_path}", file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    exclude_nodes = [int(x) for x in args.exclude_nodes.split(",")] if args.exclude_nodes else None
    df = extract_all(db_path, args.window, args.step,
                     exclude_sim=not args.include_sim,
                     start=args.start, end=args.end,
                     exclude_nodes=exclude_nodes)

    df.to_parquet(out_path, index=False)
    df.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"Salvo: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"       {out_path.with_suffix('.csv')}")
    print()
    print(df[FEATURE_NAMES].describe().round(3).to_string())


if __name__ == "__main__":
    main()
