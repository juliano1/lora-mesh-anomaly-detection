"""
REPRODUÇÃO DE UM ESTUDO ANTERIOR — não use para novas avaliações.
Defeito conhecido: os blocos vão até 90% do índice de cada nó, mas a validação de
train_lstm.py/train_cascade.py é o último trecho de 20% do tempo; logo há anomalias
injetadas no treino e na validação que calibra limiar e τ.

Injeta anomalias sintéticas num parquet de features gerado por extract_features.py,
para obter rótulos conhecidos.

Saída: data/features_labeled.parquet
  Mesmas colunas da entrada + coluna 'anomaly_type' (None = normal).

Tipos de anomalia:
  link_degradation  — packet_loss_rate↑, rssi_mean↓, pdr_mean↓
  rf_interference   — rssi_std↑, snr_mean↓, rssi_snr_divergence↑
  beacon_irregular  — beacon_interval_std↑
  route_flapping    — route_change_rate↑
  battery_failure   — battery_delta↓ (grande negativo)

Estratégia de injeção por nó:
  - BLOCKS_PER_TYPE blocos de BLOCK_SIZE vetores consecutivos por tipo
  - BLOCK_SIZE=20 garante ≥13 sequências LSTM completamente anômalas por bloco
  - Blocos separados por GAP_MIN=8 (= seq_len) vetores entre si
  - Blocos inseridos nos primeiros 90% dos vetores do nó (NÃO evita a validação,
    que começa nos últimos ~20%; ver defeito acima)
"""

import pandas as pd
import numpy as np
from pathlib import Path

SEED = 42
BLOCK_SIZE = 20
BLOCKS_PER_TYPE = 2
SEQ_LEN = 8
GAP_MIN = SEQ_LEN

ANOMALY_TYPES = [
    "link_degradation",
    "rf_interference",
    "beacon_irregular",
    "route_flapping",
    "battery_failure",
]

# (feature, modo, delta_min, delta_max)
# modo 'add': adiciona delta ao valor existente
MODIFICATIONS = {
    "link_degradation": [
        ("packet_loss_rate", "add",  0.55,  0.80),
        ("rssi_mean",        "add", -22.0, -15.0),
        ("pdr_mean",         "add",  -5.5,  -3.0),
    ],
    "rf_interference": [
        ("rssi_std",            "add", 10.0, 16.0),
        ("snr_mean",            "add",  -5.5, -4.0),
        ("rssi_snr_divergence", "add",   6.0, 14.0),
    ],
    "beacon_irregular": [
        ("beacon_interval_std", "add", 90.0, 150.0),
    ],
    "route_flapping": [
        ("route_change_rate", "add", 0.12, 0.38),
    ],
    "battery_failure": [
        ("battery_delta", "add", -38.0, -25.0),
    ],
}

CLAMP = {
    "packet_loss_rate":    (0.0,   1.0),
    "pdr_mean":            (0.0,  10.0),
    "rssi_mean":           (-100.0, None),
    "snr_mean":            (0.0,  None),
    "rssi_std":            (0.0,  None),
    "rssi_snr_divergence": (0.0,  None),
    "beacon_interval_std": (0.0,  None),
    "route_change_rate":   (0.0,  None),
}


def inject_node(node_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    node_df = node_df.sort_values("window_start").reset_index(drop=True)
    n = len(node_df)
    # Não injeta nos últimos 10% dos vetores do nó. Isso não protege a validação dos
    # scripts de treino (últimos 20% do tempo); ver o defeito no cabeçalho.
    safe_end = int(n * 0.90) - BLOCK_SIZE
    if safe_end < BLOCK_SIZE * 2:
        print(f"    Nó {node_df['node_id'].iloc[0]}: poucos vetores ({n}), pulando")
        node_df["anomaly_type"] = None
        return node_df

    node_df["anomaly_type"] = None
    used: list[tuple[int, int]] = []

    for atype in ANOMALY_TYPES:
        placed = 0
        attempts = 0
        while placed < BLOCKS_PER_TYPE and attempts < 500:
            attempts += 1
            start = int(rng.integers(GAP_MIN, safe_end))
            end = start + BLOCK_SIZE

            conflict = any(
                not (end + GAP_MIN <= s or start - GAP_MIN >= e)
                for s, e in used
            )
            if conflict:
                continue

            mods = MODIFICATIONS[atype]
            for feat, mode, lo, hi in mods:
                delta = rng.uniform(lo, hi)
                vals = node_df.loc[start:end - 1, feat].copy()
                if mode == "add":
                    vals = vals + delta
                if feat in CLAMP:
                    lo_c, hi_c = CLAMP[feat]
                    if lo_c is not None:
                        vals = vals.clip(lower=lo_c)
                    if hi_c is not None:
                        vals = vals.clip(upper=hi_c)
                node_df.loc[start:end - 1, feat] = vals

            node_df.loc[start:end - 1, "anomaly_type"] = atype
            used.append((start, end))
            placed += 1

        if placed < BLOCKS_PER_TYPE:
            print(f"    Nó {node_df['node_id'].iloc[0]} / {atype}: "
                  f"só {placed}/{BLOCKS_PER_TYPE} blocos colocados após {attempts} tentativas")

    return node_df


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features.parquet")
    ap.add_argument("--out",      default="data/features_labeled.parquet")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    src = Path(args.features)
    dst = Path(args.out)

    print(f"Carregando {src}...")
    df = pd.read_parquet(src)
    print(f"  {len(df):,} vetores, nós: {sorted(df['node_id'].unique())}")

    parts = []
    for node_id, grp in df.groupby("node_id"):
        print(f"\n  Nó {node_id} ({len(grp)} vetores):")
        injected = inject_node(grp.copy(), rng)
        n_inj = injected["anomaly_type"].notna().sum()
        print(f"    {n_inj} vetores anômalos injetados "
              f"({100 * n_inj / len(injected):.1f}%)")
        parts.append(injected)

    out = pd.concat(parts).sort_values(["node_id", "window_start"]).reset_index(drop=True)
    n_total = len(out)
    n_anom = out["anomaly_type"].notna().sum()

    print(f"\nResumo global:")
    print(f"  Total: {n_total:,} vetores")
    print(f"  Anômalos: {n_anom:,} ({100 * n_anom / n_total:.1f}%)")
    print(f"  Por tipo:")
    print(out["anomaly_type"].value_counts().to_string())

    out.to_parquet(dst, index=False)
    print(f"\nSalvo: {dst}")


if __name__ == "__main__":
    main()
