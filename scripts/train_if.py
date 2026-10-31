"""
REPRODUÇÃO DE UM ESTUDO ANTERIOR — não use para novas avaliações.
Implementação própria com defeitos conhecidos (só a última simulação por nó,
perda após reset, lacunas concatenadas, contexto de calibração da cascata).
Novas avaliações usam o backend desta versão, que não tem esses defeitos
(ai.sequence_builder, ai.feature_engineering, ai.reference_policy).

Isolation Forest isolado, usado como linha de base.

Faz holdout temporal equivalente ao do LSTM para comparabilidade:
scaler e IF são ajustados apenas no treino; as métricas do threshold são também
reportadas sobre a validação limpa.

Reprodutibilidade:
    Seeds fixos (numpy/random/sklearn) + hash SHA256 do dataset registrado em meta.

Uso:
    python scripts/train_if.py [--features data/features.parquet] [--contamination 0.03]
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def dataset_hash(df: pd.DataFrame) -> str:
    cols = ["node_id", "window_start", "window_end"] + FEATURE_NAMES
    raw = df[cols].sort_values(["node_id", "window_start"]).to_csv(index=False).encode()
    return hashlib.sha256(raw).hexdigest()


def temporal_split(df: pd.DataFrame, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.to_datetime(df["window_start"])
    t_min, t_max = ts.min(), ts.max()
    cut = t_min + (t_max - t_min) * (1 - val_frac)
    return df[ts < cut].copy(), df[ts >= cut].copy()


def split_from_column(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Usa coluna pré-computada (ex: 'train'/'val'/'holdout'). Holdout fica reservado
    para avaliação externa — o treino vê apenas train e val."""
    if col not in df.columns:
        raise ValueError(f"Coluna '{col}' não existe no parquet de features.")
    return (df[df[col] == "train"].copy(),
            df[df[col] == "val"].copy())


def train(features_path: Path, out_dir: Path,
          contamination: float, n_estimators: int,
          val_frac: float, seed: int,
          split_column: str | None = None):

    set_seeds(seed)

    df = pd.read_parquet(features_path)
    ds_hash = dataset_hash(df)

    print(f"Dataset: {len(df):,} amostras × {len(FEATURE_NAMES)} features")
    print(f"SHA-256: {ds_hash[:16]}...")
    print(f"Nós presentes: {sorted(df['node_id'].unique())}")
    print(f"Período: {df['window_start'].min()} → {df['window_end'].max()}")

    # ── Split (coluna pré-computada ou holdout temporal por fração) ───
    if split_column:
        df_train, df_val = split_from_column(df, split_column)
        print(f"Split via coluna '{split_column}' (holdout ignorado pelo treino).")
    else:
        df_train, df_val = temporal_split(df, val_frac)
    X_train = df_train[FEATURE_NAMES].values.astype(np.float32)
    X_val   = df_val[FEATURE_NAMES].values.astype(np.float32)
    split_desc = (f"split da coluna '{split_column}'" if split_column
                  else f"holdout temporal {int(val_frac*100)}%")
    print(f"Treino: {len(X_train):,}  |  Validação: {len(X_val):,}  ({split_desc})")

    print(f"Contaminação: {contamination}  |  n_estimators: {n_estimators}  |  seed: {seed}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train_s)

    scores_train = model.score_samples(X_train_s)
    scores_val   = model.score_samples(X_val_s)
    preds_val    = model.predict(X_val_s)

    # Threshold baseado no scoring do treino (IF usa contamination internamente),
    # mas também reportamos o que aconteceria na validação.
    threshold_train = float(np.percentile(scores_train, contamination * 100))
    n_anom_val = int((preds_val == -1).sum())

    print(f"\nTreino concluído.")
    print(f"  Scores treino: mean={scores_train.mean():.4f} std={scores_train.std():.4f} "
          f"min={scores_train.min():.4f} max={scores_train.max():.4f}")
    print(f"  Scores valid.: mean={scores_val.mean():.4f} std={scores_val.std():.4f} "
          f"min={scores_val.min():.4f} max={scores_val.max():.4f}")
    print(f"  Threshold (p{contamination*100:.0f} treino): {threshold_train:.4f}")
    print(f"  Anomalias marcadas na validação: {n_anom_val} "
          f"({n_anom_val/max(len(X_val),1)*100:.1f}%)")

    # ── Salvar ────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": model}, out_dir / "if_model.joblib")

    meta = {
        "model": "IsolationForest",
        "library": "scikit-learn",
        "n_train_samples": int(len(X_train)),
        "n_val_samples":   int(len(X_val)),
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "contamination": contamination,
        "n_estimators": n_estimators,
        "random_state": seed,
        "seed": seed,
        "val_frac": val_frac,
        "split_column": split_column,
        "dataset_sha256": ds_hash,
        "score_train_mean": float(scores_train.mean()),
        "score_train_std":  float(scores_train.std()),
        "score_val_mean":   float(scores_val.mean()),
        "score_val_std":    float(scores_val.std()),
        "threshold": threshold_train,
        "n_anomalies_val": n_anom_val,
        "period_train_start": str(df_train["window_start"].min()),
        "period_train_end":   str(df_train["window_start"].max()),
        "period_val_start":   str(df_val["window_start"].min()),
        "period_val_end":     str(df_val["window_start"].max()),
        "nodes": sorted(df["node_id"].unique().tolist()),
    }
    with open(out_dir / "if_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSalvo: {out_dir}/if_model.joblib")
    print(f"       {out_dir}/if_meta.json")
    return model, scaler, scores_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",      default="data/features.parquet")
    ap.add_argument("--out",           default="data/models")
    ap.add_argument("--contamination", type=float, default=0.03)
    ap.add_argument("--n-estimators",  type=int,   default=200)
    ap.add_argument("--val-frac",      type=float, default=0.2)
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--split-column",  default=None,
                    help="Nome da coluna do parquet com o split pré-computado "
                         "(train/val/holdout). Quando passada, --val-frac é ignorado.")
    args = ap.parse_args()

    train(
        Path(args.features),
        Path(args.out),
        args.contamination,
        args.n_estimators,
        args.val_frac,
        args.seed,
        split_column=args.split_column,
    )


if __name__ == "__main__":
    main()
