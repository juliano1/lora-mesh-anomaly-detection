"""
REPRODUÇÃO DE UM ESTUDO ANTERIOR — não use para novas avaliações.
Implementação própria com defeitos conhecidos (só a última simulação por nó,
perda após reset, lacunas concatenadas, contexto de calibração da cascata).
Novas avaliações usam o backend desta versão, que não tem esses defeitos
(ai.sequence_builder, ai.feature_engineering, ai.reference_policy).

LSTM autoencoder isolado.

Aprende a reconstrução de sequências temporais de features normais.
Anomalia = erro de reconstrução (MSE) acima de τ_AE calibrado na validação limpa.

Arquitetura:
    Encoder: LSTM(n_feat→hidden) → Dropout → LSTM(hidden→latent)
    Bottleneck: último hidden state → RepeatVector(seq_len)
    Decoder: Dropout → LSTM(latent→hidden) → Dropout → LSTM(hidden→hidden) → Linear(hidden→n_feat)

Regularização:
    - Dropout=0.2
    - weight_decay=1e-5 no Adam
    - ReduceLROnPlateau
    - Early stopping (paciência=5 sobre val loss)

Separação de dados:
    Holdout temporal: últimos 20% do tempo = validação.
    Scaler fit apenas no treino. Threshold p99 calibrado na validação limpa.

Reprodutibilidade:
    Seeds fixos (numpy/torch/random) + hash SHA256 do dataset registrado em meta.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

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


# ─── Reprodutibilidade ────────────────────────────────────────────────────────

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dataset_hash(df: pd.DataFrame) -> str:
    """Hash SHA-256 dos dados de features + timestamps para rastreabilidade."""
    cols = ["node_id", "window_start", "window_end"] + FEATURE_NAMES
    raw = df[cols].sort_values(["node_id", "window_start"]).to_csv(index=False).encode()
    return hashlib.sha256(raw).hexdigest()


# ─── Modelo ───────────────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, latent: int = 32,
                 dropout: float = 0.2):
        super().__init__()
        self.n_features = n_features
        self.hidden     = hidden
        self.latent     = latent
        self.dropout_p  = dropout

        self.enc1 = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop_enc    = nn.Dropout(dropout)
        self.enc2 = nn.LSTM(hidden,     latent, batch_first=True)

        self.drop_latent = nn.Dropout(dropout)

        self.dec1 = nn.LSTM(latent,  hidden, batch_first=True)
        self.drop_dec    = nn.Dropout(dropout)
        self.dec2 = nn.LSTM(hidden,  hidden, batch_first=True)
        self.out  = nn.Linear(hidden, n_features)

    def forward(self, x):
        seq_len = x.shape[1]

        h1, _ = self.enc1(x)
        h1 = self.drop_enc(h1)
        h2, (hn, _) = self.enc2(h1)

        latent = hn[-1].unsqueeze(1).repeat(1, seq_len, 1)
        latent = self.drop_latent(latent)

        d1, _ = self.dec1(latent)
        d1 = self.drop_dec(d1)
        d2, _ = self.dec2(d1)
        return self.out(d2)


# ─── Construção de sequências ─────────────────────────────────────────────────

def build_sequences(df: pd.DataFrame, seq_len: int) -> np.ndarray:
    """
    Para cada nó, cria sequências deslizantes de comprimento seq_len.
    Janelas ordenadas por window_start dentro de cada nó.
    """
    sequences = []
    for node_id, grp in df.groupby("node_id"):
        grp = grp.sort_values("window_start")
        X = grp[FEATURE_NAMES].values.astype(np.float32)
        for i in range(len(X) - seq_len + 1):
            sequences.append(X[i: i + seq_len])
    return np.array(sequences) if sequences else np.empty((0, seq_len, len(FEATURE_NAMES)))


def temporal_split(df: pd.DataFrame, val_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Holdout temporal — últimos `val_frac` do tempo vão para validação.
    Split é global (corta no mesmo instante para todos os nós).
    """
    ts = pd.to_datetime(df["window_start"])
    t_min, t_max = ts.min(), ts.max()
    cut = t_min + (t_max - t_min) * (1 - val_frac)
    train = df[ts < cut].copy()
    val   = df[ts >= cut].copy()
    return train, val


def split_from_column(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Usa coluna pré-computada (ex: 'train'/'val'/'holdout'). Holdout fica reservado
    para avaliação externa — o treino vê apenas train e val."""
    if col not in df.columns:
        raise ValueError(f"Coluna '{col}' não existe no parquet de features.")
    return (df[df[col] == "train"].copy(),
            df[df[col] == "val"].copy())


# ─── Treino ───────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, criterion) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total += loss.item() * len(xb)
            n += len(xb)
    return total / max(n, 1)


def train(features_path: Path, out_dir: Path,
          seq_len: int, hidden: int, latent: int, dropout: float,
          epochs: int, batch_size: int, lr: float, weight_decay: float,
          patience: int, val_frac: float, threshold_pct: float,
          seed: int, split_column: str | None = None):

    set_seeds(seed)

    df = pd.read_parquet(features_path)
    ds_hash = dataset_hash(df)
    print(f"Dataset: {len(df):,} vetores  |  nós: {sorted(df['node_id'].unique())}")
    print(f"SHA-256: {ds_hash[:16]}...")

    # ── Split temporal treino/validação ───────────────
    if split_column:
        df_train, df_val = split_from_column(df, split_column)
        print(f"Split via coluna '{split_column}' (holdout ignorado pelo treino).")
    else:
        df_train, df_val = temporal_split(df, val_frac)
    split_desc = (f"split da coluna '{split_column}'" if split_column
                  else f"holdout temporal {int(val_frac*100)}%")
    print(f"Treino: {len(df_train):,}  |  Validação: {len(df_val):,}  ({split_desc})")
    print(f"  treino:    {df_train['window_start'].min()} → {df_train['window_start'].max()}")
    print(f"  validação: {df_val['window_start'].min()} → {df_val['window_start'].max()}")

    # ── Normalização: scaler fit apenas no treino ─────────────────────
    scaler = StandardScaler()
    scaler.fit(df_train[FEATURE_NAMES].values)

    df_train_sc = df_train.copy()
    df_val_sc   = df_val.copy()
    df_train_sc[FEATURE_NAMES] = scaler.transform(df_train[FEATURE_NAMES].values)
    df_val_sc[FEATURE_NAMES]   = scaler.transform(df_val[FEATURE_NAMES].values)

    seqs_train = build_sequences(df_train_sc, seq_len)
    seqs_val   = build_sequences(df_val_sc,   seq_len)
    print(f"Sequências treino: {len(seqs_train):,}  |  validação: {len(seqs_val):,}  "
          f"(seq={seq_len}, feat={len(FEATURE_NAMES)})")

    if len(seqs_train) < 100:
        raise RuntimeError(f"Sequências de treino insuficientes ({len(seqs_train)}).")

    X_train_t = torch.tensor(seqs_train, dtype=torch.float32)
    X_val_t   = torch.tensor(seqs_val,   dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, X_train_t),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   X_val_t),
                              batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model     = LSTMAutoencoder(len(FEATURE_NAMES), hidden, latent, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
    )
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTreinando LSTM-AE  (epochs={epochs}, batch={batch_size}, lr={lr}, "
          f"dropout={dropout}, wd={weight_decay}, patience={patience})")
    print(f"  Parâmetros: {n_params:,}")

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            n     += len(xb)
        train_loss = total / n

        val_loss = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(cur_lr)

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val   = val_loss
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 5 == 0 or epoch == 1 or improved:
            flag = " *" if improved else ""
            print(f"  Epoch {epoch:3d}/{epochs}  train={train_loss:.6f}  "
                  f"val={val_loss:.6f}  lr={cur_lr:.2e}{flag}")

        if bad_epochs >= patience:
            print(f"  Early stopping na época {epoch} (paciência={patience}).")
            break

    # Restaura melhor estado
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Threshold τ_AE sobre validação limpa ──────────
    model.eval()
    with torch.no_grad():
        recon_val = model(X_val_t.to(device)).cpu().numpy()
    val_errors = np.mean((seqs_val - recon_val) ** 2, axis=(1, 2))
    threshold = float(np.percentile(val_errors, threshold_pct))

    # Também calcula erros no treino para diagnóstico
    with torch.no_grad():
        recon_train = model(X_train_t.to(device)).cpu().numpy()
    train_errors = np.mean((seqs_train - recon_train) ** 2, axis=(1, 2))

    print(f"\nTreino concluído (best val loss={best_val:.6f})")
    print(f"  Erro reconstr. treino: mean={train_errors.mean():.6f}  std={train_errors.std():.6f}")
    print(f"  Erro reconstr. valid.: mean={val_errors.mean():.6f}  std={val_errors.std():.6f}")
    print(f"  Threshold (p{threshold_pct:.0f} em validação): {threshold:.6f}")
    n_val_anom = int((val_errors > threshold).sum())
    print(f"  Amostras de validação acima do threshold: {n_val_anom} "
          f"({n_val_anom/max(len(val_errors),1)*100:.1f}%)")

    # ── Salvar ────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "lstm_model.pt")
    joblib.dump(scaler, out_dir / "lstm_scaler.joblib")

    meta = {
        "model": "LSTMAutoencoder",
        "library": "pytorch",
        "pytorch_version": torch.__version__,
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "seq_len": seq_len,
        "hidden": hidden,
        "latent": latent,
        "dropout": dropout,
        "n_parameters": n_params,
        "epochs_configured": epochs,
        "epochs_run": len(history["train_loss"]),
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "early_stopping_patience": patience,
        "val_frac": val_frac,
        "split_column": split_column,
        "seed": seed,
        "dataset_sha256": ds_hash,
        "n_train_samples": len(df_train),
        "n_val_samples":   len(df_val),
        "n_train_sequences": len(seqs_train),
        "n_val_sequences":   len(seqs_val),
        "best_val_loss": float(best_val),
        "train_recon_mean": float(train_errors.mean()),
        "train_recon_std":  float(train_errors.std()),
        "val_recon_mean": float(val_errors.mean()),
        "val_recon_std":  float(val_errors.std()),
        "threshold_pct": threshold_pct,
        "threshold": threshold,
        "train_loss_history": history["train_loss"],
        "val_loss_history":   history["val_loss"],
        "lr_history":         history["lr"],
        "period_train_start": str(df_train["window_start"].min()),
        "period_train_end":   str(df_train["window_start"].max()),
        "period_val_start":   str(df_val["window_start"].min()),
        "period_val_end":     str(df_val["window_start"].max()),
        "nodes": sorted(df["node_id"].unique().tolist()),
    }
    with open(out_dir / "lstm_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSalvo: {out_dir}/lstm_model.pt")
    print(f"       {out_dir}/lstm_scaler.joblib")
    print(f"       {out_dir}/lstm_meta.json")
    return model, scaler, threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",       default="data/features.parquet")
    ap.add_argument("--out",            default="data/models")
    ap.add_argument("--seq-len",        type=int,   default=8)
    ap.add_argument("--hidden",         type=int,   default=64)
    ap.add_argument("--latent",         type=int,   default=32)
    ap.add_argument("--dropout",        type=float, default=0.2)
    ap.add_argument("--epochs",         type=int,   default=100)
    ap.add_argument("--batch-size",     type=int,   default=64)
    ap.add_argument("--lr",             type=float, default=1e-3)
    ap.add_argument("--weight-decay",   type=float, default=1e-5)
    ap.add_argument("--patience",       type=int,   default=5)
    ap.add_argument("--val-frac",       type=float, default=0.2)
    ap.add_argument("--threshold-pct",  type=float, default=99.0,
                    help="Percentil do erro de reconstrução em validação para τ_AE")
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--split-column",   default=None,
                    help="Nome da coluna do parquet com o split pré-computado "
                         "(train/val/holdout). Quando passada, --val-frac é ignorado.")
    args = ap.parse_args()

    train(
        Path(args.features),
        Path(args.out),
        args.seq_len,
        args.hidden,
        args.latent,
        args.dropout,
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.patience,
        args.val_frac,
        args.threshold_pct,
        args.seed,
        split_column=args.split_column,
    )


if __name__ == "__main__":
    main()
