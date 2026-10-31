"""
REPRODUÇÃO DE UM ESTUDO ANTERIOR — não use para novas avaliações.
Implementação própria com defeitos conhecidos (só a última simulação por nó,
perda após reset, lacunas concatenadas, contexto de calibração da cascata).
Novas avaliações usam o backend desta versão, que não tem esses defeitos
(ai.sequence_builder, ai.feature_engineering, ai.reference_policy).

Cascata IF → LSTM-AE: o IF filtra as janelas suspeitas e o LSTM-AE confirma ou
descarta cada uma.

Fluxo:
    IF.predict(window) == -1  →  LSTM decide via erro de reconstrução
    IF.predict(window) ==  1  →  normal (sem consultar LSTM)

    LSTM recon_error > τ_AE  →  ANOMALIA confirmada
    LSTM recon_error ≤ τ_AE  →  falso positivo do IF (descartado)

Este script NÃO treina modelos; apenas calibra τ_AE (cascata) sobre o subconjunto
de validação LIMPA marcado como suspeito pelo IF.

Uso:
    python scripts/train_cascade.py [--features data/features.parquet]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch

from train_lstm import (
    LSTMAutoencoder,
    build_sequences,
    temporal_split,
    split_from_column,
    set_seeds,
    FEATURE_NAMES,
)


def load_if(models_dir: Path):
    bundle = joblib.load(models_dir / "if_model.joblib")
    return bundle["model"], bundle["scaler"]


def load_lstm(models_dir: Path):
    with open(models_dir / "lstm_meta.json") as f:
        meta = json.load(f)
    model = LSTMAutoencoder(
        n_features=meta["n_features"],
        hidden=meta["hidden"],
        latent=meta["latent"],
        dropout=meta.get("dropout", 0.2),
    )
    model.load_state_dict(torch.load(models_dir / "lstm_model.pt", map_location="cpu", weights_only=True))
    model.eval()
    scaler = joblib.load(models_dir / "lstm_scaler.joblib")
    return model, scaler, meta


def calibrate(features_path: Path, models_dir: Path,
              val_frac: float, lstm_threshold_pct: float, seed: int,
              split_column: str | None = None):
    """
    Calibra τ_AE da cascata sobre o subconjunto de validação suspeito pelo IF.

    val_frac deve ser idêntico ao usado no treino dos modelos base para garantir
    que a validação seja realmente holdout temporal.
    """
    set_seeds(seed)

    df = pd.read_parquet(features_path)
    if split_column:
        df_train, df_val = split_from_column(df, split_column)
        print(f"Split via coluna '{split_column}' (holdout ignorado pela cascata).")
    else:
        df_train, df_val = temporal_split(df, val_frac)
    print(f"Dataset: {len(df):,} amostras  "
          f"(treino={len(df_train):,}, validação={len(df_val):,})")

    # ── IF sobre o conjunto de validação ──────────────────────────────
    if_model, if_scaler = load_if(models_dir)
    X_val   = df_val[FEATURE_NAMES].values.astype(np.float32)
    X_val_s = if_scaler.transform(X_val)

    if_preds_val = if_model.predict(X_val_s)
    suspect_mask = (if_preds_val == -1)
    df_suspect   = df_val[suspect_mask].copy()
    n_suspect    = len(df_suspect)
    print(f"\nIF marcou como suspeitas: {n_suspect}/{len(df_val)} "
          f"({n_suspect/max(len(df_val),1)*100:.1f}%) da validação")

    if n_suspect == 0:
        raise RuntimeError(
            "IF não marcou nenhuma amostra como anômala na validação — "
            "impossível calibrar cascata. Aumente --contamination em train_if.py "
            "ou use mais dados de validação."
        )

    # ── LSTM nas suspeitas (sequências montadas só a partir das suspeitas) ──
    lstm_model, lstm_scaler, lstm_meta = load_lstm(models_dir)
    seq_len = lstm_meta["seq_len"]

    df_suspect_sc = df_suspect.copy()
    df_suspect_sc[FEATURE_NAMES] = lstm_scaler.transform(df_suspect[FEATURE_NAMES].values)
    seqs_suspect = build_sequences(df_suspect_sc, seq_len)

    if len(seqs_suspect) == 0:
        print(f"AVISO: suspeitos insuficientes para formar sequências de {seq_len}. "
              f"Fallback: usando τ_AE do LSTM puro ({lstm_meta['threshold']:.6f}).")
        lstm_threshold = lstm_meta["threshold"]
        recon_mean = recon_std = 0.0
        n_confirmed = 0
    else:
        with torch.no_grad():
            recon = lstm_model(torch.tensor(seqs_suspect, dtype=torch.float32)).numpy()
        recon_errors = np.mean((seqs_suspect - recon) ** 2, axis=(1, 2))
        lstm_threshold = float(np.percentile(recon_errors, lstm_threshold_pct))
        n_confirmed = int((recon_errors > lstm_threshold).sum())
        recon_mean = float(recon_errors.mean())
        recon_std  = float(recon_errors.std())

        print(f"\nLSTM sobre suspeitos:")
        print(f"  Sequências avaliadas: {len(seqs_suspect)}")
        print(f"  Erro reconstr.: mean={recon_mean:.6f} std={recon_std:.6f}")
        print(f"  Threshold cascata (p{lstm_threshold_pct:.0f}): {lstm_threshold:.6f}")
        print(f"  Confirmadas como anomalia: {n_confirmed} "
              f"({n_confirmed/len(recon_errors)*100:.1f}%)")

    # ── Salvar ────────────────────────────────────────────────────────
    meta = {
        "model": "IF+LSTM Cascade",
        "description": (
            "IF.predict == -1 envia para LSTM-AE; LSTM confirma ou descarta via "
            "erro de reconstrução."
        ),
        "val_frac": val_frac,
        "split_column": split_column,
        "seed": seed,
        "lstm_threshold_pct": lstm_threshold_pct,
        "lstm_threshold": lstm_threshold,
        "lstm_seq_len": seq_len,
        "n_val_samples": int(len(df_val)),
        "n_suspect_val": int(n_suspect),
        "n_suspect_sequences": int(len(seqs_suspect)) if n_suspect else 0,
        "n_confirmed_val": int(n_confirmed),
        "suspect_recon_mean": recon_mean,
        "suspect_recon_std":  recon_std,
        "feature_names": FEATURE_NAMES,
        "if_meta_ref":   "if_meta.json",
        "lstm_meta_ref": "lstm_meta.json",
        "period_val_start": str(df_val["window_start"].min()),
        "period_val_end":   str(df_val["window_start"].max()),
        "nodes": sorted(df["node_id"].unique().tolist()),
    }
    with open(models_dir / "cascade_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSalvo: {models_dir}/cascade_meta.json")
    print("\nResumo da cascata (operacional):")
    print(f"  IF.predict(x) ==  1  →  normal (sem LSTM)")
    print(f"  IF.predict(x) == -1  →  passa para LSTM")
    print(f"     LSTM recon_error > {lstm_threshold:.6f}  →  ANOMALIA")
    print(f"     LSTM recon_error ≤ {lstm_threshold:.6f}  →  normal (FP do IF descartado)")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",           default="data/features.parquet")
    ap.add_argument("--models",             default="data/models")
    ap.add_argument("--val-frac",           type=float, default=0.2,
                    help="Deve coincidir com o val-frac usado no treino do IF e LSTM")
    ap.add_argument("--lstm-threshold-pct", type=float, default=90.0,
                    help="Percentil do erro LSTM nos suspeitos para confirmar anomalia")
    ap.add_argument("--seed",               type=int,   default=42)
    ap.add_argument("--split-column",       default=None,
                    help="Nome da coluna do parquet com o split pré-computado "
                         "(train/val/holdout). Quando passada, --val-frac é ignorado.")
    args = ap.parse_args()

    calibrate(
        Path(args.features),
        Path(args.models),
        args.val_frac,
        args.lstm_threshold_pct,
        args.seed,
        split_column=args.split_column,
    )


if __name__ == "__main__":
    main()
