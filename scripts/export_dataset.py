#!/usr/bin/env python
"""
Exporta dados de uma rede/período em pacote ZIP verificável.

Origem: PostgreSQL ou SQLite (`--database-url`, padrão DATABASE_URL do backend) ou
um backup SQLite (`--sqlite`, aberto somente leitura). Mesmo código nos dois casos.

Exemplos:
    # Sobre o banco configurado em DATABASE_URL:
    python scripts/export_dataset.py --network-id 1 \\
        --start 2026-10-08T00:00:00Z --end 2026-10-15T00:00:00Z --out data/exports/semana1.zip
    # Sobre um backup SQLite (aberto somente leitura):
    python scripts/export_dataset.py --sqlite data/backup.db \\
        --source-id lab-exemplo --network-id 1 --start ... --end ... --out data/exports/semana1.zip

Opções (`--kind`): `raw` (registros brutos), `features` (11 features por nó ×
janela, com cobertura e elegibilidade) e `full` (ambos). Saída: JSON-resumo;
erros em stderr com código 2.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend" if (ROOT / "backend").is_dir() else ROOT
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="export_dataset", description="Exportação de dados para análise externa.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--database-url", help="URL SQLAlchemy da origem (padrão: DATABASE_URL)")
    src.add_argument("--sqlite", type=Path, help="arquivo SQLite (backup), aberto somente leitura")
    p.add_argument("--source-id", help="identificador da instalação (padrão: DATASET_SOURCE_ID)")
    p.add_argument("--network-id", type=int, required=True)
    p.add_argument("--nodes", help="lista de node_id separada por vírgula (padrão: todos da rede)")
    p.add_argument("--start", required=True, help="ISO 8601 com fuso, exclusivo")
    p.add_argument("--end", required=True, help="ISO 8601 com fuso, inclusivo")
    p.add_argument("--window-seconds", type=int, help="padrão: janela efetiva do IF")
    p.add_argument("--step-seconds", type=int, help="padrão: passo efetivo do IF")
    p.add_argument("--grid-origin", default="1970-01-01T00:00:00Z")
    p.add_argument("--kind", default="raw", choices=("raw", "features", "full"))
    p.add_argument("--campaign-id")
    p.add_argument("--out", type=Path, required=True)
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    from config import settings
    from ai.online.effective_config import resolve_effective_config
    from dataset_export.package import PackageError, export_dataset
    from dataset_export.request import ExportRequest, parse_utc
    from dataset_export.slice import ExportError
    from dataset_export.snapshot import build_source_engine
    from sqlalchemy.exc import SQLAlchemyError

    try:
        cfg = resolve_effective_config(None)
        request = ExportRequest(
            source_id=args.source_id if args.source_id is not None else settings.DATASET_SOURCE_ID,
            network_id=args.network_id,
            node_ids=tuple(int(n) for n in args.nodes.split(",")) if args.nodes else None,
            start=parse_utc(args.start),
            end=parse_utc(args.end),
            window_seconds=args.window_seconds if args.window_seconds is not None else cfg.if_window_minutes * 60,
            step_seconds=args.step_seconds if args.step_seconds is not None else cfg.if_step_minutes * 60,
            grid_origin=parse_utc(args.grid_origin),
            export_kind=args.kind,
            campaign_id=args.campaign_id,
        )
        if args.sqlite:
            sqlite_file = args.sqlite.resolve()
            if not sqlite_file.is_file():
                raise FileNotFoundError(f"SQLite não encontrado: {sqlite_file}")
            url = f"sqlite:///file:{sqlite_file}?mode=ro&uri=true"
        else:
            sqlite_file, url = None, args.database_url or settings.DATABASE_URL
        engine = build_source_engine(url)
        try:
            manifest = export_dataset(engine, request, args.out, settings=settings,
                                      sqlite_file=sqlite_file, now=datetime.utcnow())
        finally:
            engine.dispose()
    except (ValueError, ExportError, PackageError, FileNotFoundError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2
    except (SQLAlchemyError, OSError) as exc:
        print(f"erro de leitura/gravação ({type(exc).__name__}); nenhum pacote completo foi publicado", file=sys.stderr)
        return 2
    summary = {key: manifest[key] for key in (
        "dataset_id", "export_kind", "nodes", "tables", "timings_seconds")}
    if manifest.get("features"):
        summary["features"] = {key: manifest["features"][key]
                               for key in ("rows", "status_counts", "complete")}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
