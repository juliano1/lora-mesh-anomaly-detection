#!/usr/bin/env python
"""
Reconstrói um pacote exportado (raw/full) como banco SQLite.

    .venv/bin/python scripts/import_dataset.py --package semana1.zip --out semana1.db

Gera também `semana1.dataset.json` com o manifesto de origem e os parâmetros de
interpretação, que recálculos devem reaplicar. Erros em stderr com código 2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend" if (ROOT / "backend").is_dir() else ROOT
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="import_dataset", description="Importação de pacote exportado.")
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    from dataset_export.importer import import_dataset
    from dataset_export.package import PackageError
    from dataset_export.slice import ExportError
    from sqlalchemy.exc import SQLAlchemyError
    try:
        info = import_dataset(args.package, args.out)
    except (PackageError, FileExistsError, ValueError, ExportError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2
    except (SQLAlchemyError, OSError) as exc:
        print(f"erro de importação ({type(exc).__name__})", file=sys.stderr)
        return 2
    print(json.dumps({"database": info["database"],
                      "dataset_id": info["source_manifest"]["dataset_id"],
                      "interpretation": info["interpretation"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
