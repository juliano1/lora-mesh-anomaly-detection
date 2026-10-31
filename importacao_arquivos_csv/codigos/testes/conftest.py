"""Torna o script do detector importável nos testes (a pasta codigos/ não é um pacote)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
