"""
Parâmetros da exportação e grade temporal.

Instantes de disponibilidade: t = grid_origin + k·step, exportados para
start < t ≤ end; cada janela descreve (t − window, t]. A grade nunca reinicia no
início do arquivo, então recortes adjacentes (a, b] e (b, c] têm instantes
disjuntos. O bruto começa em `data_start()` para incluir o contexto da 1ª janela.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Optional

EPOCH = datetime(1970, 1, 1)
EXPORT_KINDS = ("raw", "features", "full")
IMPLEMENTED_KINDS = EXPORT_KINDS  # todos implementados: raw, features e full


def parse_utc(value: str) -> datetime:
    """ISO 8601 com fuso explícito → datetime UTC naive. Sem fuso: ValueError."""
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"instante sem fuso horário: {value!r} (use Z ou offset)")
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def iso_z(value: datetime) -> str:
    """UTC naive (ou aware) → ISO 8601 com Z, preservando microssegundos."""
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value.isoformat() + "Z"


@dataclass(frozen=True)
class ExportRequest:
    source_id: str
    network_id: int
    node_ids: Optional[tuple]          # None = todos os nós da rede
    start: datetime                    # exclusivo, UTC naive
    end: datetime                      # inclusivo, UTC naive
    window_seconds: int
    step_seconds: int
    grid_origin: datetime = EPOCH
    export_kind: str = "raw"
    campaign_id: Optional[str] = None

    def __post_init__(self):
        for name in ("start", "end", "grid_origin"):
            value = getattr(self, name)
            if not isinstance(value, datetime):
                raise ValueError(f"{name}: datetime obrigatório")
            if value.tzinfo is not None:
                object.__setattr__(self, name, value.astimezone(timezone.utc).replace(tzinfo=None))

    def validate(self, now: datetime, max_days: float) -> None:
        errors = []
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            errors.append("source_id ausente (configure DATASET_SOURCE_ID)")
        if self.start >= self.end:
            errors.append("início deve ser anterior ao fim")
        if self.end > now:
            errors.append("fim no futuro em relação ao corte da exportação")
        if type(self.window_seconds) is not int or self.window_seconds <= 0:
            errors.append("window_seconds deve ser positivo")
        if type(self.step_seconds) is not int or self.step_seconds <= 0:
            errors.append("step_seconds deve ser positivo")
        if not math.isfinite(max_days) or max_days <= 0:
            errors.append("limite de dias deve ser finito e positivo")
        elif self.end - self.start > timedelta(days=max_days):
            errors.append(
                f"intervalo excede {max_days:g} dias; exporte em partes com a mesma grade"
            )
        if self.export_kind not in EXPORT_KINDS:
            errors.append(f"export_kind desconhecido: {self.export_kind!r}")
        elif self.export_kind not in IMPLEMENTED_KINDS:
            errors.append(f"export_kind {self.export_kind!r} não implementado nesta versão")
        if type(self.network_id) is not int or self.network_id <= 0:
            errors.append("network_id deve ser positivo")
        if self.node_ids is not None and (not self.node_ids or any(
                type(n) is not int or n < 0 for n in self.node_ids)):
            errors.append("node_ids deve conter IDs inteiros válidos")
        if errors:
            raise ValueError("; ".join(errors))

    def grid(self) -> list:
        step = timedelta(seconds=self.step_seconds)
        k = (self.start - self.grid_origin) // step
        current = self.grid_origin + (k + 1) * step
        instants = []
        while current <= self.end:
            instants.append(current)
            current += step
        return instants

    def grid_bounds(self) -> tuple:
        """Contagem, primeiro e último instante sem alocar a grade inteira."""
        if type(self.step_seconds) is not int or self.step_seconds <= 0:
            raise ValueError("step_seconds deve ser positivo")
        step = timedelta(seconds=self.step_seconds)
        first = self.grid_origin + ((self.start - self.grid_origin) // step + 1) * step
        if first > self.end:
            return 0, None, None
        count = (self.end - first) // step + 1
        return count, first, first + (count - 1) * step

    def data_start(self) -> datetime:
        """Início (exclusivo) do bruto: contexto da 1ª janela ou o próprio início.

        O contexto inclui a janela e a leitura de bateria anterior a ela
        (``BATTERY_LOOKBACK``), que o extrator consulta.
        """
        from ai.feature_engineering import BATTERY_LOOKBACK
        _, first, _ = self.grid_bounds()
        if first is None:
            return self.start
        return min(self.start, first - timedelta(seconds=self.window_seconds) - BATTERY_LOOKBACK)
