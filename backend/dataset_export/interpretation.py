"""
Parâmetros que definem como o pacote é interpretado.

Gravados no manifesto na exportação e reaplicados ao recalcular sobre dados
importados. A configuração local da máquina receptora não muda origem nem
elegibilidade em silêncio: divergência de um valor definido explicitamente
gera erro; substituir exige override explícito, registrado no derivado.
`source_id` identifica a fonte dos dados e nunca é substituído.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Optional
import math

from dataset_export.request import iso_z, parse_utc

OVERRIDABLE = ("data_origin_tracked_since", "reference_margin_minutes")


class InterpretationConflict(Exception):
    """Configuração local explícita diverge do manifesto sem override."""


def _parse_optional(value) -> Optional[datetime]:
    return parse_utc(value) if value else None


@dataclass(frozen=True)
class InterpretationParams:
    source_id: str
    data_origin_tracked_since: Optional[datetime]
    reference_margin_minutes: float
    window_seconds: int
    step_seconds: int
    grid_origin: datetime

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id ausente")
        if not math.isfinite(self.reference_margin_minutes) or self.reference_margin_minutes < 0:
            raise ValueError("reference_margin_minutes deve ser finito e não negativo")
        if any(type(n) is not int or n <= 0 for n in (self.window_seconds, self.step_seconds)):
            raise ValueError("janela e passo devem ser inteiros positivos")

    @classmethod
    def for_export(cls, request, settings) -> "InterpretationParams":
        return cls(
            source_id=request.source_id,
            data_origin_tracked_since=_parse_optional(settings.DATA_ORIGIN_TRACKED_SINCE),
            reference_margin_minutes=float(settings.ML_REFERENCE_MARGIN_MINUTES),
            window_seconds=request.window_seconds,
            step_seconds=request.step_seconds,
            grid_origin=request.grid_origin,
        )

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "data_origin_tracked_since": (
                iso_z(self.data_origin_tracked_since) if self.data_origin_tracked_since else None
            ),
            "reference_margin_minutes": self.reference_margin_minutes,
            "window_seconds": self.window_seconds,
            "step_seconds": self.step_seconds,
            "grid_origin": iso_z(self.grid_origin),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InterpretationParams":
        return cls(
            source_id=data["source_id"],
            data_origin_tracked_since=_parse_optional(data.get("data_origin_tracked_since")),
            reference_margin_minutes=float(data["reference_margin_minutes"]),
            window_seconds=int(data["window_seconds"]),
            step_seconds=int(data["step_seconds"]),
            grid_origin=parse_utc(data["grid_origin"]),
        )


def resolve_for_computation(params: InterpretationParams, settings,
                            overrides: Optional[dict] = None):
    """(parâmetros efetivos, overrides aplicados) para recalcular sobre o pacote."""
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(OVERRIDABLE)
    if unknown:
        raise ValueError(f"parâmetros não substituíveis: {sorted(unknown)}")
    explicit = set(getattr(settings, "model_fields_set", set()))
    conflicts = []
    local_tracked = _parse_optional(settings.DATA_ORIGIN_TRACKED_SINCE)
    if (local_tracked is not None and local_tracked != params.data_origin_tracked_since
            and "data_origin_tracked_since" not in overrides):
        conflicts.append("data_origin_tracked_since local diverge do manifesto")
    local_margin = float(settings.ML_REFERENCE_MARGIN_MINUTES)
    if ("ML_REFERENCE_MARGIN_MINUTES" in explicit
            and local_margin != params.reference_margin_minutes
            and "reference_margin_minutes" not in overrides):
        conflicts.append("reference_margin_minutes local diverge do manifesto")
    if conflicts:
        raise InterpretationConflict("; ".join(conflicts) + " — use override explícito")
    effective, applied = params, {}
    for key, value in overrides.items():
        if key == "data_origin_tracked_since" and isinstance(value, str):
            value = parse_utc(value)
        if key == "reference_margin_minutes":
            value = float(value)
        effective = replace(effective, **{key: value})
        applied[key] = {"manifest": getattr(params, key), "override": value}
    return effective, applied
