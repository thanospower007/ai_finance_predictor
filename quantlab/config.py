"""Carga y validación de los parámetros metodológicos congelados.

Fuente: config/frozen_params.toml. Línea base del freeze metodológico
(Architecture Freeze v1.0). Cambiarlos es un cambio Clase D: requiere
justificación formal e invalida los resultados de evaluación previos.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class Horizons(BaseModel):
    values: list[int]

    @field_validator("values")
    @classmethod
    def _positive_non_empty(cls, v: list[int]) -> list[int]:
        if not v or any(h <= 0 for h in v):
            raise ValueError("horizons deben ser enteros positivos y no vacíos")
        return v


class WalkForward(BaseModel):
    refit_freq: int = Field(gt=0)
    min_train: int = Field(gt=0)
    purge_rule: str
    embargo_rule: str


class Lockbox(BaseModel):
    min: int = Field(gt=0)
    frac: float = Field(gt=0.0, lt=1.0)


class FrozenParams(BaseModel):
    horizons: Horizons
    walk_forward: WalkForward
    lockbox: Lockbox

    # Derivados con fuente única de verdad.
    def purge(self, h: int) -> int:
        """purge = h - 1 (regla congelada)."""
        return h - 1

    def embargo(self, h: int) -> int:
        """embargo = h (regla congelada)."""
        return h


def load_frozen_params(path: str | Path) -> FrozenParams:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return FrozenParams(
        horizons=Horizons(**data["horizons"]),
        walk_forward=WalkForward(**data["walk_forward"]),
        lockbox=Lockbox(**data["lockbox"]),
    )
