"""Lockbox Manager — partición held-out single-touch, en Python puro.

Convención congelada (ver docs/P1.4_lockbox_manager_design.md):

- Lockbox = cola de la serie:
    development = [0, lockbox_start-1]      (las primeras N-L barras)
    lockbox     = [lockbox_start, N-1]      (las últimas L barras)
  con ``lockbox_start = N - L`` y ``L = max(lockbox_min, ceil(lockbox_frac * N))``.

- Opción L (ratificada): sin buffer adicional en la frontera development/lockbox.
  El truncado natural del ``realize_index = t+1+h`` del labeling (P1.3) impide que
  un label de development use barras de la lockbox.

- Composición C2 (ratificada): el walk-forward se invoca con ``n = development_size - 1``
  (las barras del development corridas en 1 para absorber el desfase del realize_index).
  El contrato de P1.1 exige ``n >= min_train + 3h`` para >=1 fold; de ahí la
  factibilidad mínima exacta:

      development_size >= min_train + 3 * max_horizon + 1

- Single-touch: la lockbox se revela UNA vez (defensa contra data-snooping). El
  lockbox confirma al modelo ya elegido; nunca lo selecciona.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class LockboxPartition:
    """Partición inmutable development/lockbox sobre índices ``0..n_samples-1``."""

    development_index: range
    lockbox_index: range
    lockbox_start: int
    lockbox_size: int
    development_size: int


def partition_lockbox(
    n_samples: int,
    *,
    lockbox_min: int,
    lockbox_frac: float,
    min_train: int,
    max_horizon: int,
) -> LockboxPartition:
    """Particiona la serie en development (cabeza) y lockbox (cola, single-touch).

    Parameters
    ----------
    n_samples : int
        Número de observaciones (>= 1).
    lockbox_min : int
        Tamaño mínimo del lockbox (> 0). Congelado en 252.
    lockbox_frac : float
        Fracción objetivo del lockbox, en (0, 1). Congelada en 0.15.
    min_train : int
        Tamaño mínimo de entrenamiento del walk-forward (> 0). Congelado en 756.
    max_horizon : int
        Mayor horizonte h* (>= 1). Congelado en max(Horizons) = 20.

    Returns
    -------
    LockboxPartition

    Raises
    ------
    ValueError
        Si algún parámetro es inválido, si el lockbox no deja development, o si el
        development no alcanza la factibilidad mínima ``min_train + 3*max_horizon + 1``.
    """
    if n_samples < 1:
        raise ValueError("n_samples debe ser >= 1")
    if lockbox_min <= 0:
        raise ValueError("lockbox_min debe ser > 0")
    if not (0.0 < lockbox_frac < 1.0):
        raise ValueError("lockbox_frac debe estar en (0, 1)")
    if min_train <= 0:
        raise ValueError("min_train debe ser > 0")
    if max_horizon < 1:
        raise ValueError("max_horizon debe ser >= 1")

    # Regla congelada de tamaño (ceil; el mínimo domina en N pequeño).
    lockbox_size = max(lockbox_min, math.ceil(lockbox_frac * n_samples))

    if lockbox_size >= n_samples:
        raise ValueError(
            f"lockbox (L={lockbox_size}) >= n_samples ({n_samples}): no queda development"
        )

    lockbox_start = n_samples - lockbox_size
    development_size = lockbox_start  # = N - L

    # Factibilidad C2: el splitter se invoca con n = development_size - 1 y P1.1
    # exige n >= min_train + 3h*  =>  development_size >= min_train + 3*max_horizon + 1.
    min_dev = min_train + 3 * max_horizon + 1
    if development_size < min_dev:
        raise ValueError(
            f"development_size ({development_size}) < mínimo factible "
            f"({min_dev} = min_train + 3*max_horizon + 1) para h*={max_horizon}"
        )

    return LockboxPartition(
        development_index=range(0, lockbox_start),
        lockbox_index=range(lockbox_start, n_samples),
        lockbox_start=lockbox_start,
        lockbox_size=lockbox_size,
        development_size=development_size,
    )


class LockboxAlreadyTouchedError(RuntimeError):
    """Se intentó revelar la lockbox más de una vez (violación del single-touch)."""


class Lockbox:
    """Guard de un solo toque sobre el rango de índices de la lockbox.

    ``reveal()`` devuelve el rango la PRIMERA vez, marca ``is_touched`` e invoca el
    hook de auditoría ``on_touch`` (si se provee). Cualquier acceso posterior lanza
    ``LockboxAlreadyTouchedError``. El toque se consume en el intento: si el hook
    falla, la lockbox queda marcada igualmente (sin reintento).

    Nota: este guard es en-proceso. El single-touch durable entre sesiones requiere
    persistir el toque (lockbox_registry), cableado en una integración posterior; el
    hook ``on_touch`` es el punto de extensión para esa persistencia.
    """

    def __init__(
        self,
        lockbox_index: range,
        on_touch: Callable[[range], None] | None = None,
    ) -> None:
        self._lockbox_index = lockbox_index
        self._on_touch = on_touch
        self._touched = False

    @property
    def is_touched(self) -> bool:
        return self._touched

    def reveal(self) -> range:
        if self._touched:
            raise LockboxAlreadyTouchedError(
                "la lockbox ya fue revelada (single-touch): acceso rechazado"
            )
        self._touched = True
        if self._on_touch is not None:
            self._on_touch(self._lockbox_index)
        return self._lockbox_index
