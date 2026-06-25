"""Labeling Engine — targets congelados del proyecto, en Python puro.

Convención congelada (ver docs/P1.3_labeling_engine_design.md):

- next_open_t+1: decisión en la barra t, entrada en Open_{t+1}, salida en
  Open_{t+1+h}. El label de la decisión t se realiza en t+1+h.

    y_ret_h(t) = ln(O_{t+1+h} / O_{t+1})              (log, open-to-open)
    y_dir_h(t) = sign(y_ret_h(t))  con neutro (0) si y_ret_h(t) == 0

- Clase neutra: y_ret == 0 (regla congelada del proyecto; excluida de métricas
  binarias en P1.2). Coincide con la igualdad exacta de aperturas.

- realize_index(t) = t + 1 + h. Se EXPONE para que la validación walk-forward
  pueda conducirse por el lag real. Ratificación arquitectónica: purge=h-1 y
  embargo=h de P1.1 NO se modifican; la protección anti-leakage queda
  garantizada por esa combinación congelada (el colchón purge+embargo=2h-1
  cubre el bar de ejecución).

El motor solo etiqueta: no calcula features, no entrena, no opera. Recibe
únicamente la serie de aperturas, por lo que es estructuralmente incapaz de
usar la barra de decisión (no tiene acceso a Close).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Labels:
    """Targets alineados posición a posición sobre las decisiones válidas.

    Todos los vectores tienen la misma longitud. Para el k-ésimo elemento:
    la decisión está en la barra decision_index[k], su retorno es y_ret[k],
    su dirección y_dir[k] in {-1, 0, +1}, y se realiza en realize_index[k].
    """

    decision_index: tuple[int, ...]
    y_ret: tuple[float, ...]
    y_dir: tuple[int, ...]
    realize_index: tuple[int, ...]
    horizon: int


def make_labels(opens: Sequence[float], horizon: int) -> Labels:
    """Construye y_ret_h y y_dir_h para la serie de aperturas.

    Parameters
    ----------
    opens : Sequence[float]
        Serie de aperturas, ordenada en el tiempo, contigua y calendario-correcta
        (una apertura por barra). Todas finitas y > 0.
    horizon : int
        Horizonte h (>= 1).

    Returns
    -------
    Labels
        Targets sobre las decisiones válidas t in {0, ..., N-2-h}. Si la serie es
        demasiado corta (N < h+2), se devuelve un Labels vacío.

    Raises
    ------
    ValueError
        Si horizon < 1, o si alguna apertura no es finita o no es > 0.
    """
    if horizon < 1:
        raise ValueError("horizon debe ser >= 1")

    o = list(opens)

    for i, v in enumerate(o):
        if not math.isfinite(v):
            raise ValueError(f"apertura no finita en índice {i}: {v!r}")
        if v <= 0:
            raise ValueError(f"apertura debe ser > 0 en índice {i}: {v!r}")

    n = len(o)
    last_t = n - 2 - horizon  # último índice de decisión válido (puede ser < 0)

    decision_index: list[int] = []
    y_ret: list[float] = []
    y_dir: list[int] = []
    realize_index: list[int] = []

    for t in range(0, last_t + 1):  # rango vacío si last_t < 0
        entry = o[t + 1]
        exit_price = o[t + 1 + horizon]
        r = math.log(exit_price / entry)
        if r > 0:
            d = 1
        elif r < 0:
            d = -1
        else:
            d = 0  # neutro: y_ret == 0 (igualdad exacta de aperturas)
        decision_index.append(t)
        y_ret.append(r)
        y_dir.append(d)
        realize_index.append(t + 1 + horizon)

    return Labels(
        decision_index=tuple(decision_index),
        y_ret=tuple(y_ret),
        y_dir=tuple(y_dir),
        realize_index=tuple(realize_index),
        horizon=horizon,
    )
