"""Walk-forward splitter con purga y embargo.

Observaciones ordenadas en el tiempo, índices ``0..n-1``. La observación ``i``
tiene features disponibles en ``i`` y un target de horizonte ``h`` cuyo valor se
realiza en ``i+h``.

Para un punto de prueba en el índice ``t`` (bloque de tamaño ``refit_freq``):

- ``train = [0, t - 1 - purge - embargo]``   (ventana expansiva)
- ``test  = [t, t + refit_freq - 1]``

Por el freeze metodológico:

- ``purge   = h - 1``  descarta observaciones de train cuyo label ``[i, i+h]``
  solaparía el punto de prueba (solapamiento por horizonte).
- ``embargo = h``      separación adicional que neutraliza la dependencia serial
  entre el final del train y el punto de prueba.

La separación de observaciones NO usadas entre el fin de train y el primer punto
de prueba es ``purge + embargo = 2h - 1``. Consecuencia: el último label de train
se realiza en ``t - h`` (h observaciones antes del feature de prueba): no hay fuga.

Convención: especialización *forward* del esquema de López de Prado. Como en
walk-forward no existe entrenamiento posterior al test, el embargo se aplica como
separación adicional ANTES del test. Ver ``README.md`` de este paquete.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    """Un fold walk-forward. Índices sobre la serie ordenada en el tiempo."""

    train_index: tuple[int, ...]
    test_index: tuple[int, ...]


def _resolve_gap(horizon: int, purge: int | None, embargo: int | None) -> tuple[int, int]:
    p = horizon - 1 if purge is None else purge
    e = horizon if embargo is None else embargo
    return p, e


def walk_forward_splits(
    n_samples: int,
    horizon: int,
    min_train: int = 756,
    refit_freq: int = 1,
    purge: int | None = None,
    embargo: int | None = None,
):
    """Genera folds walk-forward con purga y embargo.

    Parameters
    ----------
    n_samples : int
        Número de observaciones (>= 1).
    horizon : int
        Horizonte h (>= 1). El label de ``i`` se realiza en ``i+h``.
    min_train : int
        Tamaño mínimo de la ventana de entrenamiento (>= 1).
    refit_freq : int
        Pasos entre reajustes = tamaño del bloque de test (>= 1). Congelado en 1.
    purge, embargo : int | None
        Si ``None`` se derivan del freeze: ``purge=h-1``, ``embargo=h``.
        Override solo para pruebas.

    Yields
    ------
    Fold
        ``train_index`` y ``test_index`` como tuplas de enteros.

    Raises
    ------
    ValueError
        Si algún parámetro es inválido.
    """
    if n_samples < 1:
        raise ValueError("n_samples debe ser >= 1")
    if horizon < 1:
        raise ValueError("horizon debe ser >= 1")
    if min_train < 1:
        raise ValueError("min_train debe ser >= 1")
    if refit_freq < 1:
        raise ValueError("refit_freq debe ser >= 1")

    p, e = _resolve_gap(horizon, purge, embargo)
    if p < 0 or e < 0:
        raise ValueError("purge y embargo deben ser >= 0")

    return _generate(n_samples, horizon, min_train, refit_freq, p, e)


def _generate(n_samples, horizon, min_train, refit_freq, purge, embargo):
    gap = purge + embargo  # observaciones no usadas entre train y test
    # |train| = (t - 1 - gap) + 1 = t - gap >= min_train  ->  t >= min_train + gap
    t_first = min_train + gap
    # último label del bloque realizado: (t + refit_freq - 1) + h <= n - 1
    t_last = n_samples - refit_freq - horizon

    t = t_first
    while t <= t_last:
        train_end = t - 1 - gap
        yield Fold(
            train_index=tuple(range(0, train_end + 1)),
            test_index=tuple(range(t, t + refit_freq)),
        )
        t += refit_freq
