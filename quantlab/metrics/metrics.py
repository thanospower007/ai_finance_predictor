"""Métricas núcleo del benchmark, en Python puro (solo librería estándar).

- MASE (retorno):   mean_absolute_scaled_error
- MCC  (dirección): matthews_correlation_coefficient

Convenciones congeladas: ver docstrings de cada función y docs del proyecto.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def mean_absolute_scaled_error(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    y_insample: Sequence[float],
) -> float:
    """MASE no estacional con escalado in-sample (naive de 1 paso).

        MASE = mean(|y_true - y_pred|) / mean(|y_insample[t] - y_insample[t-1]|)

    El numerador es el MAE *out-of-sample* (test). El denominador es el MAE del
    naive de un paso sobre la serie *in-sample* (entrenamiento), independiente
    del horizonte h.

    Parameters
    ----------
    y_true, y_pred : Sequence[float]
        Valores reales y predichos out-of-sample. Mismo tamaño, no vacíos.
    y_insample : Sequence[float]
        Serie in-sample (entrenamiento) para el naive de 1 paso. Tamaño >= 2.

    Returns
    -------
    float
        MASE >= 0.

    Raises
    ------
    ValueError
        Si y_true o y_pred están vacíos; si difieren en tamaño; si y_insample
        tiene menos de 2 observaciones; o si el denominador es 0 (serie
        in-sample constante).
    """
    yt = list(y_true)
    yp = list(y_pred)
    yi = list(y_insample)

    if len(yt) == 0 or len(yp) == 0:
        raise ValueError("y_true e y_pred no pueden estar vacíos")
    if len(yt) != len(yp):
        raise ValueError(
            f"tamaños incompatibles: len(y_true)={len(yt)} != len(y_pred)={len(yp)}"
        )
    if len(yi) < 2:
        raise ValueError(
            "y_insample debe tener al menos 2 observaciones para el naive de 1 paso"
        )

    numerator = sum(abs(a - p) for a, p in zip(yt, yp)) / len(yt)
    denominator = sum(abs(yi[t] - yi[t - 1]) for t in range(1, len(yi))) / (len(yi) - 1)

    if denominator == 0:
        raise ValueError(
            "denominador cero: la serie in-sample es constante (naive MAE = 0)"
        )

    return numerator / denominator


def matthews_correlation_coefficient(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> float:
    """MCC binario para dirección. Clases: +1 = subida, -1 = bajada.

    Regla congelada: las observaciones neutras (y_true == 0) se excluyen por
    completo antes del cálculo; la métrica binaria nunca usa la clase neutra.

        MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    Convención (no cubierta por el spec, fijada aquí): si tras excluir neutras la
    matriz de confusión es degenerada (denominador 0, p.ej. una sola clase
    presente en y_true o en y_pred), MCC = 0.0 (convención de sklearn). Es
    distinto de "no quedan observaciones", que lanza ValueError.

    Parameters
    ----------
    y_true : Sequence[int]
        Etiquetas reales en {-1, 0, +1} (0 = neutra, se excluye).
    y_pred : Sequence[int]
        Etiquetas predichas en {-1, +1} (sobre las posiciones retenidas).

    Returns
    -------
    float
        MCC en [-1, 1].

    Raises
    ------
    ValueError
        Si y_true e y_pred difieren en tamaño; si tras excluir neutras no quedan
        observaciones; o si hay etiquetas inválidas.
    """
    yt = list(y_true)
    yp = list(y_pred)

    if len(yt) != len(yp):
        raise ValueError(
            f"tamaños incompatibles: len(y_true)={len(yt)} != len(y_pred)={len(yp)}"
        )

    for v in yt:
        if v not in (-1, 0, 1):
            raise ValueError(f"y_true debe estar en {{-1, 0, 1}}; valor inválido: {v}")

    # Excluir neutras (y_true == 0) por posición.
    pairs = [(a, p) for a, p in zip(yt, yp) if a != 0]
    if len(pairs) == 0:
        raise ValueError("no quedan observaciones tras excluir la clase neutra")

    tp = tn = fp = fn = 0
    for a, p in pairs:
        if p not in (-1, 1):
            raise ValueError(f"y_pred debe ser ±1 (sin neutra); valor inválido: {p}")
        if a == 1 and p == 1:
            tp += 1          # positivo = +1 (subida)
        elif a == -1 and p == -1:
            tn += 1
        elif a == -1 and p == 1:
            fp += 1
        else:  # a == 1 and p == -1
            fn += 1

    numerator = tp * tn - fp * fn
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_sq == 0:
        return 0.0
    return numerator / math.sqrt(denom_sq)
