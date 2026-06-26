"""Evaluation Harness — orquestador walk-forward genérico, en Python puro.

Compone P1.1 (splitter), P1.2 (métricas), P1.3 (labeling) y P1.4 (lockbox) SIN
reimplementar su lógica. No entrena, no genera features, no optimiza y no conoce
ningún algoritmo. Ver docs/P1.5_evaluation_harness_design.md.

Decisiones congeladas que rigen este módulo:
- A1: MASE escalado por fold y promediado; MCC pooled sobre toda la serie OOS.
      Se conservan artefactos crudos por fold (test_index, y_true, y_pred, q_scale)
      para recomputar otras agregaciones sin reejecutar.
- B3: parametrizado por problema (RETURN | DIRECTION); una sola métrica por problema.
- C2-data: el modelo recibe rebanadas [0..t] de un handle opaco indexado por barra;
      los labels de test se ocultan; el harness es incapaz de acceder al lockbox.
- D1: la dirección predicha debe estar en {-1, +1}; un 0 lanza DirectionDomainError.
- Convención C2 del splitter: walk_forward_splits se invoca con n = development_size - 1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from quantlab.labeling import make_labels
from quantlab.lockbox import partition_lockbox
from quantlab.metrics import (
    matthews_correlation_coefficient,
    mean_absolute_scaled_error,
)
from quantlab.validation.walk_forward import walk_forward_splits


class Problem(Enum):
    """Problema evaluado. Determina target y métrica (B3)."""

    RETURN = "return"
    DIRECTION = "direction"


class Model(Protocol):
    """Interfaz pública mínima que debe cumplir cualquier modelo evaluable.

    - fit(data, y): ``data`` = rebanada de barras [0..train_end]; ``y`` = labels de
      train alineados (uno por decisión de train). Ajusta el estado interno.
    - predict(data): ``data`` = rebanada de barras [0..t]; devuelve la predicción del
      target del problema para la decisión en la ÚLTIMA barra (índice len(data)-1).
      RETURN: float real. DIRECTION: valor en {-1, +1}.
    """

    def fit(self, data: Sequence, y: Sequence[float]) -> None: ...

    def predict(self, data: Sequence) -> float: ...


class DirectionDomainError(ValueError):
    """Predicción de dirección fuera de {-1, +1} (incluye 0)."""


@dataclass(frozen=True)
class FoldResult:
    """Artefactos crudos de un fold (Decisión A)."""

    fold_id: int
    test_index: tuple[int, ...]
    y_true: tuple[float, ...]
    y_pred: tuple[float, ...]
    q_scale: float | None
    mase: float | None


@dataclass(frozen=True)
class EvaluationResult:
    """Resultado completo: artefactos por fold + agregados globales."""

    problem: Problem
    horizon: int
    development_size: int
    n_folds: int
    folds: tuple[FoldResult, ...]
    pooled_decision_index: tuple[int, ...]
    pooled_y_true: tuple[float, ...]
    pooled_y_pred: tuple[float, ...]
    mase: float | None
    mcc: float | None
    n_eval_points: int
    n_neutral_excluded: int | None


def _validate_direction_pred(value, t: int) -> int:
    # ``!= 1 and != -1`` rechaza 0, fracciones, NaN e inf (D1).
    if value != 1 and value != -1:
        raise DirectionDomainError(
            f"predicción de dirección en t={t} debe estar en {{-1, +1}}; recibido: {value!r}"
        )
    return int(value)


def evaluate(
    model: Model,
    *,
    data: Sequence,
    opens: Sequence[float],
    horizon: int,
    problem: Problem,
    lockbox_min: int,
    lockbox_frac: float,
    min_train: int,
    max_horizon: int,
    refit_freq: int = 1,
) -> EvaluationResult:
    """Evalúa ``model`` por walk-forward sobre el development (nunca el lockbox).

    Ver el documento de diseño para contratos, invariantes y edge cases.
    """
    # (1) Validación.
    if not isinstance(problem, Problem):
        raise ValueError("problem debe ser Problem.RETURN o Problem.DIRECTION")
    if horizon < 1:
        raise ValueError("horizon debe ser >= 1")
    n = len(data)
    if len(opens) != n:
        raise ValueError(f"len(data)={n} != len(opens)={len(opens)}: deben estar alineados")

    # (2) Partición P1.4 — sin construir Lockbox ni llamar reveal().
    part = partition_lockbox(
        n,
        lockbox_min=lockbox_min,
        lockbox_frac=lockbox_frac,
        min_train=min_train,
        max_horizon=max_horizon,
    )
    m = part.development_size

    # (3) Vistas de development. A partir de aquí no se lee ningún índice >= m.
    dev_data = data[0:m]
    dev_opens = opens[0:m]

    # (4) Labels P1.3 sobre la rebanada de development (Opción L).
    labels = make_labels(dev_opens, horizon)
    y_all = labels.y_ret if problem is Problem.RETURN else labels.y_dir

    # (5) Folds P1.1 con convención C2 (n = m - 1).
    splits = list(
        walk_forward_splits(
            n_samples=m - 1,
            horizon=horizon,
            min_train=min_train,
            refit_freq=refit_freq,
        )
    )
    if not splits:
        raise ValueError("cero folds: development insuficiente para el horizon dado")

    fold_results: list[FoldResult] = []
    pooled_idx: list[int] = []
    pooled_true: list = []
    pooled_pred: list = []

    # (6) Por fold.
    for k, fold in enumerate(splits):
        train_index = fold.train_index
        test_index = fold.test_index
        train_end = train_index[-1]

        y_train = [y_all[i] for i in train_index]      # labels de train (expuestos)
        data_train = dev_data[0 : train_end + 1]

        model.fit(data_train, y_train)                 # refit por fold (refit_freq=1 => cada paso)

        blk_true: list = []
        blk_pred: list = []
        for t in test_index:
            data_context = dev_data[0 : t + 1]         # solo barras <= t (anti-look-ahead)
            pred = model.predict(data_context)
            if problem is Problem.DIRECTION:
                pred = _validate_direction_pred(pred, t)
            else:
                if not math.isfinite(pred):
                    raise ValueError(f"predicción de retorno no finita en t={t}: {pred!r}")
                pred = float(pred)
            blk_true.append(y_all[t])                  # leído tras predecir; nunca al modelo
            blk_pred.append(pred)

        if problem is Problem.RETURN:
            # MASE del fold y Q_k, ambos vía P1.2 (sin reimplementar el denominador).
            mase_k = mean_absolute_scaled_error(blk_true, blk_pred, y_train)
            q_scale_k = 1.0 / mean_absolute_scaled_error([1.0], [0.0], y_train)
        else:
            mase_k = None
            q_scale_k = None

        fold_results.append(
            FoldResult(
                fold_id=k,
                test_index=tuple(test_index),
                y_true=tuple(blk_true),
                y_pred=tuple(blk_pred),
                q_scale=q_scale_k,
                mase=mase_k,
            )
        )
        pooled_idx.extend(test_index)
        pooled_true.extend(blk_true)
        pooled_pred.extend(blk_pred)

    # (8) Agregación.
    if problem is Problem.RETURN:
        agg_mase = sum(fr.mase for fr in fold_results) / len(fold_results)  # A1
        agg_mcc = None
        n_neutral = None
    else:
        agg_mase = None
        agg_mcc = matthews_correlation_coefficient(pooled_true, pooled_pred)  # MCC pooled
        n_neutral = sum(1 for v in pooled_true if v == 0)

    return EvaluationResult(
        problem=problem,
        horizon=horizon,
        development_size=m,
        n_folds=len(splits),
        folds=tuple(fold_results),
        pooled_decision_index=tuple(pooled_idx),
        pooled_y_true=tuple(pooled_true),
        pooled_y_pred=tuple(pooled_pred),
        mase=agg_mase,
        mcc=agg_mcc,
        n_eval_points=len(pooled_idx),
        n_neutral_excluded=n_neutral,
    )
