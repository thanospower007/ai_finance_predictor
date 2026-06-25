import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quantlab.metrics import (
    matthews_correlation_coefficient,
    mean_absolute_scaled_error,
)

# --------------------------------------------------------------------------- #
# MASE
# --------------------------------------------------------------------------- #


def test_mase_simple_known_case():
    # insample [1,2,3,4] -> naive MAE = 1.0 ; |errores| medio = 0.5 -> MASE 0.5
    assert mean_absolute_scaled_error([5, 6], [5.5, 5.5], [1, 2, 3, 4]) == 0.5


def test_mase_perfect_prediction_is_zero():
    assert mean_absolute_scaled_error([5, 6, 7], [5, 6, 7], [1, 2, 3, 4]) == 0.0


def test_mase_zero_denominator_raises():
    with pytest.raises(ValueError):
        mean_absolute_scaled_error([1, 2], [1, 2], [3, 3, 3])  # insample constante


def test_mase_incompatible_sizes_raises():
    with pytest.raises(ValueError):
        mean_absolute_scaled_error([1, 2], [1, 2, 3], [1, 2, 3])


def test_mase_empty_arrays_raises():
    with pytest.raises(ValueError):
        mean_absolute_scaled_error([], [], [1, 2, 3])


def test_mase_insample_too_short_raises():
    with pytest.raises(ValueError):
        mean_absolute_scaled_error([1.0], [1.0], [5.0])  # insample sin diferencias


# --------------------------------------------------------------------------- #
# MCC
# --------------------------------------------------------------------------- #


def test_mcc_perfect_prediction_is_one():
    assert matthews_correlation_coefficient([1, -1, 1, -1], [1, -1, 1, -1]) == 1.0


def test_mcc_inverse_prediction_is_minus_one():
    assert matthews_correlation_coefficient([1, -1, 1, -1], [-1, 1, -1, 1]) == -1.0


def test_mcc_partial_mix_known_value():
    yt = [1, 1, 1, -1, -1, -1]
    yp = [1, 1, -1, -1, -1, 1]
    # tp=2, tn=2, fp=1, fn=1 -> (4-1)/sqrt(81) = 3/9 = 1/3
    assert math.isclose(matthews_correlation_coefficient(yt, yp), 1.0 / 3.0, rel_tol=1e-12)


def test_mcc_excludes_neutral_observations():
    # posiciones 0 y 3 son neutras (y_true == 0) -> se excluyen
    yt = [0, 1, -1, 0, 1]
    yp = [1, 1, -1, -1, 1]
    # retenidas: (1,1), (-1,-1), (1,1) -> tp=2, tn=1 -> MCC 1.0
    assert matthews_correlation_coefficient(yt, yp) == 1.0


def test_mcc_all_neutral_raises():
    with pytest.raises(ValueError):
        matthews_correlation_coefficient([0, 0, 0], [1, -1, 1])


def test_mcc_incompatible_sizes_raises():
    with pytest.raises(ValueError):
        matthews_correlation_coefficient([1, -1], [1, -1, 1])


def test_mcc_degenerate_single_class_is_zero():
    # una sola clase verdadera presente -> denominador 0 -> convención: 0.0
    assert matthews_correlation_coefficient([1, 1, 1], [1, 1, -1]) == 0.0


# --------------------------------------------------------------------------- #
# Property-based (Hypothesis)
# --------------------------------------------------------------------------- #

_floats = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)


@settings(max_examples=200, deadline=None)
@given(data=st.data())
def test_mase_is_nonnegative(data):
    n = data.draw(st.integers(min_value=1, max_value=50))
    y_true = data.draw(st.lists(_floats, min_size=n, max_size=n))
    y_pred = data.draw(st.lists(_floats, min_size=n, max_size=n))
    m = data.draw(st.integers(min_value=2, max_value=50))
    y_ins = data.draw(st.lists(_floats, min_size=m, max_size=m))
    try:
        val = mean_absolute_scaled_error(y_true, y_pred, y_ins)
    except ValueError:
        return  # denominador 0 (insample constante): entrada degenerada admisible
    # MASE no tiene cota superior: una serie in-sample casi constante
    # (denominador subnormal pero != 0) puede desbordar a +inf, lo cual es
    # válido. El invariante congelado es solo MASE >= 0 (inf >= 0 es True;
    # un NaN haría fallar esta misma aserción).
    assert val >= 0.0


@settings(max_examples=200, deadline=None)
@given(data=st.data())
def test_mcc_in_unit_interval(data):
    n = data.draw(st.integers(min_value=1, max_value=50))
    y_true = data.draw(st.lists(st.sampled_from([-1, 0, 1]), min_size=n, max_size=n))
    y_pred = data.draw(st.lists(st.sampled_from([-1, 1]), min_size=n, max_size=n))
    try:
        val = matthews_correlation_coefficient(y_true, y_pred)
    except ValueError:
        return  # todas neutras: sin observaciones
    assert -1.0 - 1e-9 <= val <= 1.0 + 1e-9
