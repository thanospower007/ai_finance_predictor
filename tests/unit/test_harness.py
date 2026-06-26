import json
import math
from pathlib import Path

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from quantlab.evaluation import (
    DirectionDomainError,
    EvaluationResult,
    FoldResult,
    Problem,
    evaluate,
)
from quantlab.labeling import make_labels
from quantlab.metrics import matthews_correlation_coefficient
from quantlab.lockbox import partition_lockbox

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "evaluation" / "harness_cases.json"

# --------------------------------------------------------------------------- #
# Modelos de referencia deterministas (mismas reglas que el generador de fixtures)
# --------------------------------------------------------------------------- #


class LastReturnModel:
    """Predice el último retorno log de 1 paso de la ventana de data."""

    def fit(self, data, y):  # naive: no usa el ajuste
        pass

    def predict(self, data):
        return math.log(data[-1] / data[-2])


class SignLastModel:
    """Signo del último retorno; sign(0):=+1 -> salida siempre en {-1,+1}."""

    def fit(self, data, y):
        pass

    def predict(self, data):
        r = math.log(data[-1] / data[-2])
        return 1 if r > 0 else (-1 if r < 0 else 1)


class ZeroDirModel:
    def fit(self, data, y):
        pass

    def predict(self, data):
        return 0  # viola D1


class NonFiniteReturnModel:
    def fit(self, data, y):
        pass

    def predict(self, data):
        return float("inf")


# Parámetros pequeños y factibles para property-based.
PARAMS = dict(lockbox_min=5, lockbox_frac=0.15, min_train=10, max_horizon=20)
_pos = st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False)


def _cumsum(start, incs):
    vals = [start]
    for d in incs:
        vals.append(vals[-1] + d)
    return vals


def _increasing(n):
    """Estrategia: serie positiva estrictamente creciente de longitud n (sin neutros)."""
    return st.tuples(
        st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        st.lists(
            st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=n - 1,
            max_size=n - 1,
        ),
    ).map(lambda sp: _cumsum(sp[0], sp[1]))


def _close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def _assert_seq_close(got, exp, name):
    assert len(got) == len(exp), name
    for g, e in zip(got, exp):
        assert _close(g, e), f"{name}: {g} != {e}"


def _model_for(name):
    return {"last_return": LastReturnModel(), "sign_last": SignLastModel()}[name]


# --------------------------------------------------------------------------- #
# Unit — fixtures de referencia independientes
# --------------------------------------------------------------------------- #


def test_matches_validation_fixtures():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for case in data["cases"]:
        problem = Problem.RETURN if case["problem"] == "return" else Problem.DIRECTION
        res = evaluate(
            _model_for(case["model"]),
            data=case["opens"],
            opens=case["opens"],
            horizon=case["horizon"],
            problem=problem,
            lockbox_min=case["lmin"],
            lockbox_frac=case["lfrac"],
            min_train=case["min_train"],
            max_horizon=case["hstar"],
        )
        exp = case["expected"]
        nm = case["name"]
        assert res.development_size == exp["development_size"], nm
        assert res.n_folds == exp["n_folds"], nm
        assert list(res.pooled_decision_index) == exp["pooled_decision_index"], nm
        assert res.n_eval_points == exp["n_eval_points"], nm
        _assert_seq_close(res.pooled_y_true, exp["pooled_y_true"], nm)
        _assert_seq_close(res.pooled_y_pred, exp["pooled_y_pred"], nm)
        if problem is Problem.RETURN:
            assert res.mcc is None and res.n_neutral_excluded is None, nm
            assert _close(res.mase, exp["mase"]), nm
        else:
            assert res.mase is None, nm
            assert _close(res.mcc, exp["mcc"]), nm
            assert res.n_neutral_excluded == exp["n_neutral_excluded"], nm
        assert len(res.folds) == len(exp["folds"]), nm
        for fr, ef in zip(res.folds, exp["folds"]):
            assert list(fr.test_index) == ef["test_index"], nm
            _assert_seq_close(fr.y_true, ef["y_true"], nm)
            _assert_seq_close(fr.y_pred, ef["y_pred"], nm)
            if ef["q_scale"] is None:
                assert fr.q_scale is None and fr.mase is None, nm
            else:
                assert _close(fr.q_scale, ef["q_scale"]), nm
                assert _close(fr.mase, ef["mase"]), nm


# --------------------------------------------------------------------------- #
# Unit — comportamiento y edge cases
# --------------------------------------------------------------------------- #

_SMOKE = dict(lockbox_min=5, lockbox_frac=0.15, min_train=10, max_horizon=20)
_OPENS_UP = [100.0 + i for i in range(120)]  # estrictamente creciente, positivo


def test_direction_domain_error_on_zero():
    with pytest.raises(DirectionDomainError):
        evaluate(ZeroDirModel(), data=_OPENS_UP, opens=_OPENS_UP, horizon=1,
                 problem=Problem.DIRECTION, **_SMOKE)


def test_return_non_finite_raises():
    with pytest.raises(ValueError):
        evaluate(NonFiniteReturnModel(), data=_OPENS_UP, opens=_OPENS_UP, horizon=1,
                 problem=Problem.RETURN, **_SMOKE)


def test_all_neutral_direction_raises():
    flat = [100.0] * 120  # todos los labels y_dir == 0
    with pytest.raises(ValueError):
        evaluate(SignLastModel(), data=flat, opens=flat, horizon=1,
                 problem=Problem.DIRECTION, **_SMOKE)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate(SignLastModel(), data=_OPENS_UP, opens=_OPENS_UP[:-1], horizon=1,
                 problem=Problem.DIRECTION, **_SMOKE)


def test_infeasible_n_raises():
    short = [100.0 + i for i in range(8)]  # development < 817-equivalente
    with pytest.raises(ValueError):
        evaluate(SignLastModel(), data=short, opens=short, horizon=1,
                 problem=Problem.DIRECTION, lockbox_min=2, lockbox_frac=0.15,
                 min_train=3, max_horizon=2)


def test_composition_smoke_all_horizons():
    for h in (1, 5, 20):
        res = evaluate(SignLastModel(), data=_OPENS_UP, opens=_OPENS_UP, horizon=h,
                       problem=Problem.DIRECTION, **_SMOKE)
        assert res.n_folds > 0
        assert res.n_eval_points == len(res.pooled_decision_index)
        assert res.n_eval_points == res.n_folds  # refit_freq=1 -> 1 punto por fold
        assert res.mcc is not None and res.mase is None
        assert res.horizon == h


def test_no_double_evaluation():
    res = evaluate(SignLastModel(), data=_OPENS_UP, opens=_OPENS_UP, horizon=1,
                   problem=Problem.DIRECTION, **_SMOKE)
    idx = list(res.pooled_decision_index)
    assert idx == sorted(idx)                       # ascendente
    assert len(set(idx)) == len(idx)                # único
    assert idx == list(range(idx[0], idx[-1] + 1))  # contiguo


def test_index_alignment_with_labels():
    res = evaluate(SignLastModel(), data=_OPENS_UP, opens=_OPENS_UP, horizon=1,
                   problem=Problem.DIRECTION, **_SMOKE)
    dev_opens = _OPENS_UP[: res.development_size]
    labels = make_labels(dev_opens, 1)
    valid = set(labels.decision_index)
    for fr in res.folds:
        for t, yt in zip(fr.test_index, fr.y_true):
            assert t in valid                       # todo índice de test tiene label
            assert yt == labels.y_dir[t]            # label correcto


def test_metric_provenance_return():
    res = evaluate(LastReturnModel(), data=_OPENS_UP, opens=_OPENS_UP, horizon=1,
                   problem=Problem.RETURN, **_SMOKE)
    # A1: la media de MASE por fold reproduce el agregado
    assert _close(res.mase, sum(f.mase for f in res.folds) / len(res.folds))
    # mase_k * Q_k == |error| (un punto por fold)
    for f in res.folds:
        assert math.isclose(f.mase * f.q_scale, abs(f.y_true[0] - f.y_pred[0]),
                            rel_tol=1e-9, abs_tol=1e-12)


def test_metric_provenance_direction():
    res = evaluate(SignLastModel(), data=_OPENS_UP, opens=_OPENS_UP, horizon=1,
                   problem=Problem.DIRECTION, **_SMOKE)
    # el MCC reportado se reproduce con P1.2 sobre los artefactos pooled
    recomputed = matthews_correlation_coefficient(res.pooled_y_true, res.pooled_y_pred)
    assert _close(res.mcc, recomputed)


# --------------------------------------------------------------------------- #
# Property-based
# --------------------------------------------------------------------------- #


@settings(max_examples=60, deadline=None)
@given(data=st.data())
def test_no_lockbox_access(data):
    n = data.draw(st.integers(min_value=85, max_value=110))
    opens = data.draw(_increasing(n))
    h = data.draw(st.sampled_from([1, 5, 20]))
    base = evaluate(SignLastModel(), data=opens, opens=opens, horizon=h,
                    problem=Problem.DIRECTION, **PARAMS)
    m = base.development_size
    # perturbar TODA la cola del lockbox [m, n) no debe cambiar el resultado
    perturbed = list(opens)
    for i in range(m, n):
        perturbed[i] = perturbed[i] * 3.0 + 7.0  # sigue siendo > 0
    after = evaluate(SignLastModel(), data=perturbed, opens=perturbed, horizon=h,
                     problem=Problem.DIRECTION, **PARAMS)
    assert after == base


@settings(max_examples=60, deadline=None)
@given(data=st.data())
def test_no_lookahead(data):
    n = data.draw(st.integers(min_value=85, max_value=110))
    feat = data.draw(st.lists(_pos, min_size=n, max_size=n))   # data del modelo
    opens = data.draw(_increasing(n))  # series para labels
    base = evaluate(SignLastModel(), data=feat, opens=opens, horizon=1,
                    problem=Problem.DIRECTION, **PARAMS)
    m = base.development_size
    t_first = base.folds[0].test_index[0]
    j = t_first + 1  # índice futuro respecto del primer fold, dentro del development
    assert t_first < j < m
    perturbed = list(feat)
    perturbed[j] = perturbed[j] * 2.0 + 1.0  # cambia el futuro de data
    after = evaluate(SignLastModel(), data=perturbed, opens=opens, horizon=1,
                     problem=Problem.DIRECTION, **PARAMS)
    # la predicción del primer fold (ventana [0..t_first]) no puede cambiar
    assert after.folds[0].y_pred == base.folds[0].y_pred


@settings(max_examples=60, deadline=None)
@given(data=st.data())
def test_determinism(data):
    n = data.draw(st.integers(min_value=85, max_value=110))
    opens = data.draw(_increasing(n))
    h = data.draw(st.sampled_from([1, 5, 20]))
    r1 = evaluate(SignLastModel(), data=opens, opens=opens, horizon=h,
                  problem=Problem.DIRECTION, **PARAMS)
    r2 = evaluate(SignLastModel(), data=opens, opens=opens, horizon=h,
                  problem=Problem.DIRECTION, **PARAMS)
    assert r1 == r2


@settings(max_examples=40, deadline=None)
@given(data=st.data())
def test_direction_domain_enforced(data):
    n = data.draw(st.integers(min_value=85, max_value=110))
    opens = data.draw(_increasing(n))
    # un modelo en dominio nunca lanza
    evaluate(SignLastModel(), data=opens, opens=opens, horizon=1,
             problem=Problem.DIRECTION, **PARAMS)
    # un modelo que emite 0 siempre lanza
    with pytest.raises(DirectionDomainError):
        evaluate(ZeroDirModel(), data=opens, opens=opens, horizon=1,
                 problem=Problem.DIRECTION, **PARAMS)


@settings(max_examples=60, deadline=None)
@given(data=st.data())
def test_pooling_and_no_double_count(data):
    n = data.draw(st.integers(min_value=85, max_value=110))
    opens = data.draw(_increasing(n))
    h = data.draw(st.sampled_from([1, 5, 20]))
    res = evaluate(SignLastModel(), data=opens, opens=opens, horizon=h,
                   problem=Problem.DIRECTION, **PARAMS)
    # pooled = concatenación de los bloques por fold, en orden
    cat_idx, cat_true, cat_pred = [], [], []
    for fr in res.folds:
        cat_idx += list(fr.test_index)
        cat_true += list(fr.y_true)
        cat_pred += list(fr.y_pred)
    assert list(res.pooled_decision_index) == cat_idx
    assert list(res.pooled_y_true) == cat_true
    assert list(res.pooled_y_pred) == cat_pred
    assert res.n_eval_points == len(cat_idx)
    # índices únicos y contiguos (refit_freq=1)
    assert len(set(cat_idx)) == len(cat_idx)
    assert cat_idx == list(range(cat_idx[0], cat_idx[-1] + 1))


@settings(max_examples=40, deadline=None)
@given(data=st.data())
def test_return_provenance_property(data):
    n = data.draw(st.integers(min_value=85, max_value=110))
    opens = data.draw(_increasing(n))
    try:
        res = evaluate(LastReturnModel(), data=opens, opens=opens, horizon=1,
                       problem=Problem.RETURN, **PARAMS)
    except ValueError:
        assume(False)  # serie con in-sample constante: descartar
    assert _close(res.mase, sum(f.mase for f in res.folds) / len(res.folds))
    for f in res.folds:
        assert math.isclose(f.mase * f.q_scale, abs(f.y_true[0] - f.y_pred[0]),
                            rel_tol=1e-9, abs_tol=1e-12)
