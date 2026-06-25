import json
import math
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quantlab.labeling import Labels, make_labels

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "labeling" / "labels.json"

# --------------------------------------------------------------------------- #
# Unit
# --------------------------------------------------------------------------- #


def test_simple_known_up():
    lab = make_labels([10, 11, 12, 13, 14, 15], horizon=1)
    assert lab.decision_index == (0, 1, 2, 3)
    assert lab.realize_index == (2, 3, 4, 5)
    assert lab.y_dir == (1, 1, 1, 1)
    assert math.isclose(lab.y_ret[0], math.log(12 / 11), rel_tol=1e-12)


def test_direction_down():
    lab = make_labels([15, 14, 13, 12], horizon=1)
    assert all(d == -1 for d in lab.y_dir)


def test_neutral_exact_equality():
    # O_{t+1} == O_{t+1+h} exacto -> y_ret 0.0, y_dir 0
    lab = make_labels([10, 11, 11, 12], horizon=1)
    assert lab.decision_index[0] == 0
    assert lab.y_ret[0] == 0.0
    assert lab.y_dir[0] == 0


def test_valid_index_range_and_count():
    opens = list(range(1, 21))  # N=20, positivos
    h = 5
    lab = make_labels(opens, horizon=h)
    n = len(opens)
    assert lab.decision_index[-1] == n - 2 - h
    assert len(lab.decision_index) == n - 1 - h


def test_realize_alignment():
    lab = make_labels(list(range(1, 30)), horizon=5)
    for t, r in zip(lab.decision_index, lab.realize_index):
        assert r == t + 1 + 5


def test_multi_horizon_shapes():
    opens = list(range(1, 60))
    for h in (1, 5, 20):
        lab = make_labels(opens, horizon=h)
        assert len(lab.y_ret) == len(opens) - 1 - h
        assert lab.horizon == h


def test_too_short_returns_empty():
    lab = make_labels([10, 11, 12, 13, 14, 15], horizon=5)  # N=6 < h+2=7
    assert lab.decision_index == ()
    assert lab.y_ret == ()
    assert lab.y_dir == ()
    assert lab.realize_index == ()


def test_horizon_below_one_raises():
    with pytest.raises(ValueError):
        make_labels([10, 11, 12], horizon=0)


def test_nonpositive_open_raises():
    with pytest.raises(ValueError):
        make_labels([10, 0, 12, 13], horizon=1)
    with pytest.raises(ValueError):
        make_labels([10, -5, 12, 13], horizon=1)


def test_nonfinite_open_raises():
    with pytest.raises(ValueError):
        make_labels([10, float("nan"), 12, 13], horizon=1)
    with pytest.raises(ValueError):
        make_labels([10, float("inf"), 12, 13], horizon=1)


def test_length_consistency():
    lab = make_labels(list(range(1, 25)), horizon=3)
    n = len(lab.decision_index)
    assert len(lab.y_ret) == n
    assert len(lab.y_dir) == n
    assert len(lab.realize_index) == n


def test_matches_validation_fixtures():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for case in data["cases"]:
        lab = make_labels(case["opens"], case["horizon"])
        exp = case["expected"]
        assert list(lab.decision_index) == exp["decision_index"], case["name"]
        assert list(lab.y_dir) == exp["y_dir"], case["name"]
        assert list(lab.realize_index) == exp["realize_index"], case["name"]
        assert len(lab.y_ret) == len(exp["y_ret"]), case["name"]
        for got, e in zip(lab.y_ret, exp["y_ret"]):
            assert math.isclose(got, e, rel_tol=1e-12, abs_tol=1e-12), case["name"]


# --------------------------------------------------------------------------- #
# Property-based (Hypothesis)
# --------------------------------------------------------------------------- #

_pos = st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False)


@settings(max_examples=200, deadline=None)
@given(data=st.data())
def test_length_and_realize_invariants(data):
    n = data.draw(st.integers(min_value=1, max_value=60))
    opens = data.draw(st.lists(_pos, min_size=n, max_size=n))
    h = data.draw(st.integers(min_value=1, max_value=30))
    lab = make_labels(opens, h)
    expected_len = max(0, n - 1 - h)
    assert len(lab.decision_index) == expected_len
    assert len(lab.y_ret) == expected_len
    assert len(lab.y_dir) == expected_len
    assert len(lab.realize_index) == expected_len
    assert tuple(lab.decision_index) == tuple(range(0, expected_len))
    for t, r in zip(lab.decision_index, lab.realize_index):
        assert r == t + 1 + h
        assert r <= n - 1


@settings(max_examples=200, deadline=None)
@given(data=st.data())
def test_sign_consistency_and_domain(data):
    n = data.draw(st.integers(min_value=1, max_value=60))
    opens = data.draw(st.lists(_pos, min_size=n, max_size=n))
    h = data.draw(st.integers(min_value=1, max_value=30))
    lab = make_labels(opens, h)
    for r, d in zip(lab.y_ret, lab.y_dir):
        assert d in (-1, 0, 1)
        assert (d == 1) == (r > 0)
        assert (d == -1) == (r < 0)
        assert (d == 0) == (r == 0.0)


@settings(max_examples=200, deadline=None)
@given(data=st.data())
def test_no_lookahead(data):
    n = data.draw(st.integers(min_value=5, max_value=40))
    opens = data.draw(st.lists(_pos, min_size=n, max_size=n))
    h = data.draw(st.integers(min_value=1, max_value=n - 2))
    base = make_labels(opens, h)
    assert base.decision_index  # h <= n-2 garantiza >= 1 decisión
    # perturbar el pasado (índices <= t) no debe cambiar el label de t
    k = len(base.decision_index) // 2
    t = base.decision_index[k]
    perturbed = list(opens)
    for i in range(0, t + 1):
        perturbed[i] = perturbed[i] * 1.5 + 0.123  # sigue siendo > 0
    after = make_labels(perturbed, h)
    assert after.decision_index[k] == t
    assert after.y_ret[k] == base.y_ret[k]
    assert after.y_dir[k] == base.y_dir[k]


@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_determinism(data):
    n = data.draw(st.integers(min_value=1, max_value=40))
    opens = data.draw(st.lists(_pos, min_size=n, max_size=n))
    h = data.draw(st.integers(min_value=1, max_value=30))
    assert make_labels(opens, h) == make_labels(opens, h)
