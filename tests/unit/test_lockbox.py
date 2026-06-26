import json
import math
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quantlab.labeling import make_labels
from quantlab.lockbox import (
    Lockbox,
    LockboxAlreadyTouchedError,
    LockboxPartition,
    partition_lockbox,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lockbox" / "partitions.json"

# Parámetros congelados usados en los tests.
FROZEN = dict(lockbox_min=252, lockbox_frac=0.15, min_train=756, max_horizon=20)
MIN_DEV = FROZEN["min_train"] + 3 * FROZEN["max_horizon"] + 1  # 817

# --------------------------------------------------------------------------- #
# Unit — partición
# --------------------------------------------------------------------------- #


def test_frac_dominated_partition():
    p = partition_lockbox(10000, **FROZEN)
    assert p.lockbox_size == 1500
    assert p.lockbox_start == 8500
    assert p.development_size == 8500
    assert p.development_index == range(0, 8500)
    assert p.lockbox_index == range(8500, 10000)


def test_min_dominated_partition():
    p = partition_lockbox(1200, **FROZEN)
    assert p.lockbox_size == 252  # ceil(0.15*1200)=180 < 252 -> mínimo domina
    assert p.lockbox_start == 948
    assert p.development_size == 948


def test_ceil_rounding_up():
    # ceil(0.15*10001)=ceil(1500.15)=1501
    p = partition_lockbox(10001, **FROZEN)
    assert p.lockbox_size == 1501
    assert p.lockbox_start == 8500


def test_exact_feasibility_boundary():
    p = partition_lockbox(1069, **FROZEN)  # dev = 817 = MIN_DEV
    assert p.development_size == MIN_DEV


def test_just_below_boundary_raises():
    with pytest.raises(ValueError):
        partition_lockbox(1068, **FROZEN)  # dev = 816 < 817


def test_infeasible_small_dev_raises():
    with pytest.raises(ValueError):
        partition_lockbox(1000, **FROZEN)  # dev = 748


def test_lockbox_is_all_raises():
    # N pequeño: L = max(252, ceil(0.15*200)=30) = 252 >= 200 -> sin development
    with pytest.raises(ValueError):
        partition_lockbox(200, **FROZEN)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        partition_lockbox(0, **FROZEN)
    bad = dict(FROZEN)
    bad["lockbox_frac"] = 0.0
    with pytest.raises(ValueError):
        partition_lockbox(10000, **bad)
    bad = dict(FROZEN)
    bad["lockbox_frac"] = 1.0
    with pytest.raises(ValueError):
        partition_lockbox(10000, **bad)
    bad = dict(FROZEN)
    bad["lockbox_min"] = 0
    with pytest.raises(ValueError):
        partition_lockbox(10000, **bad)
    bad = dict(FROZEN)
    bad["min_train"] = 0
    with pytest.raises(ValueError):
        partition_lockbox(10000, **bad)
    bad = dict(FROZEN)
    bad["max_horizon"] = 0
    with pytest.raises(ValueError):
        partition_lockbox(10000, **bad)


def test_partition_covers_disjoint_contiguous():
    p = partition_lockbox(5000, **FROZEN)
    dev = list(p.development_index)
    lock = list(p.lockbox_index)
    assert dev[0] == 0
    assert lock[-1] == 5000 - 1
    assert dev[-1] + 1 == lock[0]                 # contiguos
    assert set(dev).isdisjoint(lock)              # disjuntos
    assert len(dev) + len(lock) == 5000           # cobertura total


def test_lockbox_is_tail():
    p = partition_lockbox(5000, **FROZEN)
    assert min(p.lockbox_index) == p.lockbox_start
    assert max(p.development_index) == p.lockbox_start - 1


def test_determinism():
    assert partition_lockbox(7777, **FROZEN) == partition_lockbox(7777, **FROZEN)


def test_matches_validation_fixtures():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for case in data["cases"]:
        params = dict(
            lockbox_min=case["lmin"],
            lockbox_frac=case["lfrac"],
            min_train=case["min_train"],
            max_horizon=case["hstar"],
        )
        exp = case["expected"]
        if exp.get("error"):
            with pytest.raises(ValueError):
                partition_lockbox(case["n_samples"], **params)
        else:
            p = partition_lockbox(case["n_samples"], **params)
            assert p.lockbox_size == exp["lockbox_size"], case["name"]
            assert p.lockbox_start == exp["lockbox_start"], case["name"]
            assert p.development_size == exp["development_size"], case["name"]


# --------------------------------------------------------------------------- #
# Unit — no-fuga hacia la lockbox (integración con el labeling P1.3)
# --------------------------------------------------------------------------- #


def test_no_label_leakage_into_lockbox():
    n = 1200
    opens = [100.0 + 0.1 * i for i in range(n)]  # positivos, estrictamente crecientes
    p = partition_lockbox(n, **FROZEN)
    dev_opens = opens[: p.lockbox_start]          # Opción L: solo la rebanada de development
    for h in (1, 5, 20):
        labels = make_labels(dev_opens, h)
        # ningún label de development debe realizarse en la lockbox
        assert all(r < p.lockbox_start for r in labels.realize_index), h
        if labels.realize_index:
            assert max(labels.realize_index) == p.lockbox_start - 1  # toca el borde, no lo cruza


# --------------------------------------------------------------------------- #
# Unit — single-touch
# --------------------------------------------------------------------------- #


def test_reveal_once_returns_range():
    rng = range(8500, 10000)
    lb = Lockbox(rng)
    assert lb.is_touched is False
    assert lb.reveal() == rng
    assert lb.is_touched is True


def test_reveal_twice_raises():
    lb = Lockbox(range(8500, 10000))
    lb.reveal()
    with pytest.raises(LockboxAlreadyTouchedError):
        lb.reveal()


def test_on_touch_hook_called_once():
    calls = []
    lb = Lockbox(range(0, 10), on_touch=calls.append)
    lb.reveal()
    assert calls == [range(0, 10)]
    with pytest.raises(LockboxAlreadyTouchedError):
        lb.reveal()
    assert calls == [range(0, 10)]  # el hook no se vuelve a invocar


def test_lockbox_from_partition():
    p = partition_lockbox(10000, **FROZEN)
    lb = Lockbox(p.lockbox_index)
    assert lb.reveal() == range(8500, 10000)


# --------------------------------------------------------------------------- #
# Property-based
# --------------------------------------------------------------------------- #


@settings(max_examples=300, deadline=None)
@given(n=st.integers(min_value=1, max_value=200000))
def test_partition_invariants_or_raises(n):
    try:
        p = partition_lockbox(n, **FROZEN)
    except ValueError:
        return  # inválido/infactible es aceptable
    # Regla de tamaño y geometría
    assert p.lockbox_size == max(FROZEN["lockbox_min"], math.ceil(FROZEN["lockbox_frac"] * n))
    assert p.lockbox_start == n - p.lockbox_size
    assert p.development_size == p.lockbox_start
    assert p.development_index == range(0, p.lockbox_start)
    assert p.lockbox_index == range(p.lockbox_start, n)
    # Cola, cobertura, disjunción
    assert p.lockbox_start >= 1
    assert p.development_size + p.lockbox_size == n
    # Garantías de factibilidad cuando NO lanza
    assert p.development_size >= MIN_DEV
    assert p.lockbox_size >= FROZEN["lockbox_min"]
    assert math.ceil(FROZEN["lockbox_frac"] * n) <= p.lockbox_size


@settings(max_examples=200, deadline=None)
@given(n=st.integers(min_value=1, max_value=199999))
def test_feasibility_monotonic(n):
    # Si N es factible, N+1 (mismos params) también lo es.
    try:
        partition_lockbox(n, **FROZEN)
    except ValueError:
        return
    partition_lockbox(n + 1, **FROZEN)  # no debe lanzar


@settings(max_examples=200, deadline=None)
@given(start=st.integers(min_value=0, max_value=10000),
       length=st.integers(min_value=1, max_value=10000))
def test_single_touch_property(start, length):
    rng = range(start, start + length)
    calls = []
    lb = Lockbox(rng, on_touch=calls.append)
    assert lb.is_touched is False
    assert lb.reveal() == rng
    assert lb.is_touched is True
    assert calls == [rng]
    with pytest.raises(LockboxAlreadyTouchedError):
        lb.reveal()
    assert lb.is_touched is True
    assert calls == [rng]
