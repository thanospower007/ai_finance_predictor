from hypothesis import given, settings
from hypothesis import strategies as st

from quantlab.validation import walk_forward_splits


def test_explicit_small_case():
    # n=12, h=1, min_train=5, refit=1 -> gap=1, t_first=6, t_last=10
    folds = list(walk_forward_splits(12, horizon=1, min_train=5, refit_freq=1))
    got = [(f.train_index, f.test_index) for f in folds]
    expected = [
        (tuple(range(0, 5)), (6,)),
        (tuple(range(0, 6)), (7,)),
        (tuple(range(0, 7)), (8,)),
        (tuple(range(0, 8)), (9,)),
        (tuple(range(0, 9)), (10,)),
    ]
    assert got == expected


def test_first_train_window_equals_min_train():
    folds = list(walk_forward_splits(2000, horizon=5, min_train=756, refit_freq=1))
    assert len(folds[0].train_index) == 756


def test_purge_embargo_buffer_default_rule():
    # buffer entre train y test = (h-1) + h = 2h - 1
    h = 5
    folds = list(walk_forward_splits(2000, horizon=h, min_train=756, refit_freq=1))
    f = folds[0]
    assert f.test_index[0] - f.train_index[-1] - 1 == (h - 1) + h


def test_override_purge_embargo():
    folds = list(walk_forward_splits(50, horizon=3, min_train=10,
                                     refit_freq=1, purge=0, embargo=0))
    f = folds[0]
    # gap=0 -> train acaba justo antes del test
    assert f.test_index[0] - f.train_index[-1] - 1 == 0


@settings(max_examples=200, deadline=None)
@given(
    n_samples=st.integers(min_value=1, max_value=2000),
    horizon=st.integers(min_value=1, max_value=50),
    min_train=st.integers(min_value=1, max_value=500),
    refit_freq=st.integers(min_value=1, max_value=10),
)
def test_invariants(n_samples, horizon, min_train, refit_freq):
    folds = list(walk_forward_splits(n_samples, horizon,
                                     min_train=min_train, refit_freq=refit_freq))
    gap = (horizon - 1) + horizon
    prev_end = None
    for f in folds:
        tr, te = f.train_index, f.test_index
        assert tr == tuple(range(len(tr)))          # train contiguo desde 0
        assert len(tr) >= min_train                  # respeta min_train
        assert te == tuple(range(te[0], te[0] + refit_freq))  # bloque de test
        assert te[0] - tr[-1] - 1 == gap             # buffer = purge + embargo
        assert tr[-1] + horizon < te[0]              # SIN fuga de label
        assert tr[0] == 0 and te[-1] + horizon <= n_samples - 1  # en rango y realizable
        if prev_end is not None:
            assert te[0] == prev_end + 1             # avance temporal
        prev_end = te[-1]
