import pytest

from quantlab.validation import walk_forward_splits


def test_empty_when_insufficient_data():
    # n demasiado pequeño -> ningún fold
    assert list(walk_forward_splits(10, horizon=5, min_train=10, refit_freq=1)) == []


def test_exactly_one_fold():
    # t_first == t_last  =>  exactamente 1 fold
    # h=1, min_train=5 -> gap=1, t_first=6 ; t_last = n-1-h = 6 -> n=8
    folds = list(walk_forward_splits(8, horizon=1, min_train=5, refit_freq=1))
    assert len(folds) == 1
    assert folds[0].test_index == (6,)


def test_horizon_one_no_purge():
    # h=1 -> purge=0 (embargo=1)
    folds = list(walk_forward_splits(20, horizon=1, min_train=5, refit_freq=1))
    f = folds[0]
    assert f.test_index[0] - f.train_index[-1] - 1 == 1  # solo embargo


def test_refit_freq_block():
    # refit_freq=2 -> bloques de test de tamaño 2, paso 2
    folds = list(walk_forward_splits(20, horizon=1, min_train=5, refit_freq=2))
    assert folds[0].test_index == (6, 7)
    assert folds[1].test_index == (8, 9)
    # último bloque con label realizado
    assert folds[-1].test_index[-1] + 1 <= 19


@pytest.mark.parametrize("kwargs", [
    {"n_samples": 0, "horizon": 1},
    {"n_samples": 10, "horizon": 0},
    {"n_samples": 10, "horizon": 1, "min_train": 0},
    {"n_samples": 10, "horizon": 1, "refit_freq": 0},
    {"n_samples": 10, "horizon": 1, "purge": -1},
    {"n_samples": 10, "horizon": 1, "embargo": -1},
])
def test_invalid_params_raise(kwargs):
    with pytest.raises(ValueError):
        walk_forward_splits(**kwargs)
