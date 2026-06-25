def test_frozen_params_load(frozen_params):
    assert frozen_params.horizons.values == [1, 5, 20]
    assert frozen_params.walk_forward.refit_freq == 1
    assert frozen_params.walk_forward.min_train == 756
    assert frozen_params.lockbox.min == 252
    assert frozen_params.lockbox.frac == 0.15


def test_derived_purge_embargo(frozen_params):
    for h in frozen_params.horizons.values:
        assert frozen_params.purge(h) == h - 1
        assert frozen_params.embargo(h) == h
