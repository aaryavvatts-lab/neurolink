"""Metrics must be exactly right at their endpoints: perfect, chance, and adversarial."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from neurolink.align.evaluate import (noise_ceiling, retrieval, rsa,
                                      two_way_identification,
                                      within_condition_retrieval)


def _rand(n=60, d=48, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d))


def test_perfect_prediction_saturates():
    T = _rand()
    assert two_way_identification(T, T) == pytest.approx(1.0)
    assert retrieval(T, T, np.arange(len(T)))["top1"] == pytest.approx(1.0)


def test_random_prediction_lands_at_chance():
    T, P = _rand(seed=1), _rand(seed=2)
    assert two_way_identification(P, T) == pytest.approx(0.5, abs=0.12)
    r = retrieval(P, T, np.arange(len(T)))
    assert r["percentile_rank"] == pytest.approx(0.5, abs=0.12)


def test_anticorrelated_prediction_is_worse_than_chance():
    T = _rand(seed=3)
    assert two_way_identification(-T, T) < 0.1


def test_rsa_is_one_against_itself_and_null_otherwise():
    T = _rand(seed=4)
    assert rsa(T, T, n_perm=100)["rho"] == pytest.approx(1.0)
    out = rsa(T, _rand(seed=5), n_perm=300)
    assert abs(out["rho"]) < 0.3 and out["p"] > 0.01


def test_within_condition_chance_matches_pool_size():
    rng = np.random.default_rng(6)
    cand = rng.normal(size=(60, 32))
    cand_cond = np.repeat([1, 2], 30)
    trial_cond = cand_cond.copy()
    P = rng.normal(size=(60, 32))
    out = within_condition_retrieval(P, cand, np.arange(60), trial_cond, cand_cond)
    assert out["mean_pool"] == pytest.approx(30.0)
    assert out["chance_top1"] == pytest.approx(1 / 30)
    assert out["percentile_rank"] == pytest.approx(0.5, abs=0.15)


def test_noise_ceiling_is_one_for_identical_measurements():
    A = _rand(seed=7)
    assert noise_ceiling(A, A)["mean_r"] == pytest.approx(1.0)
