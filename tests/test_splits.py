"""The splits must make stimulus leakage impossible. Verified, not assumed."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from neurolink.align.splits import cross_run_split, novel_image_splits


def test_novel_image_splits_never_share_a_stimulus():
    rng = np.random.default_rng(0)
    stim_id = rng.integers(1, 211, size=420)
    splits = novel_image_splits(stim_id, n_folds=6)
    assert len(splits) == 6
    for s in splits:
        s.check(stim_id)                       # raises on any leak
        assert not (set(stim_id[s.train]) & set(stim_id[s.test]))


def test_every_trial_is_tested_exactly_once():
    stim_id = np.repeat(np.arange(1, 71), 3)
    tested = np.concatenate([s.test for s in novel_image_splits(stim_id, 6)])
    assert sorted(tested) == list(range(len(stim_id)))


def test_check_detects_a_deliberate_leak():
    stim_id = np.repeat(np.arange(1, 21), 2)
    s = novel_image_splits(stim_id, 4)[0]
    s.train = np.concatenate([s.train, s.test[:1]])   # inject a leak
    with pytest.raises(AssertionError, match="leaks"):
        s.check(stim_id)


def test_cross_run_split_partitions_by_run():
    runs = np.array(["01"] * 10 + ["02"] * 10)
    s = cross_run_split(runs)
    assert len(s.train) == 10 and len(s.test) == 10
    assert set(runs[s.train]) == {"01"} and set(runs[s.test]) == {"02"}
