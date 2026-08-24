"""Train/test splits designed so stimulus information cannot leak.

Three regimes, answering three different questions:

A  novel-image   GroupKFold grouped by stimulus image. A test image never appears
                 in training, so performance reflects genuine generalisation to
                 an unseen stimulus. This is the honest headline setting.
B  cross-run     (sub-02 only) train on run-01, test on run-02. The same images
                 recur, so this measures image-specific identifiability -- an
                 upper bound, and the setting where a noise ceiling is definable.
C  cross-subject train on one subject, test on the other, bridged only through
                 the shared image latent space. Different hemisphere, electrode
                 count and sample rate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold


@dataclass
class Split:
    name: str
    train: np.ndarray
    test: np.ndarray
    fold: int = 0

    def check(self, groups: np.ndarray) -> None:
        """Assert no group appears on both sides. Called by every consumer."""
        overlap = set(groups[self.train]) & set(groups[self.test])
        if overlap:
            raise AssertionError(
                f"split {self.name!r} fold {self.fold} leaks {len(overlap)} stimuli "
                f"between train and test: {sorted(overlap)[:5]}")


def novel_image_splits(stim_id: np.ndarray, n_folds: int = 6) -> list[Split]:
    """Split A. Groups are stimulus images."""
    gkf = GroupKFold(n_splits=n_folds)
    idx = np.arange(len(stim_id))
    out = []
    for k, (tr, te) in enumerate(gkf.split(idx, groups=stim_id)):
        s = Split(name="novel_image", train=tr, test=te, fold=k)
        s.check(stim_id)
        out.append(s)
    return out


def cross_run_split(run_labels: np.ndarray, train_run: str = "01",
                    test_run: str = "02") -> Split:
    """Split B. Deliberately does NOT check for image overlap -- overlap is the point."""
    return Split(name="cross_run",
                 train=np.flatnonzero(run_labels == train_run),
                 test=np.flatnonzero(run_labels == test_run))


def cross_subject_split(sub_labels: np.ndarray, train_sub: str, test_sub: str) -> Split:
    return Split(name=f"cross_subject_{train_sub}->{test_sub}",
                 train=np.flatnonzero(sub_labels == train_sub),
                 test=np.flatnonzero(sub_labels == test_sub))
