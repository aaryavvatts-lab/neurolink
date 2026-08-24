"""Linear alignment from neural features to image latents.

Ridge regression is the right tool at n ~= 200 trials: it is convex, has one
hyperparameter chosen by efficient leave-one-out inside the training fold, and
does not overfit the way a deep head does at this sample size. Dimensionality is
reduced by PCA fitted *inside* each training fold, never on the full dataset.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_regressor(cfg: dict, n_train: int, n_features: int) -> Pipeline:
    a = cfg["align"]
    lo, hi, num = a["ridge_alphas_log"]
    alphas = np.logspace(lo, hi, num)
    n_pc = int(min(a["n_pca_neural"], n_train - 1, n_features))
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_pc, random_state=a["seed"])),
        ("ridge", RidgeCV(alphas=alphas, alpha_per_target=True)),
    ])


def make_classifier(cfg: dict, n_train: int, n_features: int) -> Pipeline:
    a = cfg["align"]
    n_pc = int(min(a["n_pca_neural"], n_train - 1, n_features))
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_pc, random_state=a["seed"])),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def fit_predict(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray,
                cfg: dict) -> tuple[np.ndarray, Pipeline]:
    """Map neural features to image latents. Returns predictions for Xte."""
    model = make_regressor(cfg, Xtr.shape[0], Xtr.shape[1])
    model.fit(Xtr, Ytr)
    return model.predict(Xte), model


def fit_predict_classes(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                        cfg: dict) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    """7-way condition decoding directly from neural features."""
    model = make_classifier(cfg, Xtr.shape[0], Xtr.shape[1])
    model.fit(Xtr, ytr)
    return model.predict(Xte), model.predict_proba(Xte), model


def electrode_importance(model: Pipeline, n_ch: int, n_feat_per_ch: int) -> np.ndarray:
    """Per-electrode contribution: norm of that electrode's rows in the full map.

    The pipeline is scale -> PCA -> ridge, all linear, so the composed map back to
    the original feature space is components_.T @ coef_.T scaled by 1/sigma.
    """
    sc = model.named_steps["scale"]
    pca = model.named_steps["pca"]
    ridge = model.named_steps["ridge"]
    W = pca.components_.T @ np.atleast_2d(ridge.coef_).T       # (n_features, n_targets)
    W = W / (sc.scale_[:, None] + 1e-12)
    # Features are laid out as blocks of channels per feature type.
    imp = np.linalg.norm(W, axis=1).reshape(n_feat_per_ch, n_ch)
    return imp.sum(axis=0)
