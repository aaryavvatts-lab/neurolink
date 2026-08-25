"""Forward model: predict the cortical response an image will produce.

The decoding work in this project runs brain -> image. This runs the other way,
image -> brain, and it is the part that is useful to somebody else. If you are
designing a visual experiment and you want to know whether your stimulus will
drive narrowband gamma before you book scanner or clinic time, this answers that
from five numbers you can measure off the image itself.

It is a plain linear model fitted on the real recordings, cross-validated by
image so a test image never appears in training. The site reports its accuracy
next to every prediction, because a forward model with an r of 0.5 should not be
presented the same way as one with an r of 0.9.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from scipy import stats

# Measured off the image. All five are computable in a browser from a 2-D FFT.
FEATURES = ["log_spatial_freq", "orient_concentration", "noise_exponent",
            "rms_contrast", "mean_lum"]

# Averaged over the most visually driven electrodes.
TARGETS = ["broadband", "gamma_amp", "gamma_peak_hz"]

TARGET_LABELS = {
    "broadband": "Broadband gamma (70-200 Hz)",
    "gamma_amp": "Narrowband gamma strength",
    "gamma_peak_hz": "Gamma peak frequency (Hz)",
}


def build_design(params_df, stim_id: np.ndarray) -> np.ndarray:
    rows = params_df.loc[stim_id]
    sf = np.clip(rows["spatial_freq_cpd"].to_numpy(float), 1e-3, None)
    X = np.stack([
        np.log10(sf),
        rows["orient_concentration"].to_numpy(float),
        rows["noise_exponent"].to_numpy(float),
        rows["rms_contrast"].to_numpy(float),
        rows["mean_lum"].to_numpy(float),
    ], axis=1)
    return np.nan_to_num(X, nan=0.0)


def build_targets(prep, n_top: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Response averaged over the n_top most visually driven electrodes."""
    bb = prep["feat_broadband"]
    sel = np.argsort(-bb.mean(0))[:n_top]
    Y = np.stack([
        bb[:, sel].mean(1),
        prep["feat_gamma_amp"][:, sel].mean(1),
        prep["feat_gamma_peak_hz"][:, sel].mean(1),
    ], axis=1)
    return Y, sel


def fit(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, n_folds: int = 6,
        n_perm: int = 1000, seed: int = 0) -> dict:
    """Cross-validated fit. Returns coefficients plus honest accuracy per target."""
    alphas = np.logspace(-3, 6, 40)
    P = np.zeros_like(Y)
    for tr, te in GroupKFold(n_splits=n_folds).split(X, groups=groups):
        assert not (set(groups[tr]) & set(groups[te])), "image leaked across the split"
        mx, sx = X[tr].mean(0), X[tr].std(0) + 1e-9
        my, sy = Y[tr].mean(0), Y[tr].std(0) + 1e-9
        m = RidgeCV(alphas=alphas, alpha_per_target=True).fit((X[tr] - mx) / sx,
                                                              (Y[tr] - my) / sy)
        P[te] = m.predict((X[te] - mx) / sx) * sy + my

    rng = np.random.default_rng(seed)
    acc = {}
    for i, t in enumerate(TARGETS):
        r = float(stats.pearsonr(Y[:, i], P[:, i]).statistic)
        null = np.array([stats.pearsonr(rng.permutation(Y[:, i]), P[:, i]).statistic
                         for _ in range(n_perm)])
        acc[t] = {"r": r,
                  "p": float(((np.abs(null) >= abs(r)).sum() + 1) / (n_perm + 1)),
                  "rmse": float(np.sqrt(np.mean((Y[:, i] - P[:, i]) ** 2)))}

    # Final model on everything, for export.
    mx, sx = X.mean(0), X.std(0) + 1e-9
    my, sy = Y.mean(0), Y.std(0) + 1e-9
    full = RidgeCV(alphas=alphas, alpha_per_target=True).fit((X - mx) / sx, (Y - my) / sy)

    return {
        "features": FEATURES, "targets": TARGETS, "target_labels": TARGET_LABELS,
        "x_mean": mx.tolist(), "x_std": sx.tolist(),
        "y_mean": my.tolist(), "y_std": sy.tolist(),
        "coef": np.atleast_2d(full.coef_).tolist(),
        "intercept": np.atleast_1d(full.intercept_).tolist(),
        "accuracy": acc, "n_trials": int(len(X)), "n_folds": n_folds,
        "predicted": P.tolist(), "observed": Y.tolist(),
    }
