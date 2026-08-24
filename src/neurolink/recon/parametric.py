"""Parametric reconstruction: decode the stimulus's generative parameters, then
re-render the image from those decoded numbers.

Why this rather than a diffusion decoder. With 210 trials of gratings and 1/f
noise there is no way to learn a general image prior. But this stimulus set is
fully described by a handful of parameters, all of which we measured from the
pixels in step 1. Decoding those parameters and re-rendering gives a genuine
side-by-side comparison in which every pixel is traceable to a decoded number.

What is and is not recoverable, established from the stimulus set itself:
  spatial frequency   4 values, 0.16-1.28 cpd, varies across conditions
  noise exponent      3 values, k/f^0 k/f^2 k/f^4
  RMS contrast        gratings ~0.49, noise 0.05-0.20
  orientation         CONSTANT (every grating is vertical) -- nothing to decode
  spatial phase       the ONLY thing distinguishing exemplars within a grating
                      condition, so exemplar-level identification stands or falls
                      on whether phase is decodable at all
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..stimuli.params import render_grating, render_noise

# Continuous targets decoded by ridge regression.
SCALAR_TARGETS = ["log_spatial_freq", "noise_exponent", "rms_contrast",
                  "orient_concentration"]
CIRCULAR_TARGETS = ["phase_rad"]


def build_targets(params_df, stim_id: np.ndarray) -> tuple[np.ndarray, list[str], dict]:
    """Assemble the per-trial regression targets.

    Circular quantities are represented as (sin, cos) pairs so a linear model can
    fit them without a wraparound discontinuity.
    """
    rows = params_df.loc[stim_id]
    cols, mats = [], []
    sf = rows["spatial_freq_cpd"].to_numpy(float)
    mats.append(np.log10(np.clip(sf, 1e-3, None))[:, None]); cols.append("log_spatial_freq")
    for c in ["noise_exponent", "rms_contrast", "orient_concentration"]:
        mats.append(rows[c].to_numpy(float)[:, None]); cols.append(c)
    ph = rows["phase_rad"].to_numpy(float)
    mats.append(np.stack([np.sin(ph), np.cos(ph)], axis=1))
    cols += ["phase_sin", "phase_cos"]
    Y = np.concatenate(mats, axis=1)
    Y = np.nan_to_num(Y, nan=0.0)
    return Y, cols, {"phase_true": ph}


def circular_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Jammalamadaka circular correlation coefficient."""
    a_ = a - np.angle(np.exp(1j * a).mean())
    b_ = b - np.angle(np.exp(1j * b).mean())
    num = np.sum(np.sin(a_) * np.sin(b_))
    den = np.sqrt(np.sum(np.sin(a_) ** 2) * np.sum(np.sin(b_) ** 2))
    return float(num / (den + 1e-12))


def score_parameters(Ytrue: np.ndarray, Ypred: np.ndarray, cols: list[str],
                     phase_true: np.ndarray, mask: np.ndarray | None = None,
                     n_perm: int = 1000, seed: int = 0) -> dict:
    """Per-parameter decoding accuracy with permutation p-values."""
    rng = np.random.default_rng(seed)
    m = np.ones(len(Ytrue), bool) if mask is None else mask
    out = {}
    for i, c in enumerate(cols):
        if c.startswith("phase_"):
            continue
        t, p = Ytrue[m, i], Ypred[m, i]
        if np.std(t) < 1e-9:
            out[c] = {"r": None, "note": "constant across stimuli -- nothing to decode"}
            continue
        r = float(stats.pearsonr(t, p).statistic)
        null = np.array([stats.pearsonr(rng.permutation(t), p).statistic
                         for _ in range(n_perm)])
        out[c] = {"r": r, "p": float(((np.abs(null) >= abs(r)).sum() + 1) / (n_perm + 1)),
                  "n": int(m.sum())}

    si, ci = cols.index("phase_sin"), cols.index("phase_cos")
    pred_ph = np.arctan2(Ypred[m, si], Ypred[m, ci])
    r = circular_corr(phase_true[m], pred_ph)
    null = np.array([circular_corr(rng.permutation(phase_true[m]), pred_ph)
                     for _ in range(n_perm)])
    out["phase_rad"] = {"circular_r": r,
                        "p": float(((np.abs(null) >= abs(r)).sum() + 1) / (n_perm + 1)),
                        "n": int(m.sum())}
    return out


def reconstruct(pred_row: np.ndarray, cols: list[str], cond_posterior: np.ndarray,
                deg: float, size: int = 256, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Render one trial's reconstruction from its decoded parameters.

    The decoded 7-way condition posterior decides grating vs noise; the
    regression outputs supply the continuous parameters.
    """
    g = dict(zip(cols, pred_row))
    cond = int(np.argmax(cond_posterior)) + 1
    is_grating = cond >= 4
    contrast = float(np.clip(g.get("rms_contrast", 0.2), 0.02, 0.5))
    phase = float(np.arctan2(g.get("phase_sin", 0.0), g.get("phase_cos", 1.0)))

    if is_grating:
        sf = float(np.clip(10 ** g.get("log_spatial_freq", -0.8), 0.05, 3.0))
        img = render_grating(sf, 0.0, phase, contrast, deg, size=size, square=True)
        info = {"kind": "grating", "spatial_freq_cpd": sf, "phase_rad": phase,
                "contrast": contrast, "condition": cond}
    else:
        alpha = float(np.clip(g.get("noise_exponent", 2.0), -0.5, 4.5))
        img = render_noise(alpha, contrast, size=size, seed=seed)
        info = {"kind": "noise", "noise_exponent": alpha, "contrast": contrast,
                "condition": cond}
    return img, info
