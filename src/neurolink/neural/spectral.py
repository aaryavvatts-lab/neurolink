"""Encoder A -- canonical ECoG spectral features.

Implements the broadband / narrowband-gamma decomposition of Hermes, Petridou,
Kay & Winawer (eLife 2019, doi:10.7554/eLife.47035), which was developed on this
dataset's lineage. Fitted to log10 power spectra over 30-200 Hz:

    F(x) = (beta_bb - n * x) + beta_nb * G(x | mu, sigma),      x = log10(f)

`n`, the 1/f slope, is fixed per electrode from that electrode's own averaged
baseline (gray-screen) spectrum, letting the broadband and narrowband terms vary
independently per trial. The Gaussian is constrained to peak in 30-80 Hz.

This separation matters: gratings drive a narrowband gamma oscillation while
noise patterns drive broadband elevation without a spectral peak. A decoder given
only total gamma power cannot tell those apart; given these two numbers, it can.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

LOG10 = np.log(10.0)


def welch_psd(X: np.ndarray, sfreq: float, cfg: dict,
              window: tuple[float, float], times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD over a time window. X is (n_trials, n_ch, n_times)."""
    w = cfg["features"]["welch"]
    nper = int(round(w["win_s"] * sfreq))
    nover = int(round(w["overlap_s"] * sfreq))
    m = (times >= window[0]) & (times < window[1])
    seg = X[:, :, m]
    freqs, psd = sps.welch(seg, fs=sfreq, nperseg=nper, noverlap=nover,
                           window="hann", axis=-1, detrend="constant")
    return freqs, psd                                    # (n_trials, n_ch, n_freqs)


def fit_mask(freqs: np.ndarray, cfg: dict) -> np.ndarray:
    """Frequencies used for the log-log fit: 30-200 Hz minus line-noise notches."""
    f = cfg["features"]
    lo, hi = f["fit_range"]
    m = (freqs >= lo) & (freqs <= hi)
    hw = f["line_exclude_halfwidth"]
    for lf in f["line_exclude_hz"]:
        m &= ~((freqs > lf - hw) & (freqs < lf + hw))
    return m & (freqs > 0)


def _gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def baseline_slopes(psd_blank: np.ndarray, freqs: np.ndarray, cfg: dict
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Per-electrode 1/f slope `n` and intercept from the mean baseline spectrum.

    Returns (slope (n_ch,), intercept (n_ch,)). Sign convention matches the model:
    log10 power ~ intercept - n * log10(f), so `n` is positive for a falling spectrum.
    """
    m = fit_mask(freqs, cfg)
    x = np.log10(freqs[m])
    y = np.log10(psd_blank.mean(axis=0)[:, m] + 1e-30)   # (n_ch, n_fit)
    A = np.stack([np.ones_like(x), x], axis=1)           # [intercept, -n]
    coef, *_ = np.linalg.lstsq(A, y.T, rcond=None)       # (2, n_ch)
    intercept, neg_n = coef[0], coef[1]
    return -neg_n, intercept


def decompose(psd: np.ndarray, freqs: np.ndarray, slopes: np.ndarray, cfg: dict,
              n_mu: int = 33, n_sigma: int = 9) -> dict[str, np.ndarray]:
    """Fit the two-component model to every (trial, channel) spectrum.

    Given (mu, sigma) the model is *linear* in beta_bb and beta_nb, so instead of
    running one bounded nonlinear fit per spectrum (~23k of them) we sweep a grid
    over the constrained (mu, sigma) box and solve both betas in closed form for
    every spectrum at once, then keep each spectrum's best-fitting grid point.
    Same optimum, ~2 orders of magnitude faster, and it cannot land in a local
    minimum the way a seeded nonlinear fit can.

    Returns arrays of shape (n_trials, n_ch): `broadband` (beta_bb, log10 units),
    `gamma_amp` (beta_nb), `gamma_peak_hz` (10^mu), `gamma_sigma` and `resid`.
    """
    fcfg = cfg["features"]
    m = fit_mask(freqs, cfg)
    x = np.log10(freqs[m])
    n_tr, n_ch = psd.shape[:2]

    # Remove each electrode's fixed 1/f slope, then flatten to (N, n_fit).
    logp = np.log10(psd[:, :, m] + 1e-30)
    y = (logp + slopes[None, :, None] * x[None, None, :]).reshape(n_tr * n_ch, -1)

    mu_lo, mu_hi = np.log10(fcfg["narrowband_peak_hz"])
    s_lo, s_hi = fcfg["narrowband_sigma_log10"]
    mus = np.linspace(mu_lo, mu_hi, n_mu)
    sigmas = np.linspace(s_lo, s_hi, n_sigma)

    if not np.isfinite(y).all():
        raise ValueError("non-finite log-power entering the spectral fit")

    best_ss = np.full(y.shape[0], np.inf)
    best = np.zeros((y.shape[0], 4))          # b_bb, b_nb, mu, sigma
    yy = y.sum(axis=1)

    # Apple's Accelerate BLAS raises a spurious divide-by-zero FP flag inside
    # matmul on this platform; the inputs and outputs here are verified finite
    # above, so silence the flag rather than the (real) errors it would mask.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
      for sg in sigmas:
        for mu in mus:
            g = _gauss(x, mu, sg)
            A = np.stack([np.ones_like(x), g], axis=1)          # (n_fit, 2)
            # Closed-form 2x2 normal equations, shared across all spectra.
            AtA = A.T @ A
            Aty = np.stack([yy, y @ g], axis=0)                 # (2, N)
            beta = np.linalg.solve(AtA, Aty)                    # (2, N)
            pred = beta[0][:, None] + beta[1][:, None] * g[None, :]
            ss = ((pred - y) ** 2).sum(axis=1)
            upd = ss < best_ss
            if upd.any():
                best_ss[upd] = ss[upd]
                best[upd, 0] = beta[0][upd]
                best[upd, 1] = beta[1][upd]
                best[upd, 2] = mu
                best[upd, 3] = sg

    if not np.isfinite(best).all():
        raise ValueError("spectral fit produced non-finite parameters")

    shape = (n_tr, n_ch)
    return {
        "broadband": best[:, 0].reshape(shape),
        "gamma_amp": best[:, 1].reshape(shape),
        "gamma_peak_hz": (10 ** best[:, 2]).reshape(shape),
        "gamma_sigma": best[:, 3].reshape(shape),
        "resid": np.sqrt(best_ss / x.size).reshape(shape),
    }


def band_logpower(psd: np.ndarray, freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    m = (freqs >= band[0]) & (freqs <= band[1])
    return np.log10(psd[:, :, m].mean(axis=-1) + 1e-30)


@dataclass
class SpectralFeatures:
    names: list[str]                      # per-channel feature names
    X: np.ndarray                         # (n_trials, n_ch * n_feat)
    per_feature: dict[str, np.ndarray]    # name -> (n_trials, n_ch)
    ch_names: list[str]
    gamma_peak_hz: np.ndarray


def extract(pr, cfg: dict) -> SpectralFeatures:
    """Full Encoder-A feature set for one prepped run, baseline-normalised."""
    f = cfg["features"]
    sf, times = pr.sfreq, pr.times

    freqs, psd_stim = welch_psd(pr.X, sf, cfg, tuple(f["window"]), times)
    _, psd_blank = welch_psd(pr.X_blank, sf, cfg, tuple(f["window"]), times)

    slopes, _ = baseline_slopes(psd_blank, freqs, cfg)
    d_stim = decompose(psd_stim, freqs, slopes, cfg)
    d_blank = decompose(psd_blank, freqs, slopes, cfg)

    # Baseline-normalise against the mean over gray-screen periods.
    bb = d_stim["broadband"] - d_blank["broadband"].mean(axis=0, keepdims=True)
    ga = d_stim["gamma_amp"] - d_blank["gamma_amp"].mean(axis=0, keepdims=True)
    pk = d_stim["gamma_peak_hz"]

    feats: dict[str, np.ndarray] = {
        "broadband": bb,
        "gamma_amp": ga,
        "gamma_peak_hz": np.nan_to_num(pk, nan=float(np.nanmedian(pk))),
    }
    for name, band in f["bands"].items():
        s = band_logpower(psd_stim, freqs, tuple(band))
        b = band_logpower(psd_blank, freqs, tuple(band)).mean(axis=0, keepdims=True)
        feats[name] = s - b
    s = band_logpower(psd_stim, freqs, tuple(f["broadband_report"]))
    b = band_logpower(psd_blank, freqs, tuple(f["broadband_report"])).mean(axis=0, keepdims=True)
    feats["bb_power"] = s - b

    base_m = (times >= f["baseline_window"][0]) & (times < f["baseline_window"][1])
    base_mean = pr.X[:, :, base_m].mean(axis=-1, keepdims=True)
    for i, (t0, t1) in enumerate(f["erp_windows"]):
        m = (times >= t0) & (times < t1)
        feats[f"erp{i}"] = ((pr.X[:, :, m] - base_mean).mean(axis=-1)) * 1e6

    names = list(feats)
    X = np.concatenate([feats[n] for n in names], axis=1)
    return SpectralFeatures(names=names, X=X, per_feature=feats,
                            ch_names=pr.ch_names, gamma_peak_hz=pk)
