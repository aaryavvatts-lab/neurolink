"""Brain TV -- slide a window across the continuous recording and decode every frame.

Nothing about this is trial-locked: the decoder is handed a raw 500 ms window of
LFP every 50 ms and asked what is on the screen. The model is fitted on the first
half of the run and the video is rendered from the second half, so every frame
shown is out-of-sample.
"""
from __future__ import annotations

import numpy as np

from ..neural.spectral import (band_logpower, baseline_slopes, decompose, welch_psd)


def sliding_windows(data: np.ndarray, sfreq: float, window_s: float, hop_s: float,
                    t_start: float, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    """Cut overlapping windows. Returns (n_win, n_ch, n_samp) and window-centre times."""
    n = int(round(window_s * sfreq))
    hop = int(round(hop_s * sfreq))
    s0 = int(round(t_start * sfreq))
    s1 = min(int(round(t_end * sfreq)), data.shape[1])
    starts = np.arange(s0, s1 - n, hop)
    X = np.stack([data[:, s:s + n] for s in starts])
    centres = (starts + n / 2) / sfreq
    return X, centres


def window_features(X: np.ndarray, sfreq: float, cfg: dict, slopes: np.ndarray,
                    blank_ref: dict, feat_names: list[str],
                    chunk: int = 200) -> np.ndarray:
    """Recompute the Encoder-A feature vector for each sliding window.

    Uses the same per-electrode 1/f slopes and blank-period baselines that were
    estimated for the trial-locked analysis, so the features are on exactly the
    scale the decoder was trained on.
    """
    f = cfg["features"]
    n_win = X.shape[0]
    times = np.linspace(0, X.shape[2] / sfreq, X.shape[2], endpoint=False)
    out = []
    for i in range(0, n_win, chunk):
        xb = X[i:i + chunk].astype(np.float64)
        freqs, psd = welch_psd(xb, sfreq, cfg, (times[0], times[-1] + 1e-9), times)
        d = decompose(psd, freqs, slopes, cfg)
        feats = {
            "broadband": d["broadband"] - blank_ref["broadband"],
            "gamma_amp": d["gamma_amp"] - blank_ref["gamma_amp"],
            "gamma_peak_hz": d["gamma_peak_hz"],
        }
        for name, band in f["bands"].items():
            feats[name] = band_logpower(psd, freqs, tuple(band)) - blank_ref[name]
        feats["bb_power"] = (band_logpower(psd, freqs, tuple(f["broadband_report"]))
                             - blank_ref["bb_power"])
        # The ERP features are defined relative to a pre-stimulus baseline that does
        # not exist for a free-running window; use the window's own mean-centred
        # average, which is what the trial-locked version reduces to.
        cen = xb - xb.mean(axis=-1, keepdims=True)
        half = cen.shape[-1] // 2
        feats["erp0"] = cen[:, :, :half].mean(axis=-1) * 1e6
        feats["erp1"] = cen[:, :, half:].mean(axis=-1) * 1e6
        out.append(np.concatenate([feats[n] for n in feat_names], axis=1))
    return np.concatenate(out).astype(np.float32)


def blank_reference(psd_blank: np.ndarray, freqs: np.ndarray, slopes: np.ndarray,
                    cfg: dict) -> dict:
    """Per-electrode baseline values, averaged over the run's gray-screen periods."""
    f = cfg["features"]
    d = decompose(psd_blank, freqs, slopes, cfg)
    ref = {"broadband": d["broadband"].mean(axis=0, keepdims=True),
           "gamma_amp": d["gamma_amp"].mean(axis=0, keepdims=True)}
    for name, band in f["bands"].items():
        ref[name] = band_logpower(psd_blank, freqs, tuple(band)).mean(axis=0, keepdims=True)
    ref["bb_power"] = band_logpower(
        psd_blank, freqs, tuple(f["broadband_report"])).mean(axis=0, keepdims=True)
    return ref


def stimulus_at(times: np.ndarray, events, blank_id: int) -> np.ndarray:
    """Which stimulus id was actually on screen at each window centre."""
    onset = events["onset"].to_numpy()
    dur = events["duration"].to_numpy()
    sid = events["stim_id"].to_numpy()
    out = np.full(len(times), blank_id, dtype=int)
    for o, d, s in zip(onset, dur, sid):
        out[(times >= o) & (times < o + d)] = s
    return out
