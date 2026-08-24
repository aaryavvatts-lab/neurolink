"""Encoder B -- off-the-shelf "brain foundation models", applied as intended.

These are the models the project brief calls for. Every one of them is pretrained
on *scalp EEG*:

  SignalJEPA  pretrained at 128 Hz on data bandpassed 0.5-40 Hz
  CBraMod     pretrained at 250 Hz (Nyquist 125 Hz)

ECoG visual responses live in broadband gamma at 70-200 Hz. Neither model can
represent that band at all -- not because it is small, but because the signal is
gone before the first layer. This module applies each model faithfully to its own
pretraining recipe so the comparison in Step 5 is fair rather than rigged, and
`bandwidth_accounting` quantifies exactly what is discarded.

A second, structural mismatch: SignalJEPA's convolutional front end cannot accept
a window shorter than 1.5 s, while the stimulus-evoked response here is 500 ms.
We give it -0.5 to +1.0 s, which in this alternating design is exactly
blank-stimulus-blank and so contains no neighbouring stimulus.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import signal as sps

from .ecogjepa import resample_to

SPECS = {
    "signal_jepa": dict(sfreq=128.0, band=(0.5, 40.0), window=(-0.5, 1.0),
                        checkpoint="braindecode/signal-jepa_without-chans"),
    "cbramod": dict(sfreq=250.0, band=(0.3, 124.0), window=(-0.2, 0.6),
                    checkpoint="braindecode/cbramod-pretrained"),
}


def _cut(data: np.ndarray, sf: float, onsets: np.ndarray,
         window: tuple[float, float]) -> np.ndarray:
    n0 = int(round(window[0] * sf)); n1 = int(round(window[1] * sf))
    n = n1 - n0
    out, ok = [], []
    for i, o in enumerate(onsets):
        s = int(round(o * sf)) + n0
        if s < 0 or s + n > data.shape[1]:
            continue
        out.append(data[:, s:s + n]); ok.append(i)
    return np.stack(out), np.array(ok)


def prepare_input(data: np.ndarray, sf: float, onsets: np.ndarray, spec: dict
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Bandpass, resample and epoch to match a model's pretraining recipe.

    `data` is (n_ch, n_times) in volts. Returns (n_trials, n_ch, n_times) scaled
    to microvolts, which is the unit these checkpoints were trained on.
    """
    lo, hi = spec["band"]
    hi = min(hi, sf / 2 * 0.95)
    # Second-order sections, not transfer-function form. At these sample rates the
    # low edge sits at a normalised frequency of ~2e-4, where a 4th-order Butterworth
    # in (b, a) form is numerically unstable -- it amplified the signal by ~10^105
    # instead of filtering it, which would have silently destroyed the inputs to the
    # pretrained models and made the comparison against them meaningless.
    sos = sps.butter(4, [lo / (sf / 2), hi / (sf / 2)], btype="band", output="sos")
    filt = sps.sosfiltfilt(sos, data, axis=-1)
    if not np.isfinite(filt).all():
        raise ValueError("bandpass produced non-finite output")
    rs = resample_to(filt, sf, spec["sfreq"])
    X, ok = _cut(rs, spec["sfreq"], onsets, spec["window"])
    Xu = X * 1e6                                     # volts -> microvolts
    if not np.isfinite(Xu).all():
        raise ValueError("non-finite input after resampling")
    return Xu.astype(np.float32), ok


def _chs_info(ch_names: list[str], coords_mm: np.ndarray | None) -> list[dict]:
    """Build the channel-info structure SignalJEPA needs (locations in metres)."""
    out = []
    for i, name in enumerate(ch_names):
        loc = np.zeros(12, dtype=np.float64)
        if coords_mm is not None and i < len(coords_mm):
            loc[:3] = np.asarray(coords_mm[i], dtype=np.float64) / 1000.0
        out.append({"ch_name": str(name), "loc": loc})
    return out


@torch.no_grad()
def embed(model_key: str, data: np.ndarray, sf: float, onsets: np.ndarray,
          ch_names: list[str], coords_mm: np.ndarray | None = None,
          device: str = "cpu", batch: int = 8) -> tuple[np.ndarray, dict]:
    """Extract per-trial embeddings from one pretrained model.

    Returns ((n_trials, n_ch * emb_dim), info). Raises on failure -- the caller
    decides whether a missing encoder is fatal.
    """
    from braindecode.models import CBraMod, SignalJEPA

    spec = SPECS[model_key]
    X, ok = prepare_input(data, sf, onsets, spec)
    n_tr, n_ch, n_t = X.shape

    if model_key == "signal_jepa":
        model = SignalJEPA.from_pretrained(
            spec["checkpoint"], chs_info=_chs_info(ch_names, coords_mm),
            n_times=n_t, sfreq=spec["sfreq"],
            channel_embedding="scratch", strict=False)
    else:
        model = CBraMod.from_pretrained(
            spec["checkpoint"], n_outputs=7, n_chans=n_ch,
            n_times=n_t, sfreq=spec["sfreq"])
    model = model.to(device).eval()

    feats = []
    for i in range(0, n_tr, batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        if model_key == "signal_jepa":
            # Take the pretrained convolutional feature encoder's output rather
            # than the full forward pass. The `without-chans` checkpoint ships no
            # channel-embedding weights (that is what its name means), so those are
            # randomly initialised, and they are added *after* the feature encoder.
            # Measured on this data, the full model's trial-to-trial variation is
            # 4e-5 of its mean and does not change when the input is rescaled 50x --
            # the random embedding swamps the signal path entirely. The feature
            # encoder alone gives a relative variance of ~1.9 and does track the
            # input, so it is the representation this checkpoint actually learned.
            f = model.feature_encoder(xb)
        else:
            out = model(xb, return_features=True)
            f = out["features"] if isinstance(out, dict) else out
        feats.append(f.reshape(f.shape[0], -1).float().cpu().numpy())
    E = np.concatenate(feats).astype(np.float32)
    if not np.isfinite(E).all():
        raise ValueError(f"{model_key} produced non-finite embeddings")

    info = {"model": model_key, "checkpoint": spec["checkpoint"],
            "readout": ("pretrained conv feature encoder"
                        if model_key == "signal_jepa" else "full forward, return_features"),
            "sfreq": spec["sfreq"], "band": spec["band"],
            "window": spec["window"], "n_times": n_t,
            "emb_per_trial": int(E.shape[1]),
            "n_params": int(sum(p.numel() for p in model.parameters()))}
    return E, {"info": info, "kept": ok}


def bandwidth_accounting(psd_stim: np.ndarray, psd_blank: np.ndarray,
                         freqs: np.ndarray, cutoffs=(40.0, 125.0, 200.0)) -> dict:
    """How much stimulus-evoked spectral change lives above each model's ceiling.

    Quantified as the total absolute log-power change from baseline, summed over
    frequency, split at each cutoff. This is the concrete cost of the EEG models'
    bandwidth limits on this data.
    """
    d = np.abs(np.log10(psd_stim.mean(axis=0) + 1e-30)
               - np.log10(psd_blank.mean(axis=0) + 1e-30))     # (n_ch, n_freq)
    band = (freqs >= 1.0) & (freqs <= 250.0)
    tot = d[:, band].sum()
    out = {}
    for c in cutoffs:
        m = band & (freqs <= c)
        out[f"frac_below_{c:g}Hz"] = float(d[:, m].sum() / tot)
    out["total_evoked_logpower_change"] = float(tot)
    return out
