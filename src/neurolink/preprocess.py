"""Bad-channel detection, line-noise removal, re-referencing and epoching.

`channels.tsv` in this dataset marks every channel "good", so rejection here is
entirely data-driven.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mne
import numpy as np
import pandas as pd
from scipy import signal as sps

from .bids_io import BLANK_TRIAL_TYPE, RunData, canon_ch, curated_bads
from .paths import get_paths

mne.set_log_level("ERROR")


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return (x - med) / (1.4826 * mad + 1e-12)


def detect_bad_channels(raw: mne.io.BaseRaw, cfg: dict,
                        curated: list[int] | None = None) -> tuple[list[str], pd.DataFrame]:
    """Flag flat, high-variance, line-noise-dominated and epileptiform channels.

    The returned set is the union of the dataset's own curated `status == "bad"`
    list (authoritative -- clinicians marked epileptogenic and broken electrodes)
    and this data-driven pass, which catches anything the curation missed.
    """
    bc = cfg["preprocess"]["bad_channel"]
    sf = float(raw.info["sfreq"])
    data = raw.get_data()                                  # (n_ch, n_times), volts
    ch = np.asarray(raw.ch_names)

    std = data.std(axis=1)
    log_var = np.log(std ** 2 + 1e-30)
    var_z = _robust_z(log_var)
    flat = std < bc["flat_thresh"] * np.median(std)

    # Line-noise dominance: 60 Hz power relative to neighbouring broadband power.
    nper = int(min(4 * sf, data.shape[1]))
    freqs, psd = sps.welch(data, fs=sf, nperseg=nper, axis=-1)
    line = cfg["preprocess"]["line_freq"]
    m_line = (freqs > line - 1.5) & (freqs < line + 1.5)
    m_ref = ((freqs > 45) & (freqs < 55)) | ((freqs > 65) & (freqs < 75))
    line_ratio = psd[:, m_line].mean(1) / (psd[:, m_ref].mean(1) + 1e-30)
    line_z = _robust_z(np.log(line_ratio + 1e-30))

    # Epileptiform / artefactual excursions: largest sample amplitude in MAD units.
    # An absolute cutoff here is wrong -- over 233 s of heavy-tailed neural data a
    # peak of 10-18 MAD is ordinary. Score each channel against the *array's own*
    # distribution instead, which isolates genuinely broken channels.
    mad = np.median(np.abs(data - np.median(data, axis=1, keepdims=True)), axis=1)
    spike = np.abs(data).max(axis=1) / (1.4826 * mad + 1e-30)
    spike_z = _robust_z(np.log(spike + 1e-30))

    bad = (
        flat
        | (var_z > bc["var_z_thresh"])
        | (var_z < -bc["var_z_thresh"])
        | (line_z > bc["line_ratio_thresh"])
        | (spike_z > bc["spike_z_thresh"])
    )
    detected = bad.copy()
    cur = np.zeros_like(bad)
    if curated:
        cur = np.array([canon_ch(c) in set(curated) for c in ch])
        bad = bad | cur

    report = pd.DataFrame({
        "ch_name": ch, "std": std, "var_z": var_z, "flat": flat,
        "line_ratio": line_ratio, "line_z": line_z,
        "spike_mad": spike, "spike_z": spike_z,
        "curated_bad": cur, "detected_bad": detected, "bad": bad,
    })
    return list(ch[bad]), report


def clean_raw(run: RunData, cfg: dict) -> tuple[mne.io.BaseRaw, pd.DataFrame]:
    """Notch out line harmonics, drop bad channels, common-average reference."""
    pp = cfg["preprocess"]
    raw = run.raw.copy().load_data()
    sf = float(raw.info["sfreq"])

    cur = curated_bads(get_paths(), run.sub, run.run)
    bads, report = detect_bad_channels(raw, cfg, curated=cur)
    raw.info["bads"] = bads

    harmonics = np.arange(1, pp["notch_harmonics"] + 1) * pp["line_freq"]
    harmonics = harmonics[harmonics < sf / 2 - pp["notch_width"]]
    if len(harmonics):
        raw.notch_filter(freqs=harmonics, notch_widths=pp["notch_width"],
                         picks="all", verbose="ERROR")

    if pp["reference"] == "car":
        # MNE's average reference automatically excludes info['bads'].
        raw.set_eeg_reference("average", ch_type="ecog", projection=False, verbose="ERROR")
    return raw, report


def epoch_run(raw: mne.io.BaseRaw, run: RunData, cfg: dict,
              blanks: bool = False) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, list[str]]:
    """Cut event-locked epochs.

    With `blanks=True` this locks to the interleaved gray periods instead, which
    supply the per-electrode baseline spectrum the Hermes decomposition needs.

    Returns (data (n_trials, n_good_ch, n_times), trial table, time vector, channel names).
    """
    pp = cfg["preprocess"]
    sf = float(raw.info["sfreq"])
    ev = (run.events[run.events.trial_type == BLANK_TRIAL_TYPE].reset_index(drop=True)
          if blanks else run.stim_events)
    good = [c for c in raw.ch_names if c not in raw.info["bads"]]
    picks = mne.pick_channels(raw.ch_names, include=good, ordered=True)

    n_pre = int(round(-pp["epoch_tmin"] * sf))
    n_post = int(round(pp["epoch_tmax"] * sf))
    n_times = n_pre + n_post
    data_all = raw.get_data(picks=picks)
    n_samp = data_all.shape[1]

    out, keep = [], []
    for i, onset in enumerate(ev["onset"].to_numpy()):
        s0 = int(round(onset * sf)) - n_pre
        s1 = s0 + n_times
        if s0 < 0 or s1 > n_samp:
            continue
        out.append(data_all[:, s0:s1])
        keep.append(i)

    X = np.stack(out).astype(np.float64)
    tab = ev.iloc[keep].reset_index(drop=True)
    tab["sub"] = run.sub
    tab["run"] = run.run
    times = (np.arange(n_times) - n_pre) / sf
    return X, tab, times, good


@dataclass
class PreppedRun:
    sub: str
    run: str
    sfreq: float
    X: np.ndarray                    # (n_trials, n_ch, n_times) volts
    times: np.ndarray
    trials: pd.DataFrame
    ch_names: list[str]
    X_blank: np.ndarray | None = None   # (n_blanks, n_ch, n_times), gray-screen periods
    bads: list[str] = field(default_factory=list)
    raw: mne.io.BaseRaw | None = None   # kept only when needed (Brain TV)

    @property
    def key(self) -> str:
        return f"{self.sub}_run-{self.run}"


def prepare_run(run: RunData, cfg: dict, keep_raw: bool = False) -> tuple[PreppedRun, pd.DataFrame]:
    raw, report = clean_raw(run, cfg)
    X, tab, times, good = epoch_run(raw, run, cfg)
    Xb, _, _, _ = epoch_run(raw, run, cfg, blanks=True)
    pr = PreppedRun(
        sub=run.sub, run=run.run, sfreq=float(raw.info["sfreq"]),
        X=X, times=times, trials=tab, ch_names=good, bads=list(raw.info["bads"]),
        X_blank=Xb, raw=raw if keep_raw else None,
    )
    return pr, report
