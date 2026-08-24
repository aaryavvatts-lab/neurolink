"""Read the ds005953 BIDS iEEG dataset: raw BrainVision, events, channels, electrodes."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .paths import Paths, get_paths, load_config

mne.set_log_level("ERROR")

BLANK_TRIAL_TYPE = 8


@dataclass
class RunData:
    """One task run: continuous data plus its trial table."""
    sub: str
    run: str
    raw: mne.io.BaseRaw
    events: pd.DataFrame          # all rows incl. blanks
    sfreq: float

    @property
    def stim_events(self) -> pd.DataFrame:
        """Stimulus trials only (drops the interleaved blank/gray periods)."""
        return self.events[self.events.trial_type != BLANK_TRIAL_TYPE].reset_index(drop=True)

    @property
    def key(self) -> str:
        return f"{self.sub}_run-{self.run}"


def read_events(paths: Paths, sub: str, run: str, ses: str = "01") -> pd.DataFrame:
    p = Path(str(paths.run_stem(sub, run, ses)) + "_events.tsv")
    df = pd.read_csv(p, sep="\t")
    df["trial_type"] = df["trial_type"].astype(int)
    df["stim_id"] = df["stim_file"].str.extract(r"stim_(\d+)\.png").astype(int)
    return df


def canon_ch(name) -> int:
    """Canonical electrode index from a channel/electrode name.

    The two subjects disagree on naming: sub-01 has `1..118` in channels.tsv but
    `iEEG1..iEEG118` in electrodes.tsv, while sub-02 uses `iEEG<N>` in both. Reduce
    every form to its integer index so channels and coordinates can be joined.
    """
    m = re.search(r"(\d+)$", str(name))
    if m is None:
        raise ValueError(f"cannot parse electrode index from {name!r}")
    return int(m.group(1))


def read_channels(paths: Paths, sub: str, run: str, ses: str = "01") -> pd.DataFrame:
    p = Path(str(paths.run_stem(sub, run, ses)) + "_channels.tsv")
    df = pd.read_csv(p, sep="\t", dtype={"name": str})
    df["idx"] = df["name"].map(canon_ch)
    if "status" not in df.columns:
        df["status"] = "good"
    return df


def curated_bads(paths: Paths, sub: str, run: str, ses: str = "01") -> list[int]:
    """Electrode indices the dataset curators marked `status == "bad"`.

    sub-01 flags 7 channels; sub-02 flags 35 (including iEEG1, its recording
    reference, which is therefore identically zero).
    """
    ch = read_channels(paths, sub, run, ses)
    return sorted(ch.loc[ch["status"].astype(str).str.lower() == "bad", "idx"].tolist())


def read_electrodes(paths: Paths, sub: str, ses: str = "01") -> pd.DataFrame:
    """Electrode positions in ACPC millimetres."""
    df = pd.read_csv(paths.electrodes_tsv(sub, ses), sep="\t")
    df["idx"] = df["name"].map(canon_ch)
    return df


def read_run(paths: Paths, sub: str, run: str, ses: str = "01",
             preload: bool = True) -> RunData:
    vhdr = Path(str(paths.run_stem(sub, run, ses)) + "_ieeg.vhdr")
    raw = mne.io.read_raw_brainvision(vhdr, preload=preload, verbose="ERROR")
    # BrainVision here carries no channel types; everything in this dataset is ECoG.
    raw.set_channel_types({ch: "ecog" for ch in raw.ch_names})
    events = read_events(paths, sub, run, ses)
    return RunData(sub=sub, run=run, raw=raw, events=events, sfreq=float(raw.info["sfreq"]))


def iter_runs(cfg: dict | None = None, paths: Paths | None = None):
    cfg = cfg or load_config()
    paths = paths or get_paths()
    for sub, meta in cfg["subjects"].items():
        for run in meta["runs"]:
            yield read_run(paths, sub, run)


def stimulus_table(paths: Paths | None = None, cfg: dict | None = None) -> pd.DataFrame:
    """One row per unique stimulus image, with its condition label.

    Both subjects saw the identical 211 images, so this table is subject-independent;
    we assert that rather than assume it.
    """
    cfg = cfg or load_config()
    paths = paths or get_paths()
    frames = []
    for sub, meta in cfg["subjects"].items():
        for run in meta["runs"]:
            ev = read_events(paths, sub, run)
            frames.append(ev[["stim_id", "stim_file", "trial_type"]])
    allev = pd.concat(frames, ignore_index=True)
    tab = allev.drop_duplicates("stim_id").sort_values("stim_id").reset_index(drop=True)

    # A stim_id must map to exactly one trial_type across every subject and run.
    conflict = allev.groupby("stim_id")["trial_type"].nunique()
    assert (conflict == 1).all(), f"stim_id maps to >1 condition: {conflict[conflict>1]}"

    n_expected = cfg["stimuli"]["n_images"]
    assert len(tab) == n_expected, f"expected {n_expected} stimuli, found {len(tab)}"
    tab["is_blank"] = tab.trial_type == BLANK_TRIAL_TYPE
    tab["path"] = tab.stim_file.map(lambda f: str(paths.stimuli / f))
    return tab
