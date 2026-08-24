#!/usr/bin/env python
"""Step 2 -- clean, epoch and spectrally characterise every run.

Writes one outputs/cache/prep_<sub>_run-<run>.npz per run containing stimulus and
blank epochs, the trial table, channel names and the Encoder-A feature set.
"""
from __future__ import annotations

import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from neurolink.bids_io import read_run
from neurolink.neural.spectral import extract
from neurolink.paths import get_paths, load_config
from neurolink.preprocess import prepare_run


def main() -> None:
    cfg = load_config()
    paths = get_paths()
    summary = []

    for sub, meta in cfg["subjects"].items():
        for run in meta["runs"]:
            t = time.time()
            pr, report = prepare_run(read_run(paths, sub, run), cfg)
            sf = extract(pr, cfg)

            dest = paths.cache / f"prep_{sub}_run-{run}.npz"
            np.savez_compressed(
                dest,
                X=pr.X.astype(np.float32),
                X_blank=pr.X_blank.astype(np.float32),
                times=pr.times,
                sfreq=pr.sfreq,
                ch_names=np.array(pr.ch_names, dtype=object),
                bads=np.array(pr.bads, dtype=object),
                trial_type=pr.trials.trial_type.to_numpy(),
                stim_id=pr.trials.stim_id.to_numpy(),
                onset=pr.trials.onset.to_numpy(),
                feat_X=sf.X.astype(np.float32),
                feat_names=np.array(sf.names, dtype=object),
                **{f"feat_{k}": v.astype(np.float32) for k, v in sf.per_feature.items()},
            )
            n_cur = int(report.curated_bad.sum()); n_det = int(report.detected_bad.sum())
            print(f"[{sub} run-{run}] {pr.X.shape} sf={pr.sfreq:.0f}Hz "
                  f"good={len(pr.ch_names)}/{len(report)} "
                  f"(curated_bad={n_cur} detected={n_det}) "
                  f"feat={sf.X.shape} {time.time()-t:.1f}s -> {dest.stat().st_size/1e6:.0f}MB")
            summary.append((sub, run, len(pr.ch_names), len(report)))

    print("\n[summary]")
    for sub, run, g, n in summary:
        print(f"  {sub} run-{run}: {g} good of {n} electrodes")


if __name__ == "__main__":
    main()
