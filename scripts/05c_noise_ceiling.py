#!/usr/bin/env python
"""Step 5c -- an interpretable noise ceiling, merged into the alignment results.

Subject 2 saw every image twice, which makes split-half reliability definable.
Two versions are reported because they answer different questions:

  physiological  reliability of broadband gamma (and narrowband gamma) on the
                 most visually driven electrodes. This is the quantity a decoder
                 actually exploits, and it is directly interpretable.
  full-feature   unweighted pattern correlation across the entire feature vector.
                 Reported for completeness, but it is pulled toward zero by the
                 many uninformative dimensions that a fitted decoder simply
                 downweights -- so it understates what is achievable and should
                 not be read as "the" ceiling.
"""
from __future__ import annotations

import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy import stats

from neurolink.align.evaluate import bootstrap_ci
from neurolink.paths import get_paths, load_config

N_TOP = 10


def main() -> None:
    cfg = load_config(); paths = get_paths()
    sub = "sub-02"                      # the only subject with repeated presentations
    runs = cfg["subjects"][sub]["runs"]
    if len(runs) < 2:
        print("no repeated runs; nothing to do"); return

    a = np.load(paths.cache / f"prep_{sub}_run-{runs[0]}.npz", allow_pickle=True)
    b = np.load(paths.cache / f"prep_{sub}_run-{runs[1]}.npz", allow_pickle=True)
    ida, idb = a["stim_id"], b["stim_id"]
    common = np.intersect1d(ida, idb)
    sel = np.argsort(-a["feat_broadband"].mean(0))[:N_TOP]

    out = {"subject": sub, "n_stimuli": int(len(common)), "n_electrodes": int(N_TOP),
           "definition": ("Pearson r between run-01 and run-02 single-trial responses "
                          "to the same image, on the most visually driven electrodes"),
           "features": {}}

    for feat in ["feat_broadband", "feat_gamma_amp", "feat_alpha"]:
        A = np.stack([a[feat][ida == s].mean(0) for s in common])
        B = np.stack([b[feat][idb == s].mean(0) for s in common])
        x, y = A[:, sel].mean(1), B[:, sel].mean(1)
        r = stats.pearsonr(x, y)
        per = np.array([stats.pearsonr(A[:, c], B[:, c]).statistic for c in sel])
        lo, hi = bootstrap_ci(np.stack([x, y], 1),
                             stat=lambda v: stats.pearsonr(v[:, 0], v[:, 1]).statistic)
        out["features"][feat.replace("feat_", "")] = {
            "r": float(r.statistic), "p": float(r.pvalue),
            "ci": [lo, hi],
            "median_per_electrode_r": float(np.median(per)),
            "max_per_electrode_r": float(per.max()),
        }
        print(f"{feat:16s} r = {r.statistic:+.3f} [{lo:+.3f},{hi:+.3f}]  p = {r.pvalue:.2e}  "
              f"median per-electrode {np.median(per):+.3f}")

    dest = paths.cache / "results_align.json"
    R = json.load(open(dest))
    R["noise_ceiling_physiological"] = out
    R["noise_ceiling_note"] = (
        "The full-feature noise ceiling is an unweighted pattern correlation and is "
        "biased toward zero by uninformative dimensions; the physiological version is "
        "the interpretable one.")
    with open(dest, "w") as fh:
        json.dump(R, fh, indent=2, default=float)
    print(f"\n[merge] into {dest}")


if __name__ == "__main__":
    main()
