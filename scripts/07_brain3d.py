#!/usr/bin/env python
"""Step 7 -- per-electrode decoding contribution, rendered on the pial surface.

Contribution is measured the interpretable way: retrain the 7-way condition
decoder using one electrode's features at a time and record its cross-validated
accuracy. That is a direct statement about what each electrode knows, rather
than an artefact of a particular model's internal weights.
"""
from __future__ import annotations

import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.metrics import accuracy_score

from neurolink.align import ridge
from neurolink.align.dataset import load_subject
from neurolink.align.splits import novel_image_splits
from neurolink.bids_io import canon_ch, read_electrodes
from neurolink.paths import get_paths, load_config
from neurolink.viz import brain3d

BLANK = 8


def main() -> None:
    cfg = load_config(); paths = get_paths()
    out = {}

    for sub, meta in cfg["subjects"].items():
        sd = load_subject(sub, meta["runs"], paths)
        prep = np.load(paths.cache / f"prep_{sub}_run-{meta['runs'][0]}.npz",
                       allow_pickle=True)
        feat_names = [str(f) for f in prep["feat_names"]]
        n_ch = len(sd.ch_names); n_feat = len(feat_names)

        keep = sd.trial_type != BLANK
        X = sd.feats["spectral"][keep]; sid = sd.stim_id[keep]; tt = sd.trial_type[keep]
        assert X.shape[1] == n_ch * n_feat, (X.shape, n_ch, n_feat)
        Xr = X.reshape(len(X), n_feat, n_ch)

        accs = np.zeros(n_ch)
        for c in range(n_ch):
            Xc = Xr[:, :, c]
            pred = np.zeros(len(tt), dtype=int)
            for sp in novel_image_splits(sid, cfg["align"]["n_folds"]):
                pred[sp.test], _, _ = ridge.fit_predict_classes(
                    Xc[sp.train], tt[sp.train], Xc[sp.test], cfg)
            accs[c] = accuracy_score(tt, pred)

        elec = read_electrodes(paths, sub).set_index("idx")
        coords = np.stack([elec.loc[canon_ch(c), ["x", "y", "z"]].to_numpy(float)
                           for c in sd.ch_names])
        hemi = meta["hemi"]
        files = brain3d.render(paths.pial_gii(sub, hemi), coords, accs, hemi,
                               str(paths.figures / f"brain_{sub}"),
                               title="7-way acc")
        order = np.argsort(-accs)
        print(f"[{sub}] single-electrode 7-way accuracy: max={accs.max():.3f} "
              f"median={np.median(accs):.3f} chance=0.143")
        print("  best electrodes: " +
              ", ".join(f"{sd.ch_names[i]}({accs[i]:.2f})" for i in order[:8]))
        print("  rendered: " + ", ".join(pathlib.Path(f).name for f in files))

        out[sub] = {
            "ch_names": sd.ch_names,
            "coords_mm": coords.tolist(),
            "single_electrode_acc": accs.tolist(),
            "hemi": hemi,
            "views": [pathlib.Path(f).name for f in files],
            "chance": 1 / 7,
        }

    with open(paths.cache / "results_electrodes.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[save] {paths.cache/'results_electrodes.json'}")


if __name__ == "__main__":
    main()
