#!/usr/bin/env python
"""Step 5b -- ridge regression vs the CLIP-style contrastive head.

The brief calls for contrastive alignment of the two latent spaces. It is run
here on the same folds, same features and same metrics as ridge, so the
comparison is like-for-like. At ~175 training trials a two-tower MLP has far
more capacity than the data supports; this measures how much that costs.
"""
from __future__ import annotations

import sys, json, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from neurolink.align import contrastive, evaluate as ev, ridge
from neurolink.align.dataset import (ENCODER_LABELS, image_targets,
                                     load_image_spaces, load_subject)
from neurolink.align.splits import novel_image_splits
from neurolink.paths import get_paths, load_config

BLANK = 8
SPACE = "dino518"


def main() -> None:
    cfg = load_config(); paths = get_paths()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    spaces, space_ids, _ = load_image_spaces(paths)
    out = {"space": SPACE, "subjects": {}}

    for sub, meta in cfg["subjects"].items():
        sd = load_subject(sub, meta["runs"], paths, verbose=False)
        keep = sd.trial_type != BLANK
        sid = sd.stim_id[keep]
        Y = image_targets(spaces[SPACE], space_ids, sid)
        cand_mask = np.isin(space_ids, np.unique(sid))
        cand = spaces[SPACE][cand_mask]
        pos = {int(s): i for i, s in enumerate(space_ids[cand_mask])}
        true_idx = np.array([pos[int(s)] for s in sid])
        res = {}

        for enc in sd.available():
            X = sd.feats[enc][keep]
            if not np.isfinite(X).all():
                continue
            row = {"label": ENCODER_LABELS[enc]}

            t = time.time()
            Pr = np.zeros_like(Y)
            for sp in novel_image_splits(sid, cfg["align"]["n_folds"]):
                sp.check(sid)
                Pr[sp.test], _ = ridge.fit_predict(X[sp.train], Y[sp.train], X[sp.test], cfg)
            row["ridge"] = {"two_way": ev.two_way_identification(Pr, Y),
                            "top5": ev.retrieval(Pr, cand, true_idx)["top5"],
                            "seconds": round(time.time() - t, 1)}

            # Contrastive: predictions live in the learned shared space, so both
            # sides must be projected before any metric is computed.
            t = time.time()
            tw, t5 = [], []
            for sp in novel_image_splits(sid, cfg["align"]["n_folds"]):
                sp.check(sid)
                pred, proj, _ = contrastive.fit_predict(
                    X[sp.train], Y[sp.train], X[sp.test], cfg,
                    groups_tr=sid[sp.train], device=dev)
                tw.append(ev.two_way_identification(pred, proj(Y[sp.test])))
                t5.append(ev.retrieval(pred, proj(cand), true_idx[sp.test])["top5"])
            row["contrastive"] = {"two_way": float(np.mean(tw)),
                                  "top5": float(np.mean(t5)),
                                  "seconds": round(time.time() - t, 1)}
            res[enc] = row
            print(f"[{sub}] {ENCODER_LABELS[enc]:26s} "
                  f"ridge 2-way={row['ridge']['two_way']:.3f} top5={row['ridge']['top5']:.3f} "
                  f"| contrastive 2-way={row['contrastive']['two_way']:.3f} "
                  f"top5={row['contrastive']['top5']:.3f}")
        out["subjects"][sub] = res

    with open(paths.cache / "results_contrastive.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\n[save] {paths.cache/'results_contrastive.json'}")


if __name__ == "__main__":
    main()
