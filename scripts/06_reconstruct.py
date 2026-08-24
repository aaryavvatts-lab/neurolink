#!/usr/bin/env python
"""Step 6 -- decode stimulus parameters and re-render what the subject saw."""
from __future__ import annotations

import sys, json, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from neurolink.align import ridge
from neurolink.align.dataset import load_image_spaces, load_subject
from neurolink.align.splits import novel_image_splits
from neurolink.paths import get_paths, load_config
from neurolink.recon.parametric import build_targets, reconstruct, score_parameters

BLANK = 8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="spectral")
    args = ap.parse_args()

    cfg = load_config(); paths = get_paths()
    spaces, space_ids, params = load_image_spaces(paths)
    pdf = pd.DataFrame(params["X"], columns=params["cols"], index=space_ids)
    deg = cfg["stimuli"]["visual_angle_deg"]
    out = {"encoder": args.encoder, "subjects": {}}

    for sub, meta in cfg["subjects"].items():
        sd = load_subject(sub, meta["runs"], paths)
        if args.encoder not in sd.feats:
            print(f"[{sub}] encoder {args.encoder} unavailable; skipping"); continue
        keep = sd.trial_type != BLANK
        X = sd.feats[args.encoder][keep]; sid = sd.stim_id[keep]; tt = sd.trial_type[keep]

        Y, cols, extra = build_targets(pdf, sid)
        P = np.zeros_like(Y)
        post = np.zeros((len(tt), 7))
        for sp in novel_image_splits(sid, cfg["align"]["n_folds"]):
            sp.check(sid)
            P[sp.test], _ = ridge.fit_predict(X[sp.train], Y[sp.train], X[sp.test], cfg)
            _, pr, _ = ridge.fit_predict_classes(X[sp.train], tt[sp.train], X[sp.test], cfg)
            post[sp.test] = pr

        grating = tt >= 4
        scores = {
            "all": score_parameters(Y, P, cols, extra["phase_true"]),
            "gratings_only": score_parameters(Y, P, cols, extra["phase_true"], mask=grating),
            "noise_only": score_parameters(Y, P, cols, extra["phase_true"], mask=~grating),
        }
        print(f"\n[{sub}] parameter decoding ({args.encoder}, novel-image split)")
        for scope, sc in scores.items():
            bits = []
            for k, v in sc.items():
                if v.get("r") is not None and "r" in v:
                    bits.append(f"{k}: r={v['r']:+.3f} p={v['p']:.3f}")
                elif "circular_r" in v:
                    bits.append(f"{k}: circ_r={v['circular_r']:+.3f} p={v['p']:.3f}")
                else:
                    bits.append(f"{k}: {v.get('note','n/a')}")
            print(f"  {scope:15s} " + " | ".join(bits))

        # Render a gallery: best, median and worst trials by parameter error.
        err = np.abs(P[:, cols.index("log_spatial_freq")] - Y[:, cols.index("log_spatial_freq")])
        order = np.argsort(err)
        picks = {"best": order[:6], "median": order[len(order)//2 - 3:len(order)//2 + 3],
                 "worst": order[-6:]}
        gallery = {}
        for name, idx in picks.items():
            gallery[name] = []
            for i in idx:
                img, info = reconstruct(P[i], cols, post[i], deg, size=256, seed=int(sid[i]))
                gallery[name].append({
                    "trial": int(i), "stim_id": int(sid[i]),
                    "true_condition": int(tt[i]), "info": info,
                    "recon_png": f"recon_{sub}_{int(sid[i])}.png",
                })
                from PIL import Image
                Image.fromarray((img * 255).astype(np.uint8)).save(
                    paths.figures / f"recon_{sub}_{int(sid[i])}.png")

        out["subjects"][sub] = {"scores": scores, "gallery": gallery,
                                "cols": cols, "n_trials": int(len(tt))}
        np.savez_compressed(paths.cache / f"recon_{sub}.npz",
                            Y=Y, P=P, post=post, cols=np.array(cols, dtype=object),
                            stim_id=sid, trial_type=tt,
                            phase_true=extra["phase_true"])

    with open(paths.cache / "results_recon.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\n[save] {paths.cache/'results_recon.json'}")


if __name__ == "__main__":
    main()
