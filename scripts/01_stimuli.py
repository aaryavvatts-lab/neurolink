#!/usr/bin/env python
"""Step 1 -- characterise the stimuli and build every image-side latent space.

Outputs outputs/cache/stimuli.npz holding, for all 211 images:
  * measured generative parameters (spatial frequency, orientation, phase,
    noise exponent, contrast) recovered from pixels
  * DINOv2 embeddings at each configured resolution
  * V1 Gabor-energy embeddings (the control model)
"""
from __future__ import annotations

import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from neurolink.bids_io import stimulus_table
from neurolink.paths import get_paths, load_config
from neurolink.stimuli import dino, v1model
from neurolink.stimuli.params import CONDITION_LABELS, measure_all


def main() -> None:
    cfg = load_config()
    paths = get_paths()
    tab = stimulus_table(paths, cfg)
    deg = cfg["stimuli"]["visual_angle_deg"]
    out = {}

    print(f"[stimuli] {len(tab)} images, {deg} deg visual angle")

    t = time.time()
    params = measure_all(tab, deg=deg)
    print(f"[params]  measured in {time.time()-t:.1f}s")
    for tt, lbl in CONDITION_LABELS.items():
        s = params[params.trial_type == tt]
        if not len(s):
            continue
        print(f"  cond {tt} {lbl:18s} sf={s.spatial_freq_cpd.mean():6.3f}cpd "
              f"alpha={s.noise_exponent.mean():+5.2f} conc={s.orient_concentration.mean():.3f} "
              f"rms={s.rms_contrast.mean():.3f}")
    out["param_cols"] = np.array(list(params.columns), dtype=object)
    out["params"] = params.to_numpy(dtype=np.float64)
    out["stim_id"] = params.index.to_numpy()
    out["trial_type"] = params.trial_type.to_numpy()

    order = [tab.set_index("stim_id").loc[i, "path"] for i in params.index]

    for size in cfg["vision"]["image_sizes"]:
        t = time.time()
        e = dino.embed_images(order, cfg["vision"]["dino_model"], image_size=size)
        for k, v in e.items():
            out[f"dino{size}_{k}"] = v
        print(f"[dino]    {size}px -> cls{e['cls'].shape} in {time.time()-t:.1f}s")

    t = time.time()
    gab, gnames = v1model.embed_images(order, cfg)
    out["gabor"] = gab
    out["gabor_names"] = np.array(gnames, dtype=object)
    print(f"[gabor]   {gab.shape} in {time.time()-t:.1f}s")

    dest = paths.cache / "stimuli.npz"
    np.savez_compressed(dest, **out)
    print(f"[save]    {dest}  ({dest.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
