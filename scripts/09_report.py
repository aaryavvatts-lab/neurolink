#!/usr/bin/env python
"""Step 9 -- aggregate every result into site/public/results.json and draw figures.

The website reads only this file plus the images it names. Nothing on the site is
hand-entered, so a number can never drift from the analysis that produced it.
"""
from __future__ import annotations

import sys, json, shutil, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from PIL import Image

from neurolink.align.dataset import ENCODER_FAMILY, ENCODER_LABELS
from neurolink.neural.spectral import baseline_slopes, welch_psd
from neurolink.paths import get_paths, load_config
from neurolink.recon.parametric import reconstruct
from neurolink.stimuli.params import CONDITION_LABELS, load_gray
from neurolink.viz import figures as F


def jload(p):
    return json.load(open(p)) if pathlib.Path(p).exists() else None


def main() -> None:
    cfg = load_config(); paths = get_paths()
    pub = paths.site_public
    (pub / "figures").mkdir(parents=True, exist_ok=True)
    (pub / "video").mkdir(parents=True, exist_ok=True)
    (pub / "stimuli").mkdir(parents=True, exist_ok=True)

    R = {"dataset": {}, "figures": {}, "condition_labels": CONDITION_LABELS,
         "encoder_labels": ENCODER_LABELS, "encoder_family": ENCODER_FAMILY}

    # ---------- dataset description -----------------------------------------
    subs = {}
    for sub, meta in cfg["subjects"].items():
        prep = np.load(paths.cache / f"prep_{sub}_run-{meta['runs'][0]}.npz",
                       allow_pickle=True)
        subs[sub] = {"runs": meta["runs"], "hemi": meta["hemi"],
                     "sfreq": float(prep["sfreq"]),
                     "n_good_channels": int(len(prep["ch_names"])),
                     "n_bad_channels": int(len(prep["bads"])),
                     "trials_per_run": int(len(prep["stim_id"]))}
    R["dataset"] = {
        "name": "OpenNeuro ds005953 (Hermes et al.)",
        "doi": "10.18112/openneuro.ds005953.v1.0.0",
        "subjects": subs,
        "n_images": cfg["stimuli"]["n_images"],
        "visual_angle_deg": cfg["stimuli"]["visual_angle_deg"],
        "stimulus_duration_s": 0.5,
        "conditions": CONDITION_LABELS,
    }

    # ---------- stimulus parameter table -------------------------------------
    z = np.load(paths.cache / "stimuli.npz", allow_pickle=True)
    pdf = pd.DataFrame(z["params"], columns=[str(c) for c in z["param_cols"]],
                       index=z["stim_id"])
    by_cond = []
    for tt in range(1, 8):
        s = pdf[pdf.trial_type == tt]
        by_cond.append({
            "condition": tt, "label": CONDITION_LABELS[tt], "n": int(len(s)),
            "spatial_freq_cpd": float(np.nanmean(s.spatial_freq_cpd)),
            "noise_exponent": float(np.nanmean(s.noise_exponent)),
            "rms_contrast": float(np.nanmean(s.rms_contrast)),
            "orient_concentration": float(np.nanmean(s.orient_concentration)),
            "phase_sd_rad": float(np.nanstd(s.phase_rad)),
            "orientation_sd_deg": float(np.nanstd(s.orientation_deg)),
        })
    R["stimulus_parameters"] = by_cond

    # ---------- results from earlier steps ------------------------------------
    R["alignment"] = jload(paths.cache / "results_align.json")
    R["reconstruction"] = jload(paths.cache / "results_recon.json")
    R["electrodes"] = jload(paths.cache / "results_electrodes.json")
    R["contrastive"] = jload(paths.cache / "results_contrastive.json")
    R["jepa"] = {v: jload(paths.models / f"ecogjepa_{v}_report.json")
                 for v in ("all", "sub-02")}
    R["encoders_manifest"] = jload(paths.cache / "encoders_manifest.json")
    for p in paths.cache.glob("braintv_*.json"):
        R.setdefault("braintv", []).append(jload(p))

    # ---------- figure: the Hermes replication --------------------------------
    for sub, meta in cfg["subjects"].items():
        prep = np.load(paths.cache / f"prep_{sub}_run-{meta['runs'][0]}.npz",
                       allow_pickle=True)
        sf = float(prep["sfreq"]); times = prep["times"]
        freqs, ps = welch_psd(prep["X"].astype(np.float64), sf, cfg,
                              tuple(cfg["features"]["window"]), times)
        _, pb = welch_psd(prep["X_blank"].astype(np.float64), sf, cfg,
                          tuple(cfg["features"]["window"]), times)
        bb = np.load(paths.cache / f"prep_{sub}_run-{meta['runs'][0]}.npz",
                     allow_pickle=True)["feat_broadband"]
        sel = np.argsort(-bb.mean(0))[:10]
        tt = prep["trial_type"]
        by = {c: ps[tt == c][:, sel].mean(axis=(0, 1)) for c in range(1, 8)}
        out = paths.figures / f"spectra_{sub}.png"
        F.fig_spectra(freqs, by, pb[:, sel].mean(axis=(0, 1)), out,
                      title=f"{sub}: 10 most visually driven electrodes")
        R["figures"][f"spectra_{sub}"] = out.name

    # ---------- figures driven by the alignment results -----------------------
    A = R["alignment"]
    if A:
        rows_2w, rows_cond = [], []
        for sub, sres in A["subjects"].items():
            for enc, e in sres["encoders"].items():
                rows_cond.append({"subject": sub, "encoder": enc, "label": e["label"],
                                  "family": e["family"], "value": e["condition"]["accuracy"],
                                  "ci": e["condition"]["ci"]})
                sp = e.get("spaces", {}).get("dino224")
                if sp:
                    rows_2w.append({"subject": sub, "encoder": enc, "label": e["label"],
                                    "family": e["family"], "value": sp["two_way"]})
        if rows_cond:
            o = paths.figures / "encoders_condition.png"
            F.fig_encoder_comparison(rows_cond, o, chance=1 / 7,
                                     ylabel="7-way condition accuracy",
                                     title="Which neural encoder actually sees the stimulus?")
            R["figures"]["encoders_condition"] = o.name
        if rows_2w:
            o = paths.figures / "encoders_twoway.png"
            F.fig_encoder_comparison(rows_2w, o, chance=0.5,
                                     ylabel="2-way identification vs DINOv2",
                                     title="Neural-to-DINOv2 alignment by encoder")
            R["figures"]["encoders_twoway"] = o.name
        for sub, sres in A["subjects"].items():
            best = max(sres["encoders"].items(), key=lambda kv: kv[1]["condition"]["accuracy"])
            o = paths.figures / f"confusion_{sub}.png"
            F.fig_confusion(best[1]["condition"]["confusion"], o,
                            title=f"{sub}: {best[1]['label']}")
            R["figures"][f"confusion_{sub}"] = o.name

    # ---------- figures: parameter decoding + reconstruction gallery ----------
    deg = cfg["stimuli"]["visual_angle_deg"]
    for sub in cfg["subjects"]:
        p = paths.cache / f"recon_{sub}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        cols = [str(c) for c in d["cols"]]
        o = paths.figures / f"params_{sub}.png"
        F.fig_param_scatter(d["Y"], d["P"], cols, d["trial_type"], o,
                            title=f"{sub}: stimulus parameters decoded from ECoG "
                                  f"(novel-image split)")
        R["figures"][f"params_{sub}"] = o.name

        items, gal = [], []
        rng = np.random.default_rng(0)
        pick = np.concatenate([rng.choice(np.flatnonzero(d["trial_type"] == c), 1)
                               for c in range(1, 8)])
        for i in pick:
            sid = int(d["stim_id"][i])
            img, info = reconstruct(d["P"][i], cols, d["post"][i], deg, size=256, seed=sid)
            true = load_gray(str(paths.stimuli / f"stim_{sid}.png"), 256)
            items.append((true, img, CONDITION_LABELS[int(d["trial_type"][i])]))
            Image.fromarray((img * 255).astype(np.uint8)).save(
                pub / "stimuli" / f"recon_{sub}_{sid}.png")
            Image.fromarray((true * 255).astype(np.uint8)).save(
                pub / "stimuli" / f"true_{sid}.png")
            gal.append({"stim_id": sid, "condition": int(d["trial_type"][i]),
                        "true": f"stimuli/true_{sid}.png",
                        "recon": f"stimuli/recon_{sub}_{sid}.png", "info": info})
        o = paths.figures / f"gallery_{sub}.png"
        F.fig_reconstruction_gallery(items, o,
                                     title=f"{sub}: presented vs. reconstructed from brain")
        R["figures"][f"gallery_{sub}"] = o.name
        R.setdefault("galleries", {})[sub] = gal

    # ---------- figures: JEPA curve and electrode profile ---------------------
    for v, rep in R["jepa"].items():
        if rep:
            o = paths.figures / f"jepa_{v}.png"
            F.fig_training_curve(rep, o,
                                 title=f"ECoG-JEPA pretraining ({v}) -- no stimulus labels")
            R["figures"][f"jepa_{v}"] = o.name
    if R["electrodes"]:
        for sub, e in R["electrodes"].items():
            o = paths.figures / f"electrodes_{sub}.png"
            F.fig_electrode_profile(e["single_electrode_acc"], e["ch_names"], o,
                                    title=f"{sub}: what a single electrode knows")
            R["figures"][f"electrodes_{sub}"] = o.name
            for v in e["views"]:
                R["figures"][f"brain_{sub}_{v.split('_')[-1].replace('.png','')}"] = v

    # ---------- copy assets ---------------------------------------------------
    # Only the figures the site actually references. Step 6 writes one PNG per
    # gallery trial into outputs/figures for local inspection; those are served
    # from site/public/stimuli instead, so shipping both would double the repo.
    referenced = set(R["figures"].values())
    for sub_e in (R.get("electrodes") or {}).values():
        referenced.update(sub_e.get("views", []))
    for f in paths.figures.glob("*.png"):
        if f.name in referenced:
            shutil.copy2(f, pub / "figures" / f.name)
    for f in (pub / "figures").glob("*.png"):
        if f.name not in referenced:
            f.unlink()
    for f in paths.video.glob("*.mp4"):
        shutil.copy2(f, pub / "video" / f.name)

    with open(pub / "results.json", "w") as fh:
        json.dump(R, fh, indent=2, default=float)
    print(f"[report] wrote {pub/'results.json'} "
          f"({(pub/'results.json').stat().st_size/1e3:.0f} KB)")
    print(f"[report] {len(list((pub/'figures').glob('*.png')))} figures, "
          f"{len(list((pub/'video').glob('*.mp4')))} videos")


if __name__ == "__main__":
    main()
