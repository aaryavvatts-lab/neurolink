#!/usr/bin/env python
"""Step 10 -- export everything the website needs to run models in the browser.

Three payloads:
  forward.json   image features -> predicted cortical response (the analyser tool)
  decoder.json   per-trial neural features + the linear decoder (the decode tool)
  brain.json     decimated pial surface + electrode positions and scores (3-D view)
  spectra.json   averaged power spectra per condition (interactive chart)
"""
from __future__ import annotations

import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import nibabel as nib
import pyvista as pv
from PIL import Image

from neurolink import forward as FW
from neurolink.align import ridge
from neurolink.align.splits import novel_image_splits
from neurolink.bids_io import canon_ch, read_electrodes
from neurolink.neural.spectral import welch_psd
from neurolink.paths import get_paths, load_config
from neurolink.recon.parametric import build_targets as recon_targets
from neurolink.stimuli.params import CONDITION_LABELS

R3 = lambda a: np.round(np.asarray(a, dtype=float), 3).tolist()
R4 = lambda a: np.round(np.asarray(a, dtype=float), 4).tolist()


def export_forward(paths, cfg, pdf, pub):
    out = {}
    for sub, meta in cfg["subjects"].items():
        prep = np.load(paths.cache / f"prep_{sub}_run-{meta['runs'][0]}.npz",
                       allow_pickle=True)
        X = FW.build_design(pdf, prep["stim_id"])
        Y, sel = FW.build_targets(prep)
        m = FW.fit(X, Y, prep["stim_id"])
        m["electrodes_used"] = [str(prep["ch_names"][i]) for i in sel]
        m["predicted"] = R4(m["predicted"]); m["observed"] = R4(m["observed"])
        m["trial_type"] = prep["trial_type"].tolist()
        m["stim_id"] = prep["stim_id"].tolist()
        out[sub] = m
        print(f"[forward] {sub}: " + "  ".join(
            f"{t} r={m['accuracy'][t]['r']:+.3f}" for t in FW.TARGETS))
    json.dump(out, open(pub / "forward.json", "w"))
    return out


def export_decoder(paths, cfg, pdf, pub, sub="sub-01"):
    """Per-trial features plus a decoder the browser can actually run.

    The saved weights come from folds in which the trial being shown was held
    out, so running them in the browser reproduces a genuine out-of-sample
    prediction rather than replaying a fit that already saw the answer.
    """
    meta = cfg["subjects"][sub]
    prep = np.load(paths.cache / f"prep_{sub}_run-{meta['runs'][0]}.npz", allow_pickle=True)
    Xf = prep["feat_X"].astype(np.float64)
    sid = prep["stim_id"]; tt = prep["trial_type"]
    Y, cols, extra = recon_targets(pdf, sid)

    n_tr = len(Xf)
    P = np.zeros_like(Y); post = np.zeros((n_tr, 7))
    fold_of = np.zeros(n_tr, dtype=int)
    models = []
    for k, sp in enumerate(novel_image_splits(sid, cfg["align"]["n_folds"])):
        sp.check(sid)
        reg = ridge.make_regressor(cfg, len(sp.train), Xf.shape[1]).fit(Xf[sp.train], Y[sp.train])
        clf = ridge.make_classifier(cfg, len(sp.train), Xf.shape[1]).fit(Xf[sp.train], tt[sp.train])
        P[sp.test] = reg.predict(Xf[sp.test]); post[sp.test] = clf.predict_proba(Xf[sp.test])
        fold_of[sp.test] = k

        # Compose scale -> PCA -> ridge into one affine map the browser can apply.
        #   z = (x - mu) / s
        #   p = (z - pca_mu) @ C.T
        #   y = p @ coef.T + b0
        # so with A = C.T @ coef.T:
        #   y = x @ (A / s) - ((mu / s) + pca_mu) @ A + b0
        sc, pca, rg = reg.named_steps["scale"], reg.named_steps["pca"], reg.named_steps["ridge"]
        A = pca.components_.T @ np.atleast_2d(rg.coef_).T          # (n_feat, n_targets)
        W = A / sc.scale_[:, None]
        b = np.atleast_1d(rg.intercept_) - ((sc.mean_ / sc.scale_) + pca.mean_) @ A
        exact = np.abs(Xf[sp.test] @ W + b - P[sp.test]).max()
        assert exact < 1e-6, f"affine composition wrong: {exact:.2e}"
        models.append({"W": np.round(W, 7).tolist(), "b": np.round(b, 7).tolist()})

    # Verify the exported affine map reproduces sklearn's own predictions.
    err = 0.0
    for k, sp in enumerate(novel_image_splits(sid, cfg["align"]["n_folds"])):
        W = np.array(models[k]["W"]); b = np.array(models[k]["b"])
        # Reproduce exactly what the browser will compute, rounded values and all.
        xr = np.round(Xf[sp.test], 3)
        err = max(err, float(np.abs(xr @ W + b - P[sp.test]).max()))
    print(f"[decoder] exported map matches sklearn to {err:.2e} "
          f"(target scale {np.abs(P).max():.2f})")
    assert err < 1e-2, "rounding is costing too much precision"

    keep = np.argsort(sid)
    out = {
        "subject": sub, "cols": cols, "n_features": int(Xf.shape[1]),
        "condition_labels": CONDITION_LABELS,
        "folds": models, "fold_of": fold_of.tolist(),
        "max_export_error": err,
        "trials": [{
            "i": int(i), "stim_id": int(sid[i]), "cond": int(tt[i]),
            "x": R3(Xf[i]), "post": R4(post[i]),
            "true": R4(Y[i]), "phase_true": float(extra["phase_true"][i]),
        } for i in keep],
    }
    json.dump(out, open(pub / "decoder.json", "w"))
    print(f"[decoder] {len(out['trials'])} trials, "
          f"{(pub/'decoder.json').stat().st_size/1e6:.1f} MB")
    return out


def export_brain(paths, cfg, pub, target_faces=14000):
    elec = json.load(open(paths.cache / "results_electrodes.json"))
    out = {}
    for sub, meta in cfg["subjects"].items():
        g = nib.load(str(paths.pial_gii(sub, meta["hemi"])))
        v = np.asarray(g.agg_data("NIFTI_INTENT_POINTSET"), float)
        f = np.asarray(g.agg_data("NIFTI_INTENT_TRIANGLE"), np.int64)
        mesh = pv.PolyData(v, np.hstack([np.full((len(f), 1), 3), f]).ravel())
        frac = min(0.98, max(0.0, 1.0 - target_faces / mesh.n_faces))
        small = mesh.decimate(frac).clean().compute_normals(
            point_normals=True, cell_normals=False, consistent_normals=True)
        sv = np.asarray(small.points, float)
        sf = small.faces.reshape(-1, 4)[:, 1:]
        e = elec[sub]
        out[sub] = {
            "hemi": meta["hemi"],
            "vertices": R3(sv.ravel()), "faces": np.asarray(sf).ravel().tolist(),
            "normals": R3(np.asarray(small.point_data["Normals"], float).ravel()),
            "centre": R3(sv.mean(0)),
            "electrodes": [{"name": n, "xyz": R3(c), "acc": float(a)}
                           for n, c, a in zip(e["ch_names"], e["coords_mm"],
                                              e["single_electrode_acc"])],
            "chance": e["chance"],
        }
        print(f"[brain] {sub}: {len(sv)} verts, {len(sf)} faces, "
              f"{len(out[sub]['electrodes'])} electrodes")
    json.dump(out, open(pub / "brain.json", "w"))
    return out


def export_spectra(paths, cfg, pub):
    out = {}
    for sub, meta in cfg["subjects"].items():
        prep = np.load(paths.cache / f"prep_{sub}_run-{meta['runs'][0]}.npz",
                       allow_pickle=True)
        sf = float(prep["sfreq"])
        freqs, ps = welch_psd(prep["X"].astype(np.float64), sf, cfg,
                              tuple(cfg["features"]["window"]), prep["times"])
        _, pb = welch_psd(prep["X_blank"].astype(np.float64), sf, cfg,
                          tuple(cfg["features"]["window"]), prep["times"])
        sel = np.argsort(-prep["feat_broadband"].mean(0))[:10]
        tt = prep["trial_type"]
        keep = (freqs >= 4) & (freqs <= 200)
        base = np.log10(pb[:, sel].mean(axis=(0, 1)) + 1e-30)
        out[sub] = {
            "freqs": R3(freqs[keep]),
            "baseline": R4(base[keep]),
            "conditions": {int(c): R4((np.log10(ps[tt == c][:, sel].mean(axis=(0, 1)) + 1e-30)
                                       - base)[keep]) for c in range(1, 8)},
            "n_electrodes": 10,
        }
    json.dump(out, open(pub / "spectra.json", "w"))
    print(f"[spectra] written")
    return out


def export_stimuli(paths, cfg, pdf, pub, px=192, full_px=512, noise_per_cond=6):
    """Thumbnails for browsing, plus full-size copies for the measurement tool.

    The analyser has to measure a preset exactly as the Python pipeline did, and
    the pipeline worked at 512. Measuring an upscaled 192 thumbnail instead makes
    the falloff exponent read about 1.4 too steep and wipes out the contrast of
    fine noise, because the detail is simply not in the file. Gratings compress to
    about 1 kB at 512 so every one ships full size; the noise patterns cost about
    150 kB each, so a sample of each type ships instead.
    """
    d = pub / "stim"; d.mkdir(exist_ok=True)
    dfull = pub / "stimfull"; dfull.mkdir(exist_ok=True)
    full_ids = set()
    for cond in range(1, 8):
        ids = pdf[pdf.trial_type == cond].index.tolist()
        full_ids.update(ids if cond >= 4 else ids[:noise_per_cond])

    rows = []
    for sid, r in pdf.iterrows():
        src = paths.stimuli / f"stim_{int(sid)}.png"
        big = Image.open(src).convert("L")
        big.resize((px, px), Image.LANCZOS).save(d / f"{int(sid)}.png", optimize=True)
        if int(sid) in full_ids:
            big.resize((full_px, full_px), Image.LANCZOS).save(
                dfull / f"{int(sid)}.png", optimize=True)
        rows.append({"id": int(sid), "cond": int(r["trial_type"]),
                     "full": int(sid) in full_ids,
                     "sf": None if not np.isfinite(r["spatial_freq_cpd"]) else round(float(r["spatial_freq_cpd"]), 3),
                     "alpha": None if not np.isfinite(r["noise_exponent"]) else round(float(r["noise_exponent"]), 3),
                     "contrast": round(float(r["rms_contrast"]), 4),
                     "conc": round(float(r["orient_concentration"]), 4),
                     "phase": None if not np.isfinite(r["phase_rad"]) else round(float(r["phase_rad"]), 4)})
    json.dump({"stimuli": rows, "labels": CONDITION_LABELS, "px": px,
               "full_px": full_px}, open(pub / "stimuli.json", "w"))
    mb = sum(f.stat().st_size for f in dfull.glob("*.png")) / 1e6
    print(f"[stimuli] {len(rows)} thumbnails at {px}px; "
          f"{len(full_ids)} full-size at {full_px}px ({mb:.1f} MB)")


def main() -> None:
    cfg = load_config(); paths = get_paths()
    pub = paths.site_public
    pub.mkdir(parents=True, exist_ok=True)
    z = np.load(paths.cache / "stimuli.npz", allow_pickle=True)
    pdf = pd.DataFrame(z["params"], columns=[str(c) for c in z["param_cols"]],
                       index=z["stim_id"])

    export_forward(paths, cfg, pdf, pub)
    export_decoder(paths, cfg, pdf, pub)
    export_brain(paths, cfg, pub)
    export_spectra(paths, cfg, pub)
    export_stimuli(paths, cfg, pdf, pub)
    print("\n[done] web payloads written to", pub)


if __name__ == "__main__":
    main()
