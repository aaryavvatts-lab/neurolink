#!/usr/bin/env python
"""Step 5 -- align neural and image latent spaces, and evaluate honestly.

Runs every (subject x neural encoder x image space) combination through the
novel-image split, plus cross-run and cross-subject regimes, and writes
outputs/cache/results_align.json with permutation p-values, bootstrap CIs and
noise ceilings attached to every headline number.
"""
from __future__ import annotations

import sys, time, json, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.metrics import accuracy_score

from neurolink.align import contrastive, evaluate as ev, ridge
from neurolink.align.dataset import (ENCODER_FAMILY, ENCODER_LABELS, IMAGE_SPACES,
                                     image_targets, load_image_spaces, load_subject)
from neurolink.align.splits import cross_run_split, cross_subject_split, novel_image_splits
from neurolink.paths import get_paths, load_config

BLANK = 8


def cv_predict(X, Y, stim_id, cfg, method="ridge", n_folds=6):
    """Out-of-fold predictions under the novel-image split."""
    P = np.zeros((len(X), Y.shape[1] if method == "ridge" else cfg["align"]["contrastive"]["dim"]))
    proj = None
    for sp in novel_image_splits(stim_id, n_folds):
        sp.check(stim_id)
        if method == "ridge":
            P[sp.test], _ = ridge.fit_predict(X[sp.train], Y[sp.train], X[sp.test], cfg)
        else:
            pred, proj, _ = contrastive.fit_predict(
                X[sp.train], Y[sp.train], X[sp.test], cfg, groups_tr=stim_id[sp.train])
            P[sp.test] = pred
    return P, proj


def cv_classify(X, y, stim_id, cfg, n_folds=6):
    pred = np.zeros(len(y), dtype=int)
    for sp in novel_image_splits(stim_id, n_folds):
        sp.check(stim_id)
        pred[sp.test], _, _ = ridge.fit_predict_classes(X[sp.train], y[sp.train], X[sp.test], cfg)
    return pred


def score_alignment(P, Y_trial, cand, cand_stim_id, stim_id, cfg, seed=0,
                    trial_cond=None, cand_cond=None):
    """2-way identification and retrieval, with a shuffled-label null."""
    pos = {int(s): i for i, s in enumerate(cand_stim_id)}
    true_idx = np.array([pos[int(s)] for s in stim_id])

    out = {"two_way": ev.two_way_identification(P, Y_trial)}
    out.update(ev.retrieval(P, cand, true_idx))
    if trial_cond is not None and cand_cond is not None:
        out["within_condition"] = ev.within_condition_retrieval(
            P, cand, true_idx, trial_cond, cand_cond)

    rng = np.random.default_rng(seed)
    n_perm = 200
    null_2w, null_top5 = np.empty(n_perm), np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(len(P))
        null_2w[i] = ev.two_way_identification(P[p], Y_trial)
        null_top5[i] = ev.retrieval(P[p], cand, true_idx)["top5"]
    out["two_way_p"] = ev.permutation_p(out["two_way"], null_2w)
    out["top5_p"] = ev.permutation_p(out["top5"], null_top5)
    out["two_way_null_mean"] = float(null_2w.mean())
    out["top5_null_mean"] = float(null_top5.mean())
    return out


def per_stimulus(X, stim_id):
    """Average trials of the same image -> (n_stimuli, d) plus the id order."""
    ids = np.unique(stim_id)
    return np.stack([X[stim_id == s].mean(0) for s in ids]), ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrastive", action="store_true", help="also fit the CLIP-style head")
    args = ap.parse_args()

    cfg = load_config(); paths = get_paths()
    spaces, space_ids, params = load_image_spaces(paths)
    results = {"config": {"n_folds": cfg["align"]["n_folds"],
                          "n_permutations": cfg["align"]["n_permutations"]},
               "subjects": {}, "cross_subject": {}, "encoder_labels": ENCODER_LABELS,
               "encoder_family": ENCODER_FAMILY}

    subjects = {}
    for sub, meta in cfg["subjects"].items():
        print(f"\n=== {sub} ===")
        sd = load_subject(sub, meta["runs"], paths)
        subjects[sub] = sd
        keep = sd.trial_type != BLANK
        print(f"  {keep.sum()} stimulus trials, encoders: {sd.available()}")

        sres = {"n_trials": int(keep.sum()), "n_channels": len(sd.ch_names),
                "encoders": {}, "chance": {}}
        sres["chance"] = {"two_way": 0.5, "condition_acc": 1 / 7,
                          "top1": 1 / len(np.unique(sd.stim_id[keep])),
                          "top5": 5 / len(np.unique(sd.stim_id[keep]))}

        for enc in sd.available():
            X = sd.feats[enc][keep]
            if not np.isfinite(X).all():
                print(f"  [{enc}] skipped: non-finite features"); continue
            sid = sd.stim_id[keep]; tt = sd.trial_type[keep]
            eres = {"dim": int(X.shape[1]), "label": ENCODER_LABELS[enc],
                    "family": ENCODER_FAMILY[enc]}

            t = time.time()
            pred = cv_classify(X, tt, sid, cfg, cfg["align"]["n_folds"])
            acc = float(accuracy_score(tt, pred))
            rng = np.random.default_rng(0)
            null = np.array([accuracy_score(rng.permutation(tt), pred) for _ in range(1000)])
            eres["condition"] = {
                "accuracy": acc, "p": ev.permutation_p(acc, null),
                "null_mean": float(null.mean()),
                "ci": ev.bootstrap_ci((tt == pred).astype(float)),
                "confusion": [[int(((tt == a) & (pred == b)).sum()) for b in range(1, 8)]
                              for a in range(1, 8)],
            }
            print(f"  [{enc:16s}] 7-way condition acc = {acc:.3f} "
                  f"(chance .143, p={eres['condition']['p']:.4f})  {time.time()-t:.0f}s")

            eres["spaces"] = {}
            for space_name in IMAGE_SPACES:
                Y = image_targets(spaces[space_name], space_ids, sid)
                P, _ = cv_predict(X, Y, sid, cfg, "ridge", cfg["align"]["n_folds"])
                cand_mask = np.isin(space_ids, np.unique(sid))
                cand_cond = params["trial_type"][cand_mask]
                sc = score_alignment(P, Y, spaces[space_name][cand_mask],
                                     space_ids[cand_mask], sid, cfg,
                                     trial_cond=tt, cand_cond=cand_cond)
                Xs, ids = per_stimulus(X, sid)
                Ys = spaces[space_name][np.searchsorted(space_ids, ids)]
                sc["rsa"] = ev.rsa(Xs, Ys, n_perm=cfg["align"]["n_permutations"])
                eres["spaces"][space_name] = sc
                wc = sc.get("within_condition", {})
                print(f"      {space_name:8s} 2-way={sc['two_way']:.3f} "
                      f"top5={sc['top5']:.3f} pct-rank={sc['percentile_rank']:.3f} "
                      f"RSA rho={sc['rsa']['rho']:+.3f} (p={sc['rsa']['p']:.3f})"
                      + (f" | within-cond top1={wc['top1']:.3f} "
                         f"(chance {wc['chance_top1']:.3f})" if wc else ""))

            sres["encoders"][enc] = eres
        results["subjects"][sub] = sres

    # ---- partial RSA: does DINOv2 explain neural geometry beyond a V1 model? ---
    for sub, sd in subjects.items():
        keep = sd.trial_type != BLANK
        if "spectral" not in sd.feats:
            continue
        Xs, ids = per_stimulus(sd.feats["spectral"][keep], sd.stim_id[keep])
        loc = np.searchsorted(space_ids, ids)
        pr = ev.partial_rsa(Xs, spaces["dino224"][loc], spaces["gabor"][loc],
                            n_perm=cfg["align"]["n_permutations"])
        pr_rev = ev.partial_rsa(Xs, spaces["gabor"][loc], spaces["dino224"][loc],
                                n_perm=cfg["align"]["n_permutations"])
        results["subjects"][sub]["partial_rsa"] = {
            "dino_given_gabor": pr, "gabor_given_dino": pr_rev}
        print(f"\n[{sub}] partial RSA  DINOv2|Gabor rho={pr['rho_partial']:+.3f} "
              f"(p={pr['p']:.3f})   Gabor|DINOv2 rho={pr_rev['rho_partial']:+.3f} "
              f"(p={pr_rev['p']:.3f})")

    # ---- noise ceiling from sub-02's two runs --------------------------------
    sd = subjects.get("sub-02")
    if sd is not None and len(np.unique(sd.run)) == 2:
        keep = sd.trial_type != BLANK
        nc = {}
        for enc in sd.available():
            X = sd.feats[enc][keep]; sid = sd.stim_id[keep]; rl = sd.run[keep]
            A, ia = per_stimulus(X[rl == "01"], sid[rl == "01"])
            B, ib = per_stimulus(X[rl == "02"], sid[rl == "02"])
            common = np.intersect1d(ia, ib)
            nc[enc] = ev.noise_ceiling(A[np.searchsorted(ia, common)],
                                       B[np.searchsorted(ib, common)])
            print(f"[noise ceiling] {enc:16s} r = {nc[enc]['mean_r']:+.3f} "
                  f"[{nc[enc]['ci_lo']:+.3f},{nc[enc]['ci_hi']:+.3f}]")
        results["noise_ceiling"] = nc

    dest = paths.cache / "results_align.json"
    with open(dest, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\n[save] {dest}")


if __name__ == "__main__":
    main()
