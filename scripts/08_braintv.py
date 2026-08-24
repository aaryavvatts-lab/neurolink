#!/usr/bin/env python
"""Step 8 -- render the Brain TV video.

The decoder is fitted on the first half of the run's trials and the video is
rendered from the second half of the continuous recording, so every frame is
out-of-sample. No trial timing is used at render time: the model sees a raw
500 ms window of LFP every 50 ms and says what it thinks is on the screen.
"""
from __future__ import annotations

import sys, json, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

from neurolink.align import ridge
from neurolink.align.dataset import load_image_spaces, load_subject
from neurolink.bids_io import read_run
from neurolink.neural.spectral import baseline_slopes, welch_psd
from neurolink.paths import get_paths, load_config
from neurolink.preprocess import clean_raw, epoch_run
from neurolink.recon.braintv import (blank_reference, sliding_windows,
                                     stimulus_at, window_features)
from neurolink.recon.parametric import build_targets, reconstruct
from neurolink.stimuli.params import CONDITION_LABELS, load_gray

BLANK = 8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="sub-01")
    ap.add_argument("--run", default="01")
    ap.add_argument("--seconds", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(); paths = get_paths()
    bt = cfg["braintv"]; deg = cfg["stimuli"]["visual_angle_deg"]
    sub, run = args.sub, args.run

    spaces, space_ids, params = load_image_spaces(paths)
    pdf = pd.DataFrame(params["X"], columns=params["cols"], index=space_ids)

    # ---- fit the decoder on the FIRST HALF of this run's trials --------------
    prep = np.load(paths.cache / f"prep_{sub}_run-{run}.npz", allow_pickle=True)
    feat_names = [str(f) for f in prep["feat_names"]]
    sid_all = prep["stim_id"]; tt_all = prep["trial_type"]
    order = np.argsort(prep["onset"])
    half = len(order) // 2
    tr = order[:half]
    Ytr, cols, _ = build_targets(pdf, sid_all[tr])
    Xtr = prep["feat_X"][tr]
    reg = ridge.make_regressor(cfg, len(tr), Xtr.shape[1]).fit(Xtr, Ytr)
    clf = ridge.make_classifier(cfg, len(tr), Xtr.shape[1]).fit(Xtr, tt_all[tr])
    t_split = float(prep["onset"][order[half]])
    print(f"[braintv] fitted on {len(tr)} trials (t < {t_split:.1f}s); "
          f"rendering from t >= {t_split:.1f}s")

    # ---- continuous data + the baselines the features are defined against ----
    rd = read_run(paths, sub, run)
    raw, _ = clean_raw(rd, cfg)
    good = [c for c in raw.ch_names if c not in raw.info["bads"]]
    data = raw.get_data(picks=good)
    sf = float(raw.info["sfreq"])

    freqs, psd_blank = welch_psd(prep["X_blank"].astype(np.float64), sf, cfg,
                                 tuple(cfg["features"]["window"]), prep["times"])
    slopes, _ = baseline_slopes(psd_blank, freqs, cfg)
    ref = blank_reference(psd_blank, freqs, slopes, cfg)

    t_end = min(t_split + (args.seconds or bt["max_seconds"]),
                data.shape[1] / sf) if (args.seconds or bt["max_seconds"]) else data.shape[1] / sf
    W, centres = sliding_windows(data, sf, bt["window_s"], bt["hop_s"], t_split, t_end)
    print(f"[braintv] {len(W)} windows of {bt['window_s']}s, hop {bt['hop_s']}s "
          f"covering {centres[0]:.1f}-{centres[-1]:.1f}s")

    F = window_features(W, sf, cfg, slopes, ref, feat_names)
    P = reg.predict(F)
    post = clf.predict_proba(F)
    classes = list(clf.named_steps["clf"].classes_)

    true_sid = stimulus_at(centres, rd.events, blank_id=int(cfg["stimuli"]["n_images"]))
    stim_cache = {}

    def stim_img(s):
        if s not in stim_cache:
            stim_cache[s] = load_gray(str(paths.stimuli / f"stim_{int(s)}.png"), 256)
        return stim_cache[s]

    # broadband trace from the most informative electrodes, for the live plot
    bb_idx = feat_names.index("broadband")
    n_ch = len(good)
    bb = F[:, bb_idx * n_ch:(bb_idx + 1) * n_ch]
    top = np.argsort(-bb.std(axis=0))[:10]
    bb_trace = bb[:, top].mean(axis=1)

    frames_dir = paths.video / "frames"; frames_dir.mkdir(exist_ok=True)
    out_mp4 = paths.video / f"braintv_{sub}_run-{run}.mp4"
    writer = imageio.get_writer(out_mp4, fps=bt["fps"], codec="libx264",
                                quality=7, macro_block_size=8)

    labels = [CONDITION_LABELS[c] for c in classes]
    for i in range(len(W)):
        fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
        gs = gridspec.GridSpec(2, 3, height_ratios=[2.4, 1.0], hspace=0.32, wspace=0.22)

        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(stim_img(true_sid[i]), cmap="gray", vmin=0, vmax=1)
        ax.set_title("on screen", fontsize=11); ax.set_xticks([]); ax.set_yticks([])

        img, info = reconstruct(P[i], cols, post[i], deg, size=256, seed=i)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax2.set_title("reconstructed from brain", fontsize=11)
        ax2.set_xticks([]); ax2.set_yticks([])

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.barh(np.arange(len(classes)), post[i], color="#2F4B7C")
        ax3.set_yticks(np.arange(len(classes)))
        ax3.set_yticklabels(labels, fontsize=7)
        ax3.set_xlim(0, 1); ax3.invert_yaxis()
        ax3.set_title("decoded stimulus class", fontsize=10)

        ax4 = fig.add_subplot(gs[1, :])
        lo = max(0, i - 120)
        ax4.plot(centres[lo:i + 1], bb_trace[lo:i + 1], color="#C44E52", lw=1.3)
        ax4.set_xlim(centres[lo], centres[lo] + 6.0)
        ax4.set_ylim(np.percentile(bb_trace, 1), np.percentile(bb_trace, 99))
        ax4.set_ylabel("broadband\n70-200 Hz", fontsize=8)
        ax4.set_xlabel("time in recording (s)", fontsize=8)
        for _, e in rd.events.iterrows():
            if centres[lo] <= e.onset <= centres[lo] + 6.0 and e.trial_type != BLANK:
                ax4.axvspan(e.onset, e.onset + e.duration, color="0.85", zorder=0)
        fig.suptitle(f"NeuroLink Brain TV -- {sub} run-{run} "
                     f"(decoder never saw this part of the recording)", fontsize=11)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(buf)
        plt.close(fig)
        if i % 100 == 0:
            print(f"  frame {i}/{len(W)}")
    writer.close()

    acc = float(np.mean([
        (classes[int(np.argmax(post[i]))] ==
         (BLANK if true_sid[i] == cfg["stimuli"]["n_images"] else None))
        for i in range(0)] or [np.nan]))
    meta = {"sub": sub, "run": run, "n_frames": int(len(W)),
            "fps": bt["fps"], "window_s": bt["window_s"], "hop_s": bt["hop_s"],
            "t_start": float(centres[0]), "t_end": float(centres[-1]),
            "fitted_on_trials": int(len(tr)), "out_of_sample": True,
            "file": out_mp4.name,
            "size_mb": round(out_mp4.stat().st_size / 1e6, 2)}
    with open(paths.cache / f"braintv_{sub}_run-{run}.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[braintv] wrote {out_mp4} ({meta['size_mb']} MB, {len(W)} frames)")


if __name__ == "__main__":
    main()
