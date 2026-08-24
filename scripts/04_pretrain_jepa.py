#!/usr/bin/env python
"""Step 4 -- self-supervised pretraining of ECoG-JEPA on the raw continuous LFP.

No stimulus labels and no event files are touched here. Two checkpoints are
produced:
  all    -- pretrained on every subject
  sub-02 -- pretrained on sub-02 only, so encoding sub-01 with it is a genuine
            transfer to an unseen brain (other hemisphere, other electrode
            layout, other sample rate)
"""
from __future__ import annotations

import sys, time, json, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from neurolink.bids_io import read_run
from neurolink.neural.ecogjepa import SpecConfig, build_corpus, pretrain
from neurolink.paths import get_paths, load_config
from neurolink.preprocess import clean_raw


def continuous_runs(cfg, paths, only: str | None = None):
    out = []
    for sub, meta in cfg["subjects"].items():
        if only and sub != only:
            continue
        for run in meta["runs"]:
            raw, _ = clean_raw(read_run(paths, sub, run), cfg)
            good = [c for c in raw.ch_names if c not in raw.info["bads"]]
            data = raw.get_data(picks=good) * 1e6                # volts -> microvolts
            out.append((f"{sub}_run-{run}", data, float(raw.info["sfreq"])))
            print(f"  [{sub} run-{run}] {data.shape} @ {raw.info['sfreq']:.0f} Hz")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all", choices=["all", "sub-02"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    paths = get_paths()
    if args.epochs:
        cfg["ecogjepa"]["epochs"] = args.epochs

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sc = SpecConfig(sfreq_target=cfg["ecogjepa"]["sfreq_target"])

    print(f"[jepa] variant={args.variant} device={device}")
    runs = continuous_runs(cfg, paths, only=None if args.variant == "all" else args.variant)
    corpus = build_corpus(runs, sc)
    ch_tot = sum(m["n_ch"] for m in corpus["meta"])
    fr_tot = sum(m["T"] for m in corpus["meta"])
    print(f"[jepa] corpus: {len(runs)} runs, {ch_tot} channels, "
          f"{fr_tot} frames/ch, {sc.fmin}-{sc.fmax} Hz in {len(corpus['freqs'])} bins")

    t = time.time()
    steps = args.steps or cfg["ecogjepa"].get("steps_per_epoch", 140)
    model, rep = pretrain(corpus, cfg, device=device, steps_per_epoch=steps)
    print(f"[jepa] trained {rep.n_params/1e6:.2f}M params in {time.time()-t:.0f}s")
    print(f"[jepa] val L1 {rep.val_losses[-1]:.4f} vs trivial baseline {rep.baseline_val:.4f} "
          f"({100*(1-rep.val_losses[-1]/rep.baseline_val):.1f}% better)")

    dest = paths.models / f"ecogjepa_{args.variant}.pt"
    torch.save({"state_dict": model.state_dict(), "n_freq": len(corpus["freqs"]),
                "cfg": cfg["ecogjepa"], "spec": rep.spec,
                "freqs": corpus["freqs"]}, dest)
    with open(paths.models / f"ecogjepa_{args.variant}_report.json", "w") as fh:
        json.dump({"losses": rep.losses, "val_losses": rep.val_losses,
                   "baseline_val": rep.baseline_val, "n_params": rep.n_params,
                   "variant": args.variant, "n_channels": ch_tot}, fh, indent=2)
    print(f"[jepa] saved {dest}")


if __name__ == "__main__":
    main()
