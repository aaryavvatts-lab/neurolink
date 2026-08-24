#!/usr/bin/env python
"""Step 3 -- build every neural encoder's per-trial embeddings.

Encoder A (spectral) already lives in the prep cache from step 2. This script
adds Encoder B (off-the-shelf EEG foundation models) and Encoder C (ECoG-JEPA),
plus the bandwidth accounting that explains the gap between them.

Writes outputs/cache/enc_<sub>_run-<run>.npz.
"""
from __future__ import annotations

import sys, time, json, pathlib, argparse, traceback
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from neurolink.bids_io import canon_ch, read_electrodes, read_run
from neurolink.neural import foundation
from neurolink.neural.ecogjepa import ECoGJEPA, SpecConfig, embed_epochs, spectrogram, normalize
from neurolink.neural.spectral import welch_psd
from neurolink.paths import get_paths, load_config
from neurolink.preprocess import clean_raw


def load_jepa(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    e = ck["cfg"]
    m = ECoGJEPA(n_freq=ck["n_freq"], d_model=e["d_model"], n_layers=e["n_layers"],
                 n_heads=e["n_heads"], max_len=128, dropout=0.0)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-foundation", action="store_true")
    ap.add_argument("--only", default=None,
                    help="only (re)compute this encoder key; merges into the existing cache")
    args = ap.parse_args()

    cfg = load_config(); paths = get_paths()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sc = SpecConfig(sfreq_target=cfg["ecogjepa"]["sfreq_target"])
    win = tuple(cfg["features"]["window"])

    jepas = {}
    for variant in ("all", "sub-02"):
        p = paths.models / f"ecogjepa_{variant}.pt"
        if p.exists():
            jepas[variant] = load_jepa(p, device)
            print(f"[jepa] loaded {variant}")

    man_path = paths.cache / "encoders_manifest.json"
    manifest = json.load(open(man_path)) if man_path.exists() else {}
    for sub, meta in cfg["subjects"].items():
        elec = read_electrodes(paths, sub).set_index("idx")
        for run in meta["runs"]:
            key = f"{sub}_run-{run}"
            prep = np.load(paths.cache / f"prep_{sub}_run-{run}.npz", allow_pickle=True)
            ch_names = [str(c) for c in prep["ch_names"]]
            coords = np.stack([elec.loc[canon_ch(c), ["x", "y", "z"]].to_numpy(float)
                               for c in ch_names])
            # Merge into any existing cache so a single encoder can be added or
            # recomputed without redoing the expensive ones.
            dest = paths.cache / f"enc_{sub}_run-{run}.npz"
            out = {}
            if dest.exists():
                prev = np.load(dest, allow_pickle=True)
                out = {k: prev[k] for k in prev.files}
            out.update({"ch_names": np.array(ch_names, dtype=object),
                        "coords_mm": coords,
                        "stim_id": prep["stim_id"], "trial_type": prep["trial_type"]})
            info = manifest.get(key, {})

            # --- Encoder C: our ECoG-JEPA -------------------------------------
            X = prep["X"].astype(np.float64) * 1e6
            for variant, model in jepas.items():
                if args.only and f"ecogjepa_{variant}" != args.only:
                    continue
                t = time.time()
                E = embed_epochs(model, X, float(prep["sfreq"]), sc, win,
                                 float(prep["times"][0]), device=device)
                out[f"ecogjepa_{variant}"] = E
                info[f"ecogjepa_{variant}"] = {"dim": int(E.shape[1]),
                                               "d_model": model.d_model}
                print(f"[{key}] ecogjepa_{variant} -> {E.shape} ({time.time()-t:.0f}s)")

            # --- Encoder B: off-the-shelf EEG foundation models ---------------
            need_foundation = not args.skip_foundation and (
                args.only in (None, "signal_jepa", "cbramod"))
            if need_foundation:
                raw, _ = clean_raw(read_run(paths, sub, run), cfg)
                good = [c for c in raw.ch_names if c not in raw.info["bads"]]
                assert good == ch_names, "channel order drifted between steps"
                data = raw.get_data(picks=good)
                onsets = prep["onset"]

                for mk in ("signal_jepa", "cbramod"):
                    if args.only and mk != args.only:
                        continue
                    try:
                        t = time.time()
                        E, meta_b = foundation.embed(mk, data, float(raw.info["sfreq"]),
                                                     onsets, ch_names, coords,
                                                     device=device)
                        out[mk] = E
                        out[f"{mk}_kept"] = meta_b["kept"]
                        info[mk] = meta_b["info"]
                        print(f"[{key}] {mk} -> {E.shape} ({time.time()-t:.0f}s)")
                    except Exception as exc:
                        print(f"[{key}] {mk} FAILED: {type(exc).__name__}: {exc}")
                        traceback.print_exc()
                        info[mk] = {"error": f"{type(exc).__name__}: {exc}"}

                # Bandwidth accounting, once per run.
                fr, ps = welch_psd(prep["X"].astype(np.float64), float(prep["sfreq"]),
                                   cfg, win, prep["times"])
                _, pb = welch_psd(prep["X_blank"].astype(np.float64), float(prep["sfreq"]),
                                  cfg, win, prep["times"])
                info["bandwidth"] = foundation.bandwidth_accounting(ps, pb, fr)
                print(f"[{key}] bandwidth: "
                      + ", ".join(f"{k}={v:.3f}" for k, v in info["bandwidth"].items()
                                  if k.startswith("frac")))

            np.savez_compressed(dest, **out)
            manifest[key] = info
            print(f"[{key}] saved {dest.stat().st_size/1e6:.0f}MB\n")

    with open(man_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print("[done] wrote encoders_manifest.json")


if __name__ == "__main__":
    main()
