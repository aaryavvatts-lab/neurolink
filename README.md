# NeuroLink — decoding visual perception from human intracranial LFP

**[Live site](https://neurolink-ecru.vercel.app)** · thirteen pages, two tools that run in
your browser, and a 3D viewer over the real brain surfaces.

Two people with subdural electrode arrays over occipital cortex viewed gratings and noise
patterns for half a second at a time. This repo reconstructs what they were looking at from
the raw voltage, and uses the same data to test a claim that is usually asserted rather than
measured: **that an off-the-shelf "brain foundation model" is the right encoder for
intracranial signals.**

It is not, and the reason is measurable.

---

## The findings

**1. Decoding accuracy tracks encoder bandwidth, monotonically.**

Only **9.6 %** of the stimulus-evoked change in the power spectrum falls below 40 Hz — the
ceiling SignalJEPA was pretrained under. Another 49 % sits between 40 and 125 Hz, inside
CBraMod's range. The rest is broadband gamma. Seven-way condition decoding in subject 1
lines up with exactly that ordering:

| Encoder | Family | Usable band | 7-way accuracy | RSA vs DINOv2 |
|---|---|---|---|---|
| SignalJEPA | off-the-shelf EEG | 0.5–40 Hz | 0.724 | +0.115 |
| CBraMod | off-the-shelf EEG | 0–125 Hz | 0.862 | +0.412 |
| Spectral (Hermes decomposition) | hand-crafted | full | 0.895 | +0.143 |
| ECoG-JEPA, both subjects | self-supervised on this ECoG | 4–200 Hz | 0.924 | +0.646 |
| **ECoG-JEPA, sub-02 only** | **never saw this brain** | 4–200 Hz | **0.933** | +0.634 |

Chance is 0.143. The EEG models are not bad models; they are the wrong instrument. Scaling
them would not help, because the signal is filtered away before their first layer.

**2. A transformer pretrained on ~12 minutes of this dataset's own LFP beats the
literature's hand-derived features.** ECoG-JEPA is 4.8 M parameters trained by masked
spectrogram reconstruction on the continuous recording, with no stimulus labels and no event
file. It reaches 0.924 against 0.895 for a feature set built directly from the published
model of these signals — and its representational geometry matches DINOv2's roughly four
times more closely.

**3. The self-supervised encoder transfers to a brain it has never seen.** The last row
above was pretrained *exclusively* on subject 2 — other hemisphere, 61 electrodes instead of
110, 1526 Hz instead of 3052 Hz — and never touched subject 1's recordings. Scoring subject
1 it reaches 0.933, against 0.924 for the version that did train on both: a two-trial
difference out of 210, i.e. indistinguishable. Whatever masked spectrogram reconstruction
learns about intracranial field potentials is not tied to one person's electrode layout.

**4. Contrastive alignment does not beat ridge regression — or lose to it.** Run on
identical folds and features, the CLIP-style two-tower head lands within ±0.05 of ridge on
2-way identification, winning for two encoders and losing for two. Worth stating plainly
because the expectation going in was that ridge would win outright at n ≈ 175 training
trials. It did not; the two are level. (They are also not perfectly comparable: ridge
predicts into DINOv2's own space, while the contrastive head is scored inside a 128-d space
trained to make matching easy.)

**5. DINOv2 explains no neural variance beyond a Gabor filter bank.** Partial RSA on
subject 1: DINOv2 given Gabor, ρ = +0.015 (*p* = 0.49); Gabor given DINOv2, ρ = +0.084
(*p* = 0.003). For gratings and 1/f noise, a hand-built oriented-energy model of V1 captures
the neural geometry at least as well as an 86 M-parameter self-supervised ViT. That is what
the control was there to find out.

**6. Category decoding is near ceiling; individual-image identification is at chance, and
we can say exactly why.** Every grating in this set is at the same orientation, so within a
condition the 30 exemplars differ in **spatial phase alone**. Decoding phase from the
cortical surface is reliably above chance but very weak (circular *r* = 0.25, *p* = 0.007),
which is what a phase-invariant broadband signal predicts. Within-condition top-1 retrieval
is 0.024–0.029 against a chance level of 0.033. The measured noise ceiling explains why:
broadband gamma on the most responsive electrodes replicates across repeat presentations of
the same image at only *r* = 0.22 (95 % CI [0.11, 0.34]) for single trials. Overall top-5 over all 210 images looks
impressive at 0.167, but that number is carried almost entirely by getting the category
right.

---

## Two things on the site you can use

**[Predict a response](https://neurolink-ecru.vercel.app/analyse.html)** takes any picture you
drop in, measures it with a 2-D FFT in your browser, and predicts what visual cortex would do.
It is aimed at somebody choosing stimuli for an experiment who wants to know whether they will
drive gamma before booking time. Cross-validated accuracy is printed beside every prediction
(narrowband gamma *r* = 0.957, gamma peak *r* = 0.814, broadband *r* = 0.678) and the page
warns when your picture sits outside the range the model was fitted on. Nothing is uploaded.

**[Run the decoder](https://neurolink-ecru.vercel.app/decode.html)** ships the real fitted
weights and runs them on real held-out trials in the browser. It reproduces the Python result
exactly: 0.8952 both sides, to four decimal places.

## What it produces

- Replication of the Hermes et al. gamma dissociation as a **hard gate** before any decoding
  claim (gratings drive a narrowband 30–80 Hz bump, noise does not; peak frequency climbs
  with spatial frequency)
- Parametric **reconstruction**: decode spatial frequency, noise exponent, contrast and
  phase, then re-render the image from those numbers
- **Brain TV**: a video decoding the continuous recording every 50 ms with no trial timing,
  rendered from a held-out stretch
- Per-electrode contribution rendered on each subject's **pial surface**
- A published results **website** driven entirely by `results.json`

## Reproducing

```bash
make setup          # uv venv + dependencies
bash scripts/fetch_data.sh   # OpenNeuro ds005953 into the parent directory
make all            # 01 -> 09; writes site/public/results.json and every figure
make test           # 24 tests: parameter recovery, split integrity, metric endpoints
make serve          # browse the site locally
```

The repo expects the BIDS dataset in its **parent directory** (that is the BIDS root). The
data is CC0 but ~650 MB, so it is fetched rather than vendored.

## Layout

```
src/neurolink/
  bids_io.py preprocess.py paths.py
  stimuli/    params.py     recover generative parameters from pixels
              dino.py       DINOv2 image latents
              v1model.py    log-Gabor V1 energy model (the control)
  neural/     spectral.py   Encoder A: Hermes broadband/narrowband decomposition
              foundation.py Encoder B: SignalJEPA, CBraMod
              ecogjepa.py   Encoder C: our masked-spectrogram transformer
  align/      ridge.py contrastive.py splits.py evaluate.py dataset.py
  recon/      parametric.py braintv.py
  viz/        figures.py brain3d.py
scripts/      01_stimuli … 09_report
site/public/  static site (deployed on Vercel)
```

## How the evaluation avoids fooling itself

- **Splits are grouped by stimulus image.** A test image never appears in training, and
  every fold asserts this programmatically (`Split.check`) rather than trusting the splitter.
- **PCA and scaling are fitted inside each training fold**, never on the full dataset.
- **Every headline number carries a permutation null** (1000 shuffles) and a bootstrap 95 %
  CI, and chance levels are stated rather than assumed.
- **Bad channels are the union** of the dataset's curated list and a data-driven pass. In
  subject 1 the automatic pass independently recovered the curators' 7 channels exactly.
- **Retrieval is reported within-condition as well as overall**, because the overall number
  cannot distinguish "identified the image" from "identified the category".

Two bugs found and fixed during development are worth naming, because both would have
produced confidently wrong results:

- The bandpass for the pretrained EEG models was built in transfer-function form, which at
  a normalised cutoff of ~2 × 10⁻⁴ is numerically unstable — it amplified the signal by a
  factor of 10¹⁰⁵ rather than filtering it. Switching to second-order sections fixed it.
  Left in place, it would have handed those models garbage and made the headline comparison
  meaningless.
- The initial bad-channel detector used an absolute threshold on peak amplitude and rejected
  33 of 118 channels; over 233 s of heavy-tailed neural data a 12–18 MAD peak is ordinary.
  Scoring each channel against the array's own distribution instead brought it in line with
  the curated list.

## Limitations

Two subjects, and subject 2's array had 35 of 96 contacts rejected by the curators — the
effects are much weaker there and are reported separately rather than pooled. The stimuli
are gratings and 1/f noise, not natural images; nothing here shows these methods generalise
to naturalistic vision. Reconstruction is parametric, appropriate to this stimulus family
but not a general image-reconstruction method. The pretrained EEG models are evaluated
outside their design envelope — different modality, different sample rate, and a 500 ms
response where SignalJEPA structurally requires a ≥1.5 s window; their showing here is a
statement about fit to this problem, not about their quality on scalp EEG.

## Links

- Results site: <https://neurolink-ecru.vercel.app>
- Repository: <https://github.com/aaryavvatts-lab/neurolink>

## Data

OpenNeuro [ds005953](https://openneuro.org/datasets/ds005953) — Hermes, Miller, Wandell &
Winawer. Released CC0. This is an independent reanalysis.

- Hermes D, Miller KJ, Wandell BA, Winawer J (2015). Stimulus dependence of gamma
  oscillations in human visual cortex. *Cerebral Cortex* 25(9):2951–9.
- Hermes D, Petridou N, Kay KN, Winawer J (2019). An image-computable model for the stimulus
  selectivity of gamma oscillations. *eLife* 8:e47035.
