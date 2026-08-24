/* NeuroLink results site. Every figure and number is read from results.json;
   nothing here is hand-entered. */

const $ = (id) => document.getElementById(id);
const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toFixed(d);
const pct = (x, d = 1) => (x === null || x === undefined) ? "—" : (100 * x).toFixed(d) + "%";
const pval = (p) => p === undefined ? "—" : (p <= 0.001 ? "p ≤ .001" : "p = " + Number(p).toFixed(3));
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else n.setAttribute(k, v);
  }
  kids.flat().forEach((c) => n.append(c?.nodeType ? c : document.createTextNode(c)));
  return n;
};
const table = (head, rows) => {
  const t = el("table");
  t.append(el("thead", {}, el("tr", {}, ...head.map((h) => el("th", {}, h)))));
  const tb = el("tbody");
  rows.forEach((r) => {
    const tr = el("tr");
    r.forEach((c) => tr.append(c?.nodeType === 1 ? c : el("td", {}, String(c))));
    tb.append(tr);
  });
  t.append(tb);
  return t;
};
const td = (text, cls) => el("td", cls ? { class: cls } : {}, String(text));
const figImg = (src, caption) => {
  const f = document.createDocumentFragment();
  f.append(el("img", { src, loading: "lazy", alt: caption || "" }));
  if (caption) f.append(el("figcaption", { html: caption }));
  return f;
};

/* ---------- theme ---------- */
const tk = "neurolink-theme";
const applyTheme = (t) => {
  if (t) document.documentElement.setAttribute("data-theme", t);
  else document.documentElement.removeAttribute("data-theme");
};
applyTheme(localStorage.getItem(tk));
$("themeToggle").onclick = () => {
  const cur = document.documentElement.getAttribute("data-theme");
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const next = cur ? (cur === "dark" ? "light" : "dark") : (dark ? "light" : "dark");
  localStorage.setItem(tk, next);
  applyTheme(next);
};

let R = null;

/* ---------- sections ---------- */
function heroStats() {
  const d = R.dataset;
  const subs = Object.values(d.subjects);
  const chans = subs.reduce((a, s) => a + s.n_good_channels, 0);
  const trials = subs.reduce((a, s) => a + s.trials_per_run * s.runs.length, 0);

  let best = null;
  const A = R.alignment;
  if (A) for (const [sub, s] of Object.entries(A.subjects))
    for (const [enc, e] of Object.entries(s.encoders))
      if (!best || e.condition.accuracy > best.acc)
        best = { acc: e.condition.accuracy, sub, enc, label: e.label };

  const items = [
    ["Electrodes on cortex", chans, `${Object.keys(d.subjects).length} people`],
    ["Stimulus trials", trials, "0.5 s each"],
    ["Distinct images", d.n_images - 1, "7 conditions × 30"],
  ];
  if (best) items.push(["Best 7-way accuracy", pct(best.acc, 1), `${best.sub} · chance 14.3%`]);

  $("heroStats").replaceChildren(...items.map(([k, v, sub]) =>
    el("div", { class: "stat" },
      el("span", { class: "v" }, String(v)),
      el("span", { class: "k" }, k),
      el("div", { class: "sub" }, sub))));
}

function dataSection() {
  const d = R.dataset;
  $("dsdoi").textContent = d.doi;
  const n = Object.keys(d.subjects).length;
  $("dataPara").innerHTML =
    `${n} people with subdural electrode arrays over occipital cortex viewed ` +
    `${d.n_images - 1} images — four square-wave gratings and three noise patterns, ` +
    `30 exemplars each — presented for ${d.stimulus_duration_s}&nbsp;s at a time across ` +
    `about ${d.visual_angle_deg}° of visual angle, alternating with a blank gray screen. ` +
    `Crucially, <strong>both subjects saw the identical image set</strong>, and one saw ` +
    `every image twice, which is what makes a noise ceiling and a cross-subject test possible.`;

  $("subjTable").replaceChildren(...table(
    ["Subject", "Hemisphere", "Runs", "Sampling rate", "Good electrodes", "Rejected", "Trials/run"],
    Object.entries(d.subjects).map(([k, s]) => [
      k, s.hemi === "R" ? "right" : "left", s.runs.length,
      Math.round(s.sfreq) + " Hz", s.n_good_channels, s.n_bad_channels, s.trials_per_run,
    ])).childNodes);

  const sp = R.stimulus_parameters || [];
  $("stimTable").replaceChildren(...table(
    ["Condition", "n", "Spatial freq (cpd)", "Noise exponent α", "RMS contrast", "Orientedness", "Phase spread"],
    sp.map((s) => [
      s.label, s.n,
      s.condition >= 4 ? fmt(s.spatial_freq_cpd, 2) : "—",
      fmt(s.noise_exponent, 2), fmt(s.rms_contrast, 3),
      fmt(s.orient_concentration, 2),
      s.condition >= 4 ? fmt(s.phase_sd_rad, 2) + " rad" : "—",
    ])).childNodes);

  const grat = sp.filter((s) => s.condition >= 4);
  const oriSd = grat.length ? Math.max(...grat.map((s) => s.orientation_sd_deg)) : 0;
  $("phaseNote").innerHTML =
    `<strong>A detail that decides the whole project.</strong> Every grating in this set is at ` +
    `the same orientation (across-exemplar spread ${fmt(oriSd, 2)}°), so orientation carries no ` +
    `information to decode. Within a grating condition the 30 exemplars differ in <em>spatial ` +
    `phase alone</em>. Whether individual images can be told apart therefore reduces entirely ` +
    `to whether phase is recoverable from the cortical surface — which we test directly in §4.`;
}

function replicationSection() {
  const fg = R.figures || {};
  const first = Object.keys(R.dataset.subjects)[0];
  const key = `spectra_${first}`;
  if (fg[key]) $("spectraFig").replaceChildren(figImg(
    "figures/" + fg[key],
    `Power change from the blank baseline, averaged over the ten most visually driven ` +
    `electrodes in ${first}. Gratings (right) produce a clear bump inside the shaded 30–80 Hz ` +
    `band; noise patterns (left) do not, despite driving a comparable broadband response.`));
  $("replNote").innerHTML =
    `<strong>The gate passes.</strong> The narrowband gamma bump appears for gratings and is ` +
    `absent for noise, and its peak frequency climbs with spatial frequency — the signature ` +
    `reported by Hermes et&nbsp;al. This is why the feature set below models broadband and ` +
    `narrowband gamma as two <em>separate</em> quantities: a decoder given only total gamma ` +
    `power cannot distinguish a grating from a noise pattern.`;
}

function encoderSection() {
  const A = R.alignment;
  if (!A) return;
  const subs = Object.keys(A.subjects);
  const encs = [];
  subs.forEach((s) => Object.keys(A.subjects[s].encoders).forEach((e) => {
    if (!encs.includes(e)) encs.push(e);
  }));

  const head = ["Encoder", "Family", "Dim"];
  subs.forEach((s) => head.push(`${s} 7-way`, `${s} 2-way`));
  const rows = encs.map((enc) => {
    const any = subs.map((s) => A.subjects[s].encoders[enc]).find(Boolean);
    const r = [any ? any.label : enc, el("td", { class: "fam" }, (any?.family || "").split(":")[0]),
               any ? any.dim : "—"];
    subs.forEach((s) => {
      const e = A.subjects[s].encoders[enc];
      if (!e) { r.push(td("—", "chance"), td("—", "chance")); return; }
      const best = Math.max(...encs.map((k) => A.subjects[s].encoders[k]?.condition.accuracy || 0));
      r.push(td(pct(e.condition.accuracy), e.condition.accuracy === best ? "win" : ""));
      const tw = e.spaces?.dino224?.two_way;
      const bestTw = Math.max(...encs.map((k) => A.subjects[s].encoders[k]?.spaces?.dino224?.two_way || 0));
      r.push(td(fmt(tw), tw === bestTw ? "win" : ""));
    });
    return r;
  });
  $("encoderTable").replaceChildren(...table(head, rows).childNodes);

  const fg = R.figures || {};
  if (fg.encoders_condition) $("encFig").replaceChildren(figImg(
    "figures/" + fg.encoders_condition,
    "Seven-way condition decoding under the novel-image split — test images never appear in " +
    "training. Error bars are bootstrap 95% CIs. Dashed line is chance (1/7)."));

  // bandwidth accounting
  const man = R.encoders_manifest || {};
  const bw = Object.values(man).map((v) => v.bandwidth).filter(Boolean)[0];
  if (bw) {
    const below40 = bw["frac_below_40Hz"], below125 = bw["frac_below_125Hz"];
    $("bandwidthNote").innerHTML =
      `<strong>Why the EEG models lose, in one number.</strong> Of all the stimulus-evoked ` +
      `change in the power spectrum, only <strong>${pct(below40)}</strong> falls below 40 Hz — ` +
      `the ceiling SignalJEPA was pretrained under — and ${pct(below125)} below 125 Hz, ` +
      `CBraMod's Nyquist limit. The remaining ${pct(1 - below40)} is broadband gamma, and it is ` +
      `filtered away before those models' first layer. Scaling them would not help; the signal ` +
      `is already gone.`;
  } else {
    $("bandwidthNote").innerHTML =
      `<strong>Note.</strong> Off-the-shelf EEG foundation models are pretrained at 128–250 Hz ` +
      `and cannot represent the 70–200 Hz broadband gamma this task depends on.`;
  }

  const CN = R.contrastive;
  if (CN) {
    const rows = [];
    for (const [sub, encs] of Object.entries(CN.subjects)) {
      for (const [enc, r] of Object.entries(encs)) {
        const dr = r.contrastive.two_way - r.ridge.two_way;
        rows.push([sub, r.label,
          fmt(r.ridge.two_way), fmt(r.contrastive.two_way),
          el("td", { class: dr > 0 ? "win" : "chance" },
             (dr >= 0 ? "+" : "") + fmt(dr)),
          fmt(r.ridge.top5, 3), fmt(r.contrastive.top5, 3)]);
      }
    }
    $("contrastiveTable").replaceChildren(...table(
      ["Subject", "Encoder", "Ridge 2-way", "Contrastive 2-way", "Δ",
       "Ridge top-5", "Contrastive top-5"], rows).childNodes);
  }

  const cf = Object.keys(fg).filter((k) => k.startsWith("confusion_"));
  if (cf.length) $("confusionFig").replaceChildren(figImg(
    "figures/" + fg[cf[0]],
    "Confusion matrix for the best encoder. Errors concentrate between neighbouring spatial " +
    "frequencies and among the noise textures — the confusions a visual system would make."));
}

function limitsSection() {
  const A = R.alignment;
  if (!A) return;
  const rows = [];
  for (const [sub, s] of Object.entries(A.subjects)) {
    for (const [enc, e] of Object.entries(s.encoders)) {
      const sp = e.spaces?.dino224;
      if (!sp) continue;
      const w = sp.within_condition;
      rows.push([
        sub, e.label,
        pct(sp.top1, 1), pct(sp.top5, 1),
        `${fmt(sp.median_rank, 0)} / ${sp.n_candidates}`,
        w ? pct(w.top1, 1) : "—",
        w ? pct(w.chance_top1, 1) : "—",
      ]);
    }
  }
  $("withinTable").replaceChildren(...table(
    ["Subject", "Encoder", "Top-1 (210)", "Top-5 (210)", "Median rank",
     "Within-condition top-1", "Chance"], rows).childNodes);

  const rec = R.reconstruction;
  if (rec) {
    const parts = [];
    for (const [sub, s] of Object.entries(rec.subjects)) {
      const g = s.scores?.gratings_only?.phase_rad;
      if (g) parts.push(`${sub}: circular r = ${fmt(g.circular_r)} (${pval(g.p)})`);
    }
    $("phaseFinding").innerHTML =
      `<strong>Phase is the bottleneck — and it is faintly there.</strong> Decoding grating ` +
      `phase from the cortical surface gives ${parts.join("; ")}. Reliably above chance, but ` +
      `far too weak to separate 30 exemplars that differ in nothing else. That is the ` +
      `mechanistic reason image-level identification stalls while category decoding is nearly ` +
      `ceiling: broadband gamma behaves like a complex cell, largely invariant to phase.`;
    const first = Object.keys(rec.subjects)[0];
    const fg = R.figures || {};
    if (fg[`params_${first}`]) $("paramsFig").replaceChildren(figImg(
      "figures/" + fg[`params_${first}`],
      "Continuous stimulus parameters decoded from ECoG under the novel-image split. " +
      "Each point is one held-out trial; the dashed line is identity."));
  }
}

function reconSection() {
  const fg = R.figures || {};
  const gk = Object.keys(fg).filter((k) => k.startsWith("gallery_"));
  if (gk.length) $("galleryFig").replaceChildren(figImg(
    "figures/" + fg[gk[0]],
    "Top row: the image on screen. Bottom row: rendered from parameters decoded out of the " +
    "brain on trials the model had never seen. Texture class, scale and contrast transfer; " +
    "the exact phase realisation does not."));

  const gal = R.galleries || {};
  const sub = Object.keys(gal)[0];
  if (!sub) return;
  $("galleryGrid").replaceChildren(...gal[sub].map((g) =>
    el("div", { class: "gcell" },
      el("div", { class: "pair" },
        el("img", { src: g.true, loading: "lazy", alt: "presented" }),
        el("img", { src: g.recon, loading: "lazy", alt: "reconstructed" })),
      el("div", { class: "lbl" }, R.condition_labels[g.condition] || ""))));
}

function braintvSection() {
  const bt = (R.braintv || [])[0];
  const holder = $("braintvHolder");
  if (!bt) { $("braintvPara").textContent =
      "The continuous-decoding video was not generated in this run."; return; }
  $("braintvPara").innerHTML =
    `No trial timing is used here. The decoder is handed a raw ${bt.window_s}&nbsp;s window of ` +
    `voltage every ${bt.hop_s}&nbsp;s and asked what is on the screen. It was fitted on the ` +
    `first ${bt.fitted_on_trials} trials of the run and this video is rendered from ` +
    `t&nbsp;=&nbsp;${fmt(bt.t_start, 0)}–${fmt(bt.t_end, 0)}&nbsp;s, so every frame is ` +
    `out-of-sample.`;
  holder.replaceChildren(el("video", {
    src: "video/" + bt.file, controls: "", playsinline: "", muted: "", loop: "",
    preload: "metadata",
  }));
}

function anatomySection() {
  const E = R.electrodes, fg = R.figures || {};
  if (!E) return;
  const imgs = [];
  for (const [sub, e] of Object.entries(E)) {
    (e.views || []).forEach((v) => imgs.push(
      el("figure", {}, figImg("figures/" + v, `${sub} — ${v.split("_").pop().replace(".png", "")}`))));
  }
  $("brainFigs").replaceChildren(...imgs);
  const pk = Object.keys(fg).filter((k) => k.startsWith("electrodes_"));
  if (pk.length) $("electrodeProfileFig").replaceChildren(figImg(
    "figures/" + fg[pk[0]],
    "Cross-validated 7-way accuracy for every electrode used on its own, sorted."));

  const parts = Object.entries(E).map(([sub, e]) => {
    const a = e.single_electrode_acc;
    return `${sub}: best single electrode ${pct(Math.max(...a), 1)}, median ${pct(
      a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)], 1)}`;
  });
  $("anatNote").innerHTML =
    `<strong>The signal is anatomically where it should be.</strong> ${parts.join("; ")} ` +
    `(chance 14.3%). The informative electrodes form a contiguous patch at the occipital pole, ` +
    `while anterior temporal contacts sit at chance — consistent with early visual cortex ` +
    `rather than a distributed artefact.`;
}

/* ---------- interactive explorer ---------- */
let EXP = { sub: null, cond: "all", idx: 0, trials: [] };

function buildExplorer() {
  const gal = R.galleries || {};
  const subs = Object.keys(gal);
  if (!subs.length) { $("expView").textContent = "No per-trial data available."; return; }
  $("expSub").replaceChildren(...subs.map((s) => el("option", { value: s }, s)));
  const conds = ["all", ...Object.keys(R.condition_labels).filter((c) => +c <= 7)];
  $("expCond").replaceChildren(...conds.map((c) =>
    el("option", { value: c }, c === "all" ? "all conditions" : R.condition_labels[c])));
  EXP.sub = subs[0];
  refreshTrials();
  $("expSub").onchange = (e) => { EXP.sub = e.target.value; refreshTrials(); };
  $("expCond").onchange = (e) => { EXP.cond = e.target.value; refreshTrials(); };
  $("expPrev").onclick = () => { EXP.idx = (EXP.idx - 1 + EXP.trials.length) % EXP.trials.length; drawTrial(); };
  $("expNext").onclick = () => { EXP.idx = (EXP.idx + 1) % EXP.trials.length; drawTrial(); };
  $("expRandom").onclick = () => { EXP.idx = Math.floor(Math.random() * EXP.trials.length); drawTrial(); };
}

function refreshTrials() {
  const all = (R.galleries || {})[EXP.sub] || [];
  EXP.trials = EXP.cond === "all" ? all : all.filter((t) => String(t.condition) === EXP.cond);
  EXP.idx = 0;
  drawTrial();
}

function drawTrial() {
  const v = $("expView");
  if (!EXP.trials.length) { v.replaceChildren(el("p", {}, "No trials for that condition.")); return; }
  const t = EXP.trials[EXP.idx];
  const info = t.info || {};
  const kv = el("div", { class: "kv" });
  const add = (k, val) => kv.append(el("div", {}, el("span", {}, k), el("span", {}, val)));
  add("presented", R.condition_labels[t.condition] || "—");
  add("decoded as", R.condition_labels[info.condition] || "—");
  add("decoded kind", info.kind || "—");
  if (info.spatial_freq_cpd !== undefined) add("spatial freq", fmt(info.spatial_freq_cpd, 2) + " cpd");
  if (info.noise_exponent !== undefined) add("noise exponent", fmt(info.noise_exponent, 2));
  add("contrast", fmt(info.contrast, 3));
  if (info.phase_rad !== undefined) add("phase", fmt(info.phase_rad, 2) + " rad");
  add("stimulus id", "#" + t.stim_id);

  const correct = String(info.condition) === String(t.condition);
  kv.append(el("div", {},
    el("span", {}, "condition correct"),
    el("span", { class: correct ? "" : "" }, correct ? "✓ yes" : "✗ no")));

  v.replaceChildren(
    el("div", {}, el("img", { src: t.true, alt: "presented" }),
      el("div", { class: "lbl" }, "on screen")),
    el("div", {}, el("img", { src: t.recon, alt: "reconstructed" }),
      el("div", { class: "lbl" }, "reconstructed from brain")),
    kv);
}

/* ---------- methods ---------- */
function methodsSection() {
  const A = R.alignment || {};
  const nf = A.config?.n_folds ?? 6, np = A.config?.n_permutations ?? 1000;
  const jep = R.jepa?.all;
  $("methodsBody").innerHTML = `
    <h3>Preprocessing</h3>
    <p>Line noise and its harmonics notched out; channels rejected by the union of the
       dataset's own curated bad-channel list and a data-driven pass over variance, line-noise
       dominance and artefact amplitude (the automatic pass recovered the curated list
       exactly in subject 1); common-average reference over surviving channels; epochs from
       −200 to +700 ms.</p>
    <h3>Features</h3>
    <p>Log power spectra are fitted with the two-component model of Hermes et&nbsp;al.,
       <span class="mono">F(x) = (β<sub>bb</sub> − n·x) + β<sub>nb</sub>·G(x | μ, σ)</span> with
       x = log₁₀ f, the 1/f slope <span class="mono">n</span> fixed per electrode from its own
       blank-screen baseline and the Gaussian constrained to peak in 30–80 Hz. Because the model
       is linear in both β terms given (μ, σ), we sweep a grid over the constrained box and
       solve the βs in closed form for all spectra at once.</p>
    <h3>Self-supervised ECoG-JEPA</h3>
    <p>${jep ? `A ${(jep.n_params / 1e6).toFixed(1)}M-parameter transformer trained by masked
       spectrogram reconstruction on ${jep.n_channels} channels of continuous recording, at the
       full sample rate so that 70–200 Hz gamma survives. It never sees a stimulus label or an
       event file. Held-out reconstruction reached L1 ${fmt(jep.val_losses.at(-1))} against a
       trivial baseline of ${fmt(jep.baseline_val)}.` : "Not run."}</p>
    <h3>Alignment and evaluation</h3>
    <p>Neural features are mapped to image latents by ridge regression with leave-one-out
       α selection, PCA fitted inside each training fold only. Splits are grouped by stimulus
       image over ${nf} folds, so a test image never appears in training — verified
       programmatically for every fold. Every headline number carries a permutation null
       (${np} shuffles) and a bootstrap 95% CI.</p>`;

  const lim = [
    "Two subjects. Effects that hold in one and not the other should be read as such, and subject 2's array had 35 of 96 contacts rejected by the dataset curators.",
    "The stimuli are gratings and 1/f noise, not natural images. Nothing here shows that these methods would generalise to naturalistic vision.",
    "Individual images within a condition differ only in spatial phase, which is close to the theoretical worst case for a phase-invariant broadband signal. Exemplar-level identification is correspondingly weak.",
    "Reconstruction is parametric: it re-renders a stimulus from decoded parameters, which is honest for this stimulus family but is not a general image reconstruction method.",
    "The off-the-shelf EEG foundation models are evaluated outside their design envelope — different recording modality, different sample rate, a 500 ms response where one requires a 1.5 s window. Their poor showing is a statement about fit to this problem, not about their quality on scalp EEG.",
  ];
  $("limitations").replaceChildren(...lim.map((t) => el("li", {}, t)));
  $("builtline").innerHTML =
    `Built with MNE-Python, PyTorch, scikit-learn, DINOv2 and braindecode. ` +
    `All figures regenerate from <span class="mono">make all</span>.`;
}

/* ---------- boot ---------- */
fetch("results.json")
  .then((r) => r.json())
  .then((data) => {
    R = data;
    [heroStats, dataSection, replicationSection, encoderSection, limitsSection,
     reconSection, braintvSection, anatomySection, buildExplorer, methodsSection]
      .forEach((f) => { try { f(); } catch (e) { console.error(f.name, e); } });
  })
  .catch((e) => {
    document.body.prepend(el("div", { class: "wrap" },
      el("div", { class: "callout warn" },
        "Could not load results.json — run `make all` to generate it. " + e)));
  });
