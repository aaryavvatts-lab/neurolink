/* Draw a stimulus from a set of parameters, the same way the Python renderer does. */

export function drawGrating(ctx, size, { sfCpd, phase, contrast, degrees = 25, orientDeg = 0 }) {
  const im = ctx.createImageData(size, size);
  const th = (orientDeg * Math.PI) / 180;
  const ct = Math.cos(th), st = Math.sin(th);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const gx = ((x - size / 2) / size) * degrees;
      const gy = ((y - size / 2) / size) * degrees;
      const proj = gx * ct + gy * st;
      const w = Math.sin(2 * Math.PI * sfCpd * proj + phase);
      const v = Math.max(0, Math.min(1, 0.5 + contrast * Math.sign(w)));
      const p = (y * size + x) * 4, g = Math.round(v * 255);
      im.data[p] = g; im.data[p + 1] = g; im.data[p + 2] = g; im.data[p + 3] = 255;
    }
  }
  ctx.putImageData(im, 0, 0);
}

/* 1/f^alpha noise. Built by summing a modest number of random sinusoids rather
   than an inverse FFT: it gives the same power spectrum for display purposes and
   keeps the file small. The seed makes a given trial always draw the same way. */
export function drawNoise(ctx, size, { alpha, contrast, seed = 1 }) {
  let s = (seed * 2654435761) >>> 0;
  const rnd = () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  const field = new Float64Array(size * size);
  const comps = 220;
  for (let k = 0; k < comps; k++) {
    const f = Math.pow(2, 0.5 + rnd() * 5.5);                 // ~1.4 to ~90 cycles
    const amp = Math.pow(f, -alpha / 2);
    const ang = rnd() * Math.PI * 2, ph = rnd() * Math.PI * 2;
    const kx = (2 * Math.PI * f * Math.cos(ang)) / size;
    const ky = (2 * Math.PI * f * Math.sin(ang)) / size;
    for (let y = 0; y < size; y++) {
      const base = y * size, kyy = ky * y + ph;
      for (let x = 0; x < size; x++) field[base + x] += amp * Math.cos(kx * x + kyy);
    }
  }
  let mean = 0; for (let i = 0; i < field.length; i++) mean += field[i];
  mean /= field.length;
  let sd = 0; for (let i = 0; i < field.length; i++) sd += (field[i] - mean) ** 2;
  sd = Math.sqrt(sd / field.length) || 1;

  const im = ctx.createImageData(size, size);
  for (let i = 0; i < field.length; i++) {
    const v = Math.max(0, Math.min(1, 0.5 + ((field[i] - mean) / sd) * contrast));
    const p = i * 4, g = Math.round(v * 255);
    im.data[p] = g; im.data[p + 1] = g; im.data[p + 2] = g; im.data[p + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}

/** Run the exported linear decoder on one trial's features. */
export function runDecoder(model, trial) {
  const W = model.folds[model.fold_of[trial.i]].W;
  const b = model.folds[model.fold_of[trial.i]].b;
  const out = b.slice();
  for (let i = 0; i < trial.x.length; i++) {
    const xi = trial.x[i];
    if (xi === 0) continue;
    const row = W[i];
    for (let k = 0; k < out.length; k++) out[k] += xi * row[k];
  }
  return out;
}

export function decodedParams(model, vec, post) {
  const g = {};
  model.cols.forEach((c, i) => { g[c] = vec[i]; });
  const cond = post.indexOf(Math.max(...post)) + 1;
  const phase = Math.atan2(g.phase_sin ?? 0, g.phase_cos ?? 1);
  return {
    cond,
    isGrating: cond >= 4,
    sfCpd: Math.min(3, Math.max(0.05, Math.pow(10, g.log_spatial_freq ?? -0.8))),
    alpha: Math.min(4.5, Math.max(-0.5, g.noise_exponent ?? 2)),
    contrast: Math.min(0.5, Math.max(0.02, g.rms_contrast ?? 0.2)),
    conc: g.orient_concentration ?? 0,
    phase,
  };
}
