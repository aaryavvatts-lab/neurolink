/* Measure an image the same way the Python pipeline does, but in the browser.

   The five numbers below are what the forward model takes as input. They are
   computed here with the same definitions used on the training images, so a
   picture you drop in is treated exactly like one of the originals:

     spatial frequency     peak of the radial power spectrum, in cycles per degree
     orientation strength  how concentrated the energy is at one angle, 0 to 1
     noise exponent        slope of the log power spectrum against log frequency
     RMS contrast          standard deviation of the greyscale image
     mean luminance        its mean

   The transform is a plain radix-2 FFT on a 512 by 512 crop, matching the
   Python pipeline exactly. An earlier version used 256 to stay quick, but
   shrinking that far throws detail away: the falloff exponent came out 0.6 to
   1.1 too steep and the contrast of fine noise collapsed from 0.051 to 0.013.
   The model was fitted on the Python measurements, so the browser has to
   measure the same way or it is answering a different question.
*/

const N = 512;

function fft1d(re, im, inverse = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inverse ? 2 : -2) * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
}

function fft2d(gray) {
  const re = new Float64Array(N * N), im = new Float64Array(N * N);
  re.set(gray);
  const rr = new Float64Array(N), ri = new Float64Array(N);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) { rr[x] = re[y * N + x]; ri[x] = im[y * N + x]; }
    fft1d(rr, ri);
    for (let x = 0; x < N; x++) { re[y * N + x] = rr[x]; im[y * N + x] = ri[x]; }
  }
  for (let x = 0; x < N; x++) {
    for (let y = 0; y < N; y++) { rr[y] = re[y * N + x]; ri[y] = im[y * N + x]; }
    fft1d(rr, ri);
    for (let y = 0; y < N; y++) { re[y * N + x] = rr[y]; im[y * N + x] = ri[y]; }
  }
  return { re, im };
}

/** Pull a 256x256 greyscale array in 0..1 out of any drawable image source. */
export function toGray(source) {
  const c = document.createElement("canvas");
  c.width = N; c.height = N;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#808080"; ctx.fillRect(0, 0, N, N);
  const sw = source.naturalWidth || source.width, sh = source.naturalHeight || source.height;
  const side = Math.min(sw, sh);                    // centre crop keeps the aspect honest
  ctx.drawImage(source, (sw - side) / 2, (sh - side) / 2, side, side, 0, 0, N, N);
  const d = ctx.getImageData(0, 0, N, N).data;
  const g = new Float64Array(N * N);
  for (let i = 0; i < N * N; i++) {
    // Rec. 709 luma, then gamma is left alone: the originals are greyscale already.
    g[i] = (0.2126 * d[i * 4] + 0.7152 * d[i * 4 + 1] + 0.0722 * d[i * 4 + 2]) / 255;
  }
  return { gray: g, canvas: c };
}

export function analyse(gray, degrees = 25) {
  let mean = 0;
  for (let i = 0; i < gray.length; i++) mean += gray[i];
  mean /= gray.length;
  let sd = 0;
  for (let i = 0; i < gray.length; i++) sd += (gray[i] - mean) ** 2;
  sd = Math.sqrt(sd / gray.length);

  if (sd < 1e-5) {
    return { blank: true, meanLum: mean, rmsContrast: 0, spatialFreq: NaN,
             orientDeg: NaN, orientConc: 0, noiseExponent: NaN, spectrum: null, radial: null };
  }

  // Hann window in both directions, matching the Python side.
  const win = new Float64Array(N);
  for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1));
  const w = new Float64Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) w[y * N + x] = (gray[y * N + x] - mean) * win[y] * win[x];

  const { re, im } = fft2d(w);
  const power = new Float64Array(N * N);
  for (let i = 0; i < N * N; i++) power[i] = re[i] * re[i] + im[i] * im[i];

  const half = N / 2;
  const rSum = new Float64Array(half + 2), rCnt = new Float64Array(half + 2);
  let peakR = 1, peakVal = -1;
  const angBins = 36, angSum = new Float64Array(angBins);

  const at = (fy, fx) => power[((fy + N) % N) * N + ((fx + N) % N)];
  for (let fy = -half; fy < half; fy++) {
    for (let fx = -half; fx < half; fx++) {
      const r = Math.round(Math.hypot(fy, fx));
      if (r < 1 || r > half) continue;
      rSum[r] += at(fy, fx); rCnt[r]++;
    }
  }
  const radial = [];
  for (let r = 1; r <= half; r++) {
    const v = rCnt[r] ? rSum[r] / rCnt[r] : 0;
    radial.push({ r, power: v });
    if (v > peakVal) { peakVal = v; peakR = r; }
  }

  // Slope of log power against log frequency over the mid band.
  // Same window as the Python side (4 to 150 of a 512-wide transform).
  const lo = 4, hi = Math.min(150, half - 2);
  let sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
  for (let r = lo; r <= hi; r++) {
    const p = radial[r - 1].power;
    if (!(p > 0)) continue;
    const X = Math.log(r), Y = Math.log(p);
    sx += X; sy += Y; sxx += X * X; sxy += X * Y; n++;
  }
  const slope = n > 1 ? (n * sxy - sx * sy) / (n * sxx - sx * sx) : NaN;

  // Orientation from the ring around the dominant frequency.
  let cr = 0, ci = 0, tot = 0;
  for (let fy = -half; fy < half; fy++) {
    for (let fx = -half; fx < half; fx++) {
      const r = Math.hypot(fy, fx);
      if (r < peakR - 4 || r > peakR + 4) continue;
      const p = at(fy, fx);
      let th = Math.atan2(fy, fx); if (th < 0) th += Math.PI;
      cr += p * Math.cos(2 * th); ci += p * Math.sin(2 * th); tot += p;
      angSum[Math.min(angBins - 1, Math.floor(th / Math.PI * angBins))] += p;
    }
  }
  const conc = tot > 0 ? Math.hypot(cr, ci) / tot : 0;
  const orient = tot > 0 ? ((Math.atan2(ci / tot, cr / tot) / 2) + Math.PI) % Math.PI : NaN;

  // A small log-power picture of the spectrum, for display.
  const view = 128, spec = new Float64Array(view * view);
  let smin = Infinity, smax = -Infinity;
  for (let y = 0; y < view; y++) for (let x = 0; x < view; x++) {
    const fy = y - view / 2, fx = x - view / 2;
    const v = Math.log10(at(fy, fx) + 1e-12);
    spec[y * view + x] = v; if (v < smin) smin = v; if (v > smax) smax = v;
  }

  return {
    blank: false,
    meanLum: mean,
    rmsContrast: sd,
    cyclesPerImage: peakR,
    spatialFreq: peakR / degrees,
    orientDeg: (orient * 180) / Math.PI,
    orientConc: conc,
    noiseExponent: -slope,
    radial,
    orientHist: Array.from(angSum),
    spectrum: { data: spec, size: view, min: smin, max: smax },
  };
}

/** Apply the exported linear forward model to one set of measurements. */
export function predict(model, m) {
  const feats = [
    Math.log10(Math.max(m.spatialFreq || 1e-3, 1e-3)),
    m.orientConc,
    Number.isFinite(m.noiseExponent) ? m.noiseExponent : 0,
    m.rmsContrast,
    m.meanLum,
  ];
  const z = feats.map((v, i) => (v - model.x_mean[i]) / model.x_std[i]);
  return model.targets.map((t, k) => {
    let acc = model.intercept[k];
    for (let i = 0; i < z.length; i++) acc += model.coef[k][i] * z[i];
    return acc * model.y_std[k] + model.y_mean[k];
  });
}
