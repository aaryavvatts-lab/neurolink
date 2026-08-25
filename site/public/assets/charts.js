/* Small SVG chart set. Written here rather than pulled from a library so the
   marks, spacing and hover behaviour match the rest of the page exactly.

   Rules kept throughout: one y-axis per chart, recessive grid, thin marks,
   2px gaps between adjacent fills, a legend whenever more than one series is
   drawn, and a hover layer on every plot. */

import { el, cssVar } from "./shell.js";

const NS = "http://www.w3.org/2000/svg";
const s = (tag, attrs = {}) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) n.setAttribute(k, v);
  return n;
};

let tipEl = null;
function tip() {
  if (!tipEl) { tipEl = el("div", { class: "tip", role: "status", "aria-live": "polite" }); document.body.append(tipEl); }
  return tipEl;
}
export function showTip(x, y, html) {
  const t = tip(); t.innerHTML = html; t.classList.add("on");
  const r = t.getBoundingClientRect();
  let left = x + 14, top = y - r.height - 10;
  if (left + r.width > innerWidth - 8) left = x - r.width - 14;
  if (top < 8) top = y + 16;
  t.style.left = left + "px"; t.style.top = top + "px";
}
export function hideTip() { if (tipEl) tipEl.classList.remove("on"); }

function frame(w, h, m) {
  const svg = s("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" });
  const g = s("g", { transform: `translate(${m.l},${m.t})` });
  svg.append(g);
  return { svg, g, iw: w - m.l - m.r, ih: h - m.t - m.b };
}

function axes(g, iw, ih, xTicks, yTicks, xLabel, yLabel, opts = {}) {
  const grid = cssVar("--line"), axis = cssVar("--line-2"), muted = cssVar("--muted");
  yTicks.forEach((t) => {
    g.append(s("line", { x1: 0, x2: iw, y1: t.y, y2: t.y, stroke: grid, "stroke-width": 1 }));
    const tx = s("text", { x: -8, y: t.y + 4, "text-anchor": "end", fill: muted, "font-size": 11 });
    tx.textContent = t.label; g.append(tx);
  });
  xTicks.forEach((t) => {
    if (opts.xGrid) g.append(s("line", { x1: t.x, x2: t.x, y1: 0, y2: ih, stroke: grid, "stroke-width": 1 }));
    g.append(s("line", { x1: t.x, x2: t.x, y1: ih, y2: ih + 4, stroke: axis, "stroke-width": 1 }));
    const tx = s("text", { x: t.x, y: ih + 17, "text-anchor": "middle", fill: muted, "font-size": 11 });
    tx.textContent = t.label; g.append(tx);
  });
  g.append(s("line", { x1: 0, x2: iw, y1: ih, y2: ih, stroke: axis, "stroke-width": 1 }));
  if (xLabel) {
    const t = s("text", { x: iw / 2, y: ih + 36, "text-anchor": "middle", fill: muted, "font-size": 11.5 });
    t.textContent = xLabel; g.append(t);
  }
  if (yLabel) {
    const t = s("text", { transform: `translate(${-46},${ih / 2}) rotate(-90)`, "text-anchor": "middle", fill: muted, "font-size": 11.5 });
    t.textContent = yLabel; g.append(t);
  }
}

const niceTicks = (lo, hi, n = 5) => {
  const span = hi - lo || 1;
  const step = Math.pow(10, Math.floor(Math.log10(span / n)));
  const err = (span / n) / step;
  const mult = err >= 7.5 ? 10 : err >= 3 ? 5 : err >= 1.5 ? 2 : 1;
  const d = step * mult;
  const out = [];
  for (let v = Math.ceil(lo / d) * d; v <= hi + 1e-9; v += d) out.push(+v.toFixed(10));
  return out;
};

/* ---------- grouped bar chart, horizontal category axis ---------- */
export function barChart(host, { title, sub, series, chance, yLabel, fmt = (v) => v.toFixed(3), height = 260, max }) {
  host.replaceChildren();
  if (title) host.append(el("h4", {}, title));
  if (sub) host.append(el("p", { class: "sub" }, sub));

  const w = 760, m = { l: 58, r: 16, t: 10, b: 76 };
  const { svg, g, iw, ih } = frame(w, height, m);
  const hi = max ?? Math.max(...series.map((d) => d.value), chance ?? 0) * 1.16;
  const y = (v) => ih - (v / hi) * ih;
  const ticks = niceTicks(0, hi, 4).map((v) => ({ y: y(v), label: fmt(v) }));
  const bw = Math.min(74, (iw / series.length) - 12);

  axes(g, iw, ih, [], ticks, null, yLabel);

  series.forEach((d, i) => {
    const cx = (i + 0.5) * (iw / series.length);
    const x = cx - bw / 2, top = y(d.value), h = ih - top;
    const r = Math.min(4, h);
    const path = s("path", {
      d: `M${x},${ih} L${x},${top + r} Q${x},${top} ${x + r},${top} L${x + bw - r},${top} Q${x + bw},${top} ${x + bw},${top + r} L${x + bw},${ih} Z`,
      fill: d.color, stroke: cssVar("--surface"), "stroke-width": 2,
    });
    path.style.cursor = "pointer";
    path.addEventListener("pointerenter", (e) => showTip(e.clientX, e.clientY,
      `<b>${d.label}</b><div class="row"><span>${yLabel || "value"}</span><span>${fmt(d.value)}</span></div>` +
      (d.ci ? `<div class="row"><span>95% CI</span><span>${fmt(d.ci[0])} to ${fmt(d.ci[1])}</span></div>` : "") +
      (d.note ? `<div class="row"><span>${d.note}</span><span></span></div>` : "")));
    path.addEventListener("pointermove", (e) => showTip(e.clientX, e.clientY, tip().innerHTML));
    path.addEventListener("pointerleave", hideTip);
    g.append(path);

    if (d.ci) {
      g.append(s("line", { x1: cx, x2: cx, y1: y(d.ci[0]), y2: y(d.ci[1]), stroke: cssVar("--ink-2"), "stroke-width": 1.5 }));
      [d.ci[0], d.ci[1]].forEach((v) => g.append(s("line", { x1: cx - 5, x2: cx + 5, y1: y(v), y2: y(v), stroke: cssVar("--ink-2"), "stroke-width": 1.5 })));
    }
    // Direct label: identity never rests on colour alone. Sits above the error
    // bar cap when there is one, so the two never collide.
    const labelY = d.ci ? Math.min(top, y(d.ci[1])) - 9 : top - 7;
    const val = s("text", { x: cx, y: labelY, "text-anchor": "middle", fill: cssVar("--ink"), "font-size": 11.5, "font-weight": 600 });
    val.textContent = fmt(d.value); g.append(val);

    const words = String(d.label).split(" ");
    let line = "", lines = [];
    words.forEach((wd) => {
      if ((line + " " + wd).trim().length > 13) { lines.push(line.trim()); line = wd; }
      else line += " " + wd;
    });
    lines.push(line.trim());
    lines.slice(0, 3).forEach((ln, k) => {
      const t = s("text", { x: cx, y: ih + 16 + k * 12, "text-anchor": "middle", fill: cssVar("--ink-2"), "font-size": 10.5 });
      t.textContent = ln; g.append(t);
    });
  });

  if (chance !== undefined && chance !== null) {
    g.append(s("line", { x1: 0, x2: iw, y1: y(chance), y2: y(chance), stroke: cssVar("--ink-2"), "stroke-width": 1.5, "stroke-dasharray": "5 4" }));
    const t = s("text", { x: iw - 2, y: y(chance) - 6, "text-anchor": "end", fill: cssVar("--ink-2"), "font-size": 11 });
    t.textContent = "chance"; g.append(t);
  }
  host.append(svg);
  return svg;
}

/* ---------- multi-line chart with a shared crosshair ---------- */
export function lineChart(host, { title, sub, x, series, xLabel, yLabel, logX = false, height = 300, bands = [], yFmt = (v) => v.toFixed(2), xFmt = (v) => String(v) }) {
  host.replaceChildren();
  if (title) host.append(el("h4", {}, title));
  if (sub) host.append(el("p", { class: "sub" }, sub));

  const w = 760, m = { l: 62, r: 18, t: 12, b: 46 };
  const { svg, g, iw, ih } = frame(w, height, m);
  const xs = logX ? x.map(Math.log10) : x;
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const all = series.flatMap((sr) => sr.values).filter(Number.isFinite);
  let y0 = Math.min(...all), y1 = Math.max(...all);
  const pad = (y1 - y0) * 0.12 || 1; y0 -= pad; y1 += pad;
  const X = (v) => ((logX ? Math.log10(v) : v) - x0) / (x1 - x0) * iw;
  const Y = (v) => ih - (v - y0) / (y1 - y0) * ih;

  bands.forEach((b) => {
    g.append(s("rect", { x: X(b.from), y: 0, width: X(b.to) - X(b.from), height: ih, fill: cssVar("--surface-2") }));
    if (b.label) {
      const t = s("text", { x: (X(b.from) + X(b.to)) / 2, y: 12, "text-anchor": "middle", fill: cssVar("--muted"), "font-size": 10.5 });
      t.textContent = b.label; g.append(t);
    }
  });

  const xt = (logX ? [4, 10, 20, 40, 60, 100, 200] : niceTicks(x0, x1, 6))
    .filter((v) => v >= Math.min(...x) && v <= Math.max(...x))
    .map((v) => ({ x: X(v), label: xFmt(v) }));
  const yt = niceTicks(y0, y1, 5).map((v) => ({ y: Y(v), label: yFmt(v) }));
  axes(g, iw, ih, xt, yt, xLabel, yLabel);

  if (y0 < 0 && y1 > 0) g.append(s("line", { x1: 0, x2: iw, y1: Y(0), y2: Y(0), stroke: cssVar("--line-2"), "stroke-width": 1, "stroke-dasharray": "4 4" }));

  series.forEach((sr) => {
    const d = sr.values.map((v, i) => `${i ? "L" : "M"}${X(x[i]).toFixed(2)},${Y(v).toFixed(2)}`).join(" ");
    g.append(s("path", { d, fill: "none", stroke: sr.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
  });

  const cross = s("line", { y1: 0, y2: ih, stroke: cssVar("--ink-2"), "stroke-width": 1, opacity: 0 });
  g.append(cross);
  const dots = series.map((sr) => {
    const c = s("circle", { r: 4, fill: sr.color, stroke: cssVar("--surface"), "stroke-width": 2, opacity: 0 });
    g.append(c); return c;
  });
  const hit = s("rect", { x: 0, y: 0, width: iw, height: ih, fill: "transparent" });
  hit.style.cursor = "crosshair";
  hit.addEventListener("pointermove", (ev) => {
    const box = svg.getBoundingClientRect();
    const px = (ev.clientX - box.left) / box.width * w - m.l;
    let bi = 0, bd = Infinity;
    x.forEach((v, i) => { const d = Math.abs(X(v) - px); if (d < bd) { bd = d; bi = i; } });
    cross.setAttribute("x1", X(x[bi])); cross.setAttribute("x2", X(x[bi])); cross.setAttribute("opacity", .55);
    dots.forEach((c, k) => { c.setAttribute("cx", X(x[bi])); c.setAttribute("cy", Y(series[k].values[bi])); c.setAttribute("opacity", 1); });
    showTip(ev.clientX, ev.clientY,
      `<b>${xFmt(x[bi])}${xLabel ? " " + xLabel.replace(/\s*\(.*\)/, "") : ""}</b>` +
      series.map((sr) => `<div class="row"><span><i style="display:inline-block;width:.6rem;height:.6rem;background:${sr.color};border-radius:2px;margin-right:.3rem"></i>${sr.label}</span><span>${yFmt(sr.values[bi])}</span></div>`).join(""));
  });
  hit.addEventListener("pointerleave", () => {
    cross.setAttribute("opacity", 0); dots.forEach((c) => c.setAttribute("opacity", 0)); hideTip();
  });
  g.append(hit);
  host.append(svg);

  if (series.length > 1) {
    host.append(el("div", { class: "legend" },
      series.map((sr) => el("span", {},
        el("i", { style: `background:${sr.color}` }), sr.label))));
  }
  return svg;
}

/* ---------- scatter with identity line ---------- */
export function scatter(host, { title, sub, points, xLabel, yLabel, height = 320, identity = true, fmt = (v) => v.toFixed(2) }) {
  host.replaceChildren();
  if (title) host.append(el("h4", {}, title));
  if (sub) host.append(el("p", { class: "sub" }, sub));
  const w = 500, m = { l: 60, r: 16, t: 12, b: 48 };
  const { svg, g, iw, ih } = frame(w, height, m);
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  let lo = Math.min(...xs, ...ys), hi = Math.max(...xs, ...ys);
  const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
  const X = (v) => (v - lo) / (hi - lo) * iw, Y = (v) => ih - (v - lo) / (hi - lo) * ih;
  const t = niceTicks(lo, hi, 5);
  axes(g, iw, ih, t.map((v) => ({ x: X(v), label: fmt(v) })), t.map((v) => ({ y: Y(v), label: fmt(v) })), xLabel, yLabel);
  if (identity) g.append(s("line", { x1: X(lo), y1: Y(lo), x2: X(hi), y2: Y(hi), stroke: cssVar("--line-2"), "stroke-width": 1.5, "stroke-dasharray": "5 4" }));
  points.forEach((p) => {
    const c = s("circle", { cx: X(p.x), cy: Y(p.y), r: 4.5, fill: p.color, stroke: cssVar("--surface"), "stroke-width": 1.5, opacity: .92 });
    c.style.cursor = "pointer";
    c.addEventListener("pointerenter", (e) => showTip(e.clientX, e.clientY,
      `<b>${p.label || ""}</b><div class="row"><span>shown</span><span>${fmt(p.x)}</span></div><div class="row"><span>decoded</span><span>${fmt(p.y)}</span></div>`));
    c.addEventListener("pointerleave", hideTip);
    g.append(c);
  });
  host.append(svg);
  return svg;
}

/* ---------- confusion matrix, one hue light to dark ---------- */
export function matrix(host, { title, sub, values, labels, height = 400 }) {
  host.replaceChildren();
  if (title) host.append(el("h4", {}, title));
  if (sub) host.append(el("p", { class: "sub" }, sub));
  const n = labels.length, w = 620, m = { l: 132, r: 20, t: 10, b: 96 };
  const { svg, g, iw, ih } = frame(w, height, m);
  const cell = Math.min(iw / n, ih / n);
  const rows = values.map((r) => { const t = r.reduce((a, b) => a + b, 0) || 1; return r.map((v) => v / t); });
  const ramp = ["#eef4fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
  const col = (v) => ramp[Math.min(ramp.length - 1, Math.floor(v * ramp.length))];
  rows.forEach((r, i) => r.forEach((v, j) => {
    const x = j * cell, y = i * cell;
    const rect = s("rect", { x: x + 1, y: y + 1, width: cell - 2, height: cell - 2, rx: 3, fill: col(v) });
    rect.style.cursor = "pointer";
    rect.addEventListener("pointerenter", (e) => showTip(e.clientX, e.clientY,
      `<b>${labels[i]}</b><div class="row"><span>decoded as</span><span>${labels[j]}</span></div><div class="row"><span>share</span><span>${(v * 100).toFixed(0)}%</span></div><div class="row"><span>trials</span><span>${values[i][j]}</span></div>`));
    rect.addEventListener("pointerleave", hideTip);
    g.append(rect);
    if (v >= 0.08) {
      const t = s("text", { x: x + cell / 2, y: y + cell / 2 + 4, "text-anchor": "middle", "font-size": 10.5, fill: v > 0.55 ? "#fff" : "#0b0b0b" });
      t.textContent = Math.round(v * 100); g.append(t);
    }
  }));
  labels.forEach((lb, i) => {
    const t = s("text", { x: -8, y: i * cell + cell / 2 + 4, "text-anchor": "end", fill: cssVar("--ink-2"), "font-size": 10.5 });
    t.textContent = lb; g.append(t);
    const b = s("text", { transform: `translate(${i * cell + cell / 2},${n * cell + 12}) rotate(45)`, "text-anchor": "start", fill: cssVar("--ink-2"), "font-size": 10.5 });
    b.textContent = lb; g.append(b);
  });
  const yl = s("text", { transform: `translate(${-116},${n * cell / 2}) rotate(-90)`, "text-anchor": "middle", fill: cssVar("--muted"), "font-size": 11.5 });
  yl.textContent = "shown"; g.append(yl);
  host.append(svg);
  host.append(el("p", { class: "sub", style: "margin-top:.5rem" },
    "Numbers are the share of trials in each row, as a percentage. Darker means more."));
  return svg;
}

/* ---------- horizontal probability bars ---------- */
export function probBars(host, { items, height = 16 }) {
  host.replaceChildren();
  const max = Math.max(...items.map((i) => i.value), 1e-6);
  items.forEach((it) => {
    const row = el("div", { class: "bar", style: "display:grid;grid-template-columns:9.5rem 1fr 2.8rem;gap:.5rem;align-items:center;font-size:.8rem;margin:.22rem 0" },
      el("span", { style: it.highlight ? "font-weight:650" : "" }, it.label),
      el("span", { style: `background:var(--surface-2);border-radius:3px;height:${height}px;overflow:hidden;display:block` },
        el("span", { style: `display:block;height:100%;width:${(it.value / max * 100).toFixed(1)}%;background:${it.color}` })),
      el("span", { class: "mono", style: "text-align:right;font-variant-numeric:tabular-nums" }, (it.value * 100).toFixed(0) + "%"));
    host.append(row);
  });
}
