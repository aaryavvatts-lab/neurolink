/* Shared chrome: nav, footer, theme, cookie notice. No trackers anywhere. */
export const PAGES = [
  ["index.html", "Project"],
  ["data.html", "The data"],
  ["results.html", "Results"],
  ["analyse.html", "Predict a response"],
  ["decode.html", "Run the decoder"],
  ["brain.html", "3D brain"],
  ["reconstruct.html", "Reconstructions"],
  ["methods.html", "Methods"],
  ["references.html", "References"],
];

const THEME_KEY = "neurolink.theme";

export function applyTheme(t) {
  if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
  else document.documentElement.removeAttribute("data-theme");
}
applyTheme(localStorage.getItem(THEME_KEY));

export function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  kids.flat().forEach((c) => { if (c !== null && c !== undefined) n.append(c?.nodeType ? c : String(c)); });
  return n;
}

export function mountChrome(current) {
  const head = el("header", { class: "top" },
    el("div", { class: "wrap topbar" },
      el("a", { class: "brand", href: "index.html" }, "NeuroLink",
        el("span", {}, "Seeing through electrodes")),
      el("nav", { class: "main", "aria-label": "Main" },
        PAGES.map(([href, label]) =>
          el("a", { href, "aria-current": href === current ? "page" : null }, label))),
      el("button", {
        class: "theme-btn", type: "button", "aria-label": "Switch between light and dark",
        onclick: () => {
          const cur = document.documentElement.getAttribute("data-theme");
          const sysDark = matchMedia("(prefers-color-scheme: dark)").matches;
          const next = cur ? (cur === "dark" ? "light" : "dark") : (sysDark ? "light" : "dark");
          localStorage.setItem(THEME_KEY, next); applyTheme(next);
          document.dispatchEvent(new CustomEvent("themechange"));
        },
      }, "◐")));
  document.body.prepend(head);
  document.body.prepend(el("a", { class: "skip", href: "#main" }, "Skip to content"));

  const foot = el("footer", { class: "site" },
    el("div", { class: "wrap" },
      el("nav", { "aria-label": "Site information" },
        el("a", { href: "privacy.html" }, "Privacy"),
        el("a", { href: "terms.html" }, "Terms"),
        el("a", { href: "cookies.html" }, "Cookies"),
        el("a", { href: "accessibility.html" }, "Accessibility"),
        el("a", { href: "https://github.com/aaryavvatts-lab/neurolink", rel: "noopener" }, "Code on GitHub")),
      el("p", {}, "A student reanalysis of an open brain recording. Not a medical device, and not medical advice."),
      el("p", { class: "mono", style: "font-size:.78rem;color:var(--muted)" },
        "Data: OpenNeuro ds005953, released CC0 by Hermes, Miller, Wandell and Winawer.")));
  document.body.append(foot);

  cookieNotice();
}

function cookieNotice() {
  const KEY = "neurolink.cookieNotice";
  if (localStorage.getItem(KEY) === "seen") return;
  const bar = el("div", { class: "cookie", role: "region", "aria-label": "Storage notice" },
    el("div", { class: "wrap" },
      el("p", {}, "This site stores one thing on your device: whether you picked light or dark. " +
        "No cookies, no analytics, no tracking. ",
        el("a", { href: "cookies.html" }, "Read the details")),
      el("button", {
        class: "primary", type: "button",
        onclick: () => { localStorage.setItem(KEY, "seen"); bar.remove(); },
      }, "Got it")));
  document.body.append(bar);
}

export async function loadJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`could not load ${path} (${r.status})`);
  return r.json();
}

export const fmt = (x, d = 3) =>
  (x === null || x === undefined || Number.isNaN(x)) ? "n/a" : Number(x).toFixed(d);
export const pct = (x, d = 1) =>
  (x === null || x === undefined || Number.isNaN(x)) ? "n/a" : (100 * x).toFixed(d) + "%";
export const pval = (p) => p === undefined || p === null ? "n/a"
  : (p <= 0.001 ? "p ≤ .001" : "p = " + Number(p).toFixed(3));

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* Colour is assigned by condition, and the condition order is meaningful:
   the three noise patterns run one way in their exponent, the four gratings
   run the other way in spatial frequency. Two ordered ramps, not seven
   unrelated hues. */
export const CONDITION_COLOR = {
  1: "--noise-1", 2: "--noise-2", 3: "--noise-3",
  4: "--grat-1", 5: "--grat-2", 6: "--grat-3", 7: "--grat-4",
};
export const condColor = (c) => cssVar(CONDITION_COLOR[c] || "--muted");
export const FAMILY_COLOR = {
  "A: hand-crafted": "--famA",
  "B: off-the-shelf EEG foundation model": "--famB",
  "C: self-pretrained on this ECoG": "--famC",
};
export const famColor = (f) => cssVar(FAMILY_COLOR[f] || "--muted");
