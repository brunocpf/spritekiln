/* SpriteKiln web UI — drives the SpriteKiln ComfyUI workflows through serve.py (same origin). */
"use strict";

// ---- constants -------------------------------------------------------------------------------
const GROUPS = [
  ["Generate", ["01", "02", "03", "19"]],
  ["Edit", ["04", "05", "21"]],
  ["Rotate & views", ["06", "07", "08"]],
  ["Sheets & animation", ["09", "09b", "10", "11", "16", "17", "18"]],
  ["Tiles", ["12", "13", "20"]],
  ["Tools", ["14", "15"]],
  ["HD 2D art", ["22", "23", "24", "25", "26"]],
  ["Rigging", ["27"]],
  ["Lighting", ["28", "28b"]],
];
// smooth (non-pixel) art: no pixelated rendering, no pixel zoom
const HD_GROUPS = new Set(["HD 2D art", "Rigging"]);
const isHD = id => /_hd_/.test(id || "") || GROUPS.some(([g, nums]) => HD_GROUPS.has(g) && nums.includes(wfNum(id || "")));
const SAMPLERS = new Set(["KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced"]);
const NODE_LABELS = {
  SK_StylePreset: "Style", SK_AnimationPreset: "Animation", SK_DirectionPrompts: "Directions",
  SK_Palette: "Palette", SK_Cleanup: "Cleanup & outline", SK_BackgroundRemove: "Background removal",
  SK_PixelPerfect: "Pixel grid", SK_ToCanvas: "Model canvas", SK_ColorRamp: "Colour ramps",
  SK_PaletteSwap: "Palette swap", SK_HueShift: "Hue shift", SK_RegionMask: "Region to change",
  SK_SliceSheet: "Sheet slicing", SK_Tileset: "Tileset", SK_Stabilize: "Stabilize frames",
  ImageFromBatch: "Frame range",
  SK_HDStylePreset: "Style", SK_HDCutout: "Cutout", SK_HDCanvas: "Model canvas", SK_PartPrompts: "Parts",
  SK_PartCutout: "Part cutout", SK_PartLayout: "Part layout", ImageScale: "Size",
  SK_SpriteNormalMap: "Normal map",
};
const EDIT_CLASSES = new Set(Object.keys(NODE_LABELS));
// PF nodes whose fields belong on the main form (the rest go under "Advanced" in AI workflows)
const MAIN_CLASSES = new Set(["SK_StylePreset", "SK_AnimationPreset", "SK_DirectionPrompts", "SK_RegionMask",
  "SK_ColorRamp", "SK_PaletteSwap", "SK_HueShift", "SK_HDStylePreset", "SK_PartPrompts", "ImageScale",
  "SK_SpriteNormalMap"]);
const ADV_INPUTS = {
  SK_StylePreset: ["model", "long_side", "extra"],
  SK_AnimationPreset: ["background"],
  SK_DirectionPrompts: ["prompt_style", "template", "background"],
  SK_HDStylePreset: ["extra"],
  SK_PartPrompts: ["completion_template"],
};
// classes where only some literal inputs are worth a field
const ONLY_INPUTS = {
  SK_HDCutout: ["out_long_side", "margin", "soft_edges", "remove_shadow"],
  SK_HDCanvas: ["width", "height", "padding_percent", "background"],
  SK_PartCutout: ["expand_percent"],
  ImageScale: ["width", "height"],
  SK_SpriteNormalMap: ["inflate", "radius", "detail", "detail_blur", "strength", "convention", "ai_weight"],
};
const PROMPTISH = new Set(["subject", "custom_motion", "template", "extra", "text", "value", "character"]);
const SEED_MAX = 2 ** 32;

// ---- small helpers ---------------------------------------------------------------------------
const $ = (s, root = document) => root.querySelector(s);
const PROPS = new Set(["value", "checked", "selected", "disabled", "open", "hidden", "multiple"]);
function el(tag, props, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
    else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
    else if (PROPS.has(k)) e[k] = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat(Infinity)) if (c != null && c !== false) e.append(c instanceof Node ? c : String(c));
  return e;
}
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage full or blocked */ } },
  del(k) { try { localStorage.removeItem(k); } catch { /* blocked */ } },
};
(function migrateStorage() {
  try {
    for (const k of Object.keys(localStorage)) {
      if (!k.startsWith("pf.")) continue;
      const nk = "sk." + k.slice(3);
      if (localStorage.getItem(nk) == null) {
        let v = localStorage.getItem(k);
        if (k === "pf.presets" || k === "pf.looks") v = v.replace(/"PF_/g, '"SK_');
        localStorage.setItem(nk, v);
      }
      localStorage.removeItem(k);
    }
  } catch { /* storage blocked: nothing to migrate */ }
})();
const humanize = s => { s = String(s).replace(/_/g, " ").trim(); return s.charAt(0).toUpperCase() + s.slice(1); };
const wfNum = id => id.split("_")[0];
const fmtTime = ms => {
  const s = Math.max(0, Math.round(ms / 1000));
  return s < 60 ? `${s}s` : s < 3600 ? `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`
    : `${Math.floor(s / 3600)}h${String(Math.floor(s / 60) % 60).padStart(2, "0")}`;
};
const fmtDur = ms => ms < 60000 ? `${(ms / 1000).toFixed(1)} s` : fmtTime(ms);
const byNumericId = (a, b) => (parseInt(a, 10) - parseInt(b, 10)) || String(a).localeCompare(String(b));
// node classes were renamed PF_* -> SK_*; the old IDs remain as aliases, so treat both as one
const canon = cls => String(cls).replace(/^PF_/, "SK_");
const isLink = v => Array.isArray(v) && v.length === 2 && typeof v[1] === "number";
const viewUrl = (filename, subfolder = "", type = "output") =>
  "/view?" + new URLSearchParams({ filename, subfolder, type });
function splitPath(p) { const i = p.lastIndexOf("/"); return i < 0 ? ["", p] : [p.slice(0, i), p.slice(i + 1)]; }
const randSeed = () => Math.floor(Math.random() * SEED_MAX);
function sessionId() {
  try {
    let id = sessionStorage.getItem("sk.client");
    if (!id) { id = crypto.randomUUID().replace(/-/g, ""); sessionStorage.setItem("sk.client", id); }
    return id;
  } catch { return Math.random().toString(16).slice(2) + Date.now().toString(16); }
}
const CLIENT_ID = sessionId();

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}
async function postJSON(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  let data = null;
  try { data = await r.json(); } catch { /* empty body */ }
  return { ok: r.ok, status: r.status, data };
}

// ---- app state -------------------------------------------------------------------------------
const state = {
  workflows: [],          // from /sk/workflows
  graphs: {},             // id -> api graph (latest fetched)
  wf: null,               // { id, meta, api, model }
  jobs: store.get("sk.jobs", []),
  zoom: store.get("sk.zoom", 0),
  checker: store.get("sk.checker", true),
  pendingImage: null,     // image name to drop into the next workflow's first LoadImage (Send to…)
  queueRemaining: null,
  activeDrop: null,
};
const objInfo = {};

async function loadObjInfo(classes, force = []) {
  const fetchInfo = async c => { try { return (await getJSON("/object_info/" + encodeURIComponent(c)))[c] || null; } catch { return null; } };
  await Promise.all([...new Set(classes)].filter(c => !(c in objInfo) || force.includes(c)).map(async c => {
    // a server still on the other prefix knows the node under its alias
    let info = await fetchInfo(c);
    if (!info && /^(SK|PF)_/.test(c)) info = await fetchInfo(c.replace(/^(SK|PF)_/, m => (m === "SK_" ? "PF_" : "SK_")));
    objInfo[c] = info;
    objInfo[canon(c)] ??= info;
  }));
}
function inputSpec(cls, name) {
  const inp = objInfo[cls]?.input;
  return inp?.required?.[name] || inp?.optional?.[name] || null;
}
function normSpec(spec, value) {
  if (!spec) {
    if (typeof value === "boolean") return { type: "BOOLEAN", opts: {} };
    if (typeof value === "number") return { type: Number.isInteger(value) ? "INT" : "FLOAT", opts: {} };
    return { type: "STRING", opts: { multiline: String(value).length > 60 } };
  }
  const [t, o = {}] = spec;
  if (Array.isArray(t)) return { type: "COMBO", options: t, opts: o };
  if (t === "COMBO") return { type: "COMBO", options: o.options || [], opts: o };
  return { type: t, opts: o };
}

// ---- graph analysis: which inputs become form fields ------------------------------------------
function ancestorClasses(api, nid, seen = new Set()) {
  const out = new Set();
  for (const v of Object.values(api[nid]?.inputs || {})) {
    if (!isLink(v) || seen.has(v[0])) continue;
    seen.add(v[0]);
    const src = api[v[0]];
    if (!src) continue;
    out.add(src.class_type);
    for (const c of ancestorClasses(api, v[0], seen)) out.add(c);
  }
  return out;
}

function nodeLabel(api, nid) {
  const n = api[nid];
  const title = n._meta?.title;
  if (title && title !== n.class_type) return title;
  if (canon(n.class_type) === "SK_SavePixelArt" && typeof n.inputs.filename_prefix === "string")
    return humanize(n.inputs.filename_prefix.split("/").pop());
  if (canon(n.class_type) === "SK_SaveParts" && typeof n.inputs.folder === "string")
    return `Rig parts: ${n.inputs.folder.split("/").pop()}`;
  return NODE_LABELS[canon(n.class_type)] || objInfo[n.class_type]?.display_name || n.class_type;
}

function describe(api) {
  const classes = Object.values(api).map(n => canon(n.class_type));
  const noAI = !classes.some(c => SAMPLERS.has(c));
  const unets = Object.values(api).map(n => String(n.inputs.unet_name || "")).join(" ").toLowerCase();
  const models = [];
  if (unets.includes("z_image")) models.push("Z-Image Turbo");
  if (unets.includes("flux")) models.push("FLUX.2 klein");
  if (unets.includes("qwen")) models.push("Qwen-Image-Edit");
  if (unets.includes("wan")) models.push("Wan 2.2");
  const video = classes.some(c => c.startsWith("Wan"));
  const eta = noAI ? "~2 s" : video && classes.includes("SK_DirectionPrompts") ? "~40 min"
    : video ? "~7–10 min" : "~15–120 s";
  return { noAI, models, eta };
}

function buildModel(api) {
  const { noAI } = describe(api);
  const ids = Object.keys(api).sort(byNumericId);
  const images = [], prompts = [], settings = [], mainNodes = [], advNodes = [];
  const seeds = [], fps = [];
  const labelCount = {};
  const nodes = {};
  for (const nid of ids) {
    const n = api[nid], cls = canon(n.class_type), title = n._meta?.title || n.class_type;
    const literal = Object.entries(n.inputs).filter(([k, v]) => !isLink(v) && (!ONLY_INPUTS[cls] || ONLY_INPUTS[cls].includes(k)));
    const field = (input, value, extra = {}) => ({ key: `${nid}.${input}`, nid, cls, input, value, ...extra });

    for (const [k, v] of literal)
      if ((k === "seed" || k === "noise_seed") && !cls.startsWith("SK_") && typeof v === "number") seeds.push({ nid, input: k, value: v });

    if (cls === "LoadImage") { images.push(field("image", n.inputs.image, { label: title === "LoadImage" ? "Image" : title, kind: "image" })); continue; }
    if (cls === "CLIPTextEncode" && typeof n.inputs.text === "string") {
      const f = field("text", n.inputs.text, { label: title, kind: "prompt" });
      if (/negative/i.test(title)) advNodes.push({ nid, label: title, cls, fields: [f] });
      else prompts.push(f);
      continue;
    }
    if (cls === "PrimitiveStringMultiline" || cls === "PrimitiveString") {
      if (typeof n.inputs.value === "string") prompts.push(field("value", n.inputs.value, { label: title, kind: cls === "PrimitiveString" ? "line" : "prompt" }));
      continue;
    }
    if (cls === "PrimitiveInt") {
      if (typeof n.inputs.value === "number") settings.push(field("value", n.inputs.value, { label: title }));
      continue;
    }
    if (cls === "SK_SavePixelArt") {
      if (n.inputs.as_animation === true && typeof n.inputs.fps === "number") fps.push({ nid, value: n.inputs.fps });
      continue;
    }
    if (!EDIT_CLASSES.has(cls) || !literal.length) continue;
    const fromUserImage = ancestorClasses(api, nid).has("LoadImage");
    if (cls === "ImageFromBatch" && !(fromUserImage && noAI)) continue;
    if (cls === "ImageScale" && title === n.class_type) continue;
    if (cls === "SK_Stabilize" && !fromUserImage) continue;
    const label = nodeLabel(api, nid);
    labelCount[label] = (labelCount[label] || 0) + 1;
    const node = { nid, cls, label, fields: literal.map(([k, v]) => field(k, v, { label: humanize(k) })) };
    nodes[nid] = node;
    const isMain = noAI || MAIN_CLASSES.has(cls) || (cls === "SK_SliceSheet" && fromUserImage);
    if (isMain) {
      const adv = ADV_INPUTS[cls] || [];
      const mainFields = node.fields.filter(f => !adv.includes(f.input));
      const advFields = node.fields.filter(f => adv.includes(f.input));
      if (mainFields.length) mainNodes.push({ ...node, fields: mainFields });
      if (advFields.length) advNodes.push({ ...node, fields: advFields, label: node.label + " — more" });
    } else advNodes.push(node);
  }
  // disambiguate identical labels: by the node's style ("Style · texture …") or a number ("Palette (2)")
  const seen = {};
  for (const node of [...mainNodes, ...advNodes]) {
    const more = node.label.endsWith(" — more");
    const base = node.label.replace(/ — more$/, "");
    if (labelCount[base] <= 1) continue;
    const style = api[node.nid].inputs.style;
    if (typeof style === "string") { node.label = `${base} · ${style}${more ? " — more" : ""}`; continue; }
    if (more) continue;
    seen[base] = (seen[base] || 0) + 1;
    if (seen[base] > 1) node.label = `${base} (${seen[base]})`;
  }
  return { images, prompts, settings, mainNodes, advNodes, seeds, fps, noAI };
}

// ---- field widgets ---------------------------------------------------------------------------
const HEX_RE = /#[0-9a-f]{6}\b/gi;

function fieldWidget(f, onChange) {
  const spec = normSpec(inputSpec(f.cls, f.input), f.value);
  const o = spec.opts || {};
  const id = `f-${f.nid}-${f.input}`;
  const tip = o.tooltip ? el("span", { class: "tip", title: o.tooltip, text: "ⓘ" }) : null;
  const label = el("label", { for: id, title: o.tooltip || null }, el("span", { text: f.label }), tip);
  let input, get, set, wide = false, extra = null;

  if (spec.type === "COMBO") {
    const opts = spec.options.slice();
    if (!opts.includes(f.value)) opts.unshift(f.value);
    input = el("select", { id }, opts.map(v => el("option", { value: String(v), text: String(v) + (spec.options.includes(v) ? "" : " (not installed)") })));
    get = () => { const v = input.value; return spec.options.find(x => String(x) === v) ?? v; };
    set = v => { if ([...input.options].some(o => o.value === String(v))) input.value = String(v); };
  } else if (spec.type === "BOOLEAN") {
    input = el("input", { id, type: "checkbox" });
    const lab = el("label", { class: "switch", for: id }, input, el("span", { text: f.label }), tip);
    get = () => input.checked;
    set = v => { input.checked = !!v; };
    set(f.value);
    input.addEventListener("change", onChange);
    return { elem: el("div", { class: "field" }, lab), get, set, wide: false };
  } else if (spec.type === "INT" || spec.type === "FLOAT") {
    const isInt = spec.type === "INT";
    const min = Number.isFinite(o.min) && Math.abs(o.min) < 1e15 ? o.min : null;
    const max = Number.isFinite(o.max) && Math.abs(o.max) < 1e15 ? o.max : null;
    const step = o.step ?? (isInt ? 1 : 0.01);
    const snap4 = /4n\s*\+\s*1/.test(f.label);
    const norm = v => {
      let x = Number(v);
      if (!Number.isFinite(x)) x = f.value;
      if (isInt) x = Math.round(x);
      if (snap4) x = Math.max(1, Math.round((x - 1) / 4) * 4 + 1);
      if (min != null) x = Math.max(min, x);
      if (max != null) x = Math.min(max, x);
      return isInt ? x : Math.round(x * 1e6) / 1e6;
    };
    const num = el("input", { id, type: "number", step: "any", inputmode: isInt ? "numeric" : "decimal" });
    if (min != null) num.min = min;
    if (max != null) num.max = max;
    if (min != null && max != null && max - min <= 4096 && !snap4) {
      const range = el("input", { type: "range", min, max, step, "aria-label": f.label });
      range.addEventListener("input", () => { num.value = range.value; onChange(); });
      num.classList.add("num-sm");
      num.addEventListener("change", () => { num.value = norm(num.value); range.value = num.value; onChange(); });
      input = el("div", { class: "row" }, range, num);
      set = v => { num.value = norm(v); range.value = num.value; };
    } else {
      num.addEventListener("change", () => { num.value = norm(num.value); onChange(); });
      input = num;
      set = v => { num.value = norm(v); };
    }
    get = () => norm(num.value);
    if (snap4) extra = el("div", { class: "hint", text: "Rounded to 4n+1 (e.g. 17, 33, 49, 81)." });
    set(f.value);
    return { elem: el("div", { class: "field" }, label, input, extra), get, set, wide: false };
  } else {
    // STRING
    const multiline = o.multiline || f.kind === "prompt" || String(f.value).length > 60 || String(f.value).includes("\n");
    const isColor = /colou?r$/.test(f.input) && /^#[0-9a-f]{6}$/i.test(String(f.value));
    const isHexList = /colou?rs$/.test(f.input);
    if (isColor) {
      const txt = el("input", { id, type: "text", spellcheck: "false" });
      const pick = el("input", { type: "color", "aria-label": f.label + " picker" });
      pick.addEventListener("input", () => { txt.value = pick.value; onChange(); });
      txt.addEventListener("input", () => { if (/^#[0-9a-f]{6}$/i.test(txt.value)) pick.value = txt.value; onChange(); });
      input = el("div", { class: "row" }, pick, txt);
      get = () => txt.value;
      set = v => { txt.value = v; if (/^#[0-9a-f]{6}$/i.test(v)) pick.value = v; };
    } else {
      input = multiline
        ? el("textarea", { id, rows: f.kind === "prompt" || PROMPTISH.has(f.input) ? 4 : 2, class: f.kind === "prompt" || f.input === "subject" ? "prompt" : null, spellcheck: "true" })
        : el("input", { id, type: "text", spellcheck: isHexList ? "false" : "true" });
      wide = true;
      get = () => input.value;
      set = v => { input.value = v; };
      if (isHexList) {
        const sw = el("div", { class: "swatches" });
        const paint = () => sw.replaceChildren(...(input.value.match(HEX_RE) || []).map(c => el("span", { style: { background: c }, title: c })));
        input.addEventListener("input", paint);
        extra = sw;
        set = v => { input.value = v; paint(); };
      }
    }
    if (isColor) wide = false;
    input.addEventListener("input", onChange);
  }
  if (spec.type === "COMBO") input.addEventListener("change", onChange);
  set(f.value);
  return { elem: el("div", { class: "field" + (wide ? " wide" : "") }, label, input, extra), get, set, wide };
}

function imageWidget(f, onChange) {
  let value = f.value, pending = null;
  const thumb = el("img", { alt: "", draggable: "false" });
  const name = el("b");
  const status = el("span");
  const picker = el("select", { "aria-label": `Pick an existing input for ${f.label}` });
  const file = el("input", { type: "file", accept: "image/*", hidden: true });
  const fillPicker = () => {
    const opts = (inputSpec("LoadImage", "image")?.[0] || []).slice();
    if (Array.isArray(opts) && !opts.includes(value)) opts.unshift(value);
    picker.replaceChildren(el("option", { value: "", text: "…or pick an uploaded image" }),
      ...(Array.isArray(opts) ? opts : []).map(o => el("option", { value: o, text: o })));
    picker.value = value;
  };
  const show = () => {
    const [sub, fn] = splitPath(value);
    thumb.src = viewUrl(fn, sub, "input");
    name.textContent = fn;
    fillPicker();
  };
  const drop = el("div", { class: "drop", tabindex: "0", role: "button", "aria-label": `${f.label}: click, drop or paste an image` },
    el("div", { class: "thumb" }, thumb),
    el("div", { class: "meta" }, name, el("span", { text: "Click, drop or paste (⌘V) an image" }), " ", status, picker),
    file);
  thumb.addEventListener("load", () => { status.textContent = `· ${thumb.naturalWidth}×${thumb.naturalHeight}`; });
  thumb.addEventListener("error", () => { status.textContent = "· not found in ComfyUI's input folder"; });
  const accept = blob => {
    if (!blob || !/^image\//.test(blob.type)) return;
    drop.classList.add("uploading");
    status.textContent = "· uploading…";
    pending = uploadBlob(blob, blob.name || "pasted.png").then(n => {
      value = n; show(); onChange();
      loadObjInfo(["LoadImage"], ["LoadImage"]).then(fillPicker);
    }).catch(e => { status.textContent = "· " + e.message; }).finally(() => { drop.classList.remove("uploading"); pending = null; });
  };
  drop.addEventListener("click", e => { if (e.target !== picker && !picker.contains(e.target)) file.click(); });
  drop.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } });
  drop.addEventListener("focus", () => { state.activeDrop = accept; });
  drop.addEventListener("pointerenter", () => { state.activeDrop = accept; });
  drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", e => { e.preventDefault(); drop.classList.remove("over"); accept(e.dataTransfer.files[0]); });
  file.addEventListener("change", () => { accept(file.files[0]); file.value = ""; });
  picker.addEventListener("change", () => { if (picker.value) { value = picker.value; show(); onChange(); } });
  show();
  return {
    elem: el("div", { class: "field wide" }, el("div", { class: "label", text: f.label }), drop),
    get: () => value, set: v => { value = v; show(); },
    accept, wait: () => pending, wide: true,
  };
}

async function uploadBlob(blob, name) {
  const buf = await blob.arrayBuffer();
  let tag;
  try {
    const d = new Uint8Array(await crypto.subtle.digest("SHA-1", buf));
    tag = [...d.slice(0, 4)].map(b => b.toString(16).padStart(2, "0")).join("");
  } catch { tag = Date.now().toString(36); }
  const ext = ((name.match(/\.(png|jpe?g|webp|gif|bmp)$/i) || [".png"])[0]).toLowerCase();
  const base = name.replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9_-]+/g, "_").slice(0, 40) || "image";
  const fname = `sk_${tag}_${base}${ext}`;
  const fd = new FormData();
  fd.append("image", new File([buf], fname, { type: blob.type || "image/png" }));
  fd.append("overwrite", "true");
  const r = await fetch("/upload/image", { method: "POST", body: fd });
  if (!r.ok) throw new Error(`upload failed (HTTP ${r.status})`);
  const j = await r.json();
  return j.subfolder ? `${j.subfolder}/${j.name}` : j.name;
}

// ---- sidebar ---------------------------------------------------------------------------------
function renderNav() {
  const q = $("#wf-search").value.trim().toLowerCase();
  const byNum = Object.fromEntries(state.workflows.map(w => [wfNum(w.id), w]));
  const used = new Set();
  const groups = GROUPS.map(([name, nums]) => [name, nums.map(n => byNum[n]).filter(Boolean)]);
  groups.forEach(([, ws]) => ws.forEach(w => used.add(w.id)));
  const rest = state.workflows.filter(w => !used.has(w.id));
  if (rest.length) groups.push(["Other", rest]);
  const busy = new Set(state.jobs.filter(j => j.status === "queued" || j.status === "running").map(j => j.wf));
  const nav = $("#wf-nav");
  nav.replaceChildren(...groups.map(([name, ws]) => {
    const items = ws.filter(w => !q || `${w.title} ${w.description} ${name}`.toLowerCase().includes(q));
    if (!items.length) return null;
    return el("div", { class: "nav-group" }, el("h3", { text: name }), items.map(w => {
      const m = w.title.match(/^(.*?)\s*\(([^)]*)\)\s*$/);
      return el("button", {
        type: "button", class: "nav-item", title: w.description, "aria-current": state.wf?.id === w.id ? "true" : "false",
        onclick: () => { location.hash = "#/" + w.id; },
      }, el("span", { class: "num", text: wfNum(w.id) }),
      el("span", {}, el("span", { class: "t", text: m ? m[1] : w.title }), m ? el("span", { class: "sub", text: m[2] }) : null),
      busy.has(w.id) ? el("span", { class: "busy", title: "running" }) : null);
    }));
  }).filter(Boolean));
}

// ---- workflow form ---------------------------------------------------------------------------
let saveTimer = null;
function formState() { return store.get("sk.form." + state.wf.id, {}); }
function saveForm() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    const w = state.wf;
    if (!w) return;
    const values = {};
    for (const f of w.widgets) values[f.key] = f.get();
    store.set("sk.form." + w.id, { values, seed: w.seed.get(), seedRandom: w.seed.random(), fps: w.fps?.get() });
  }, 300);
}

async function selectWorkflow(id, overrides = null) {
  const meta = state.workflows.find(w => w.id === id) || state.workflows[0];
  if (!meta) return;
  let api;
  try { api = await getJSON("/sk/workflow/" + encodeURIComponent(meta.id)); }
  catch (e) { $("#wf-fields").replaceChildren(el("p", { class: "run-status err", text: "Could not load workflow: " + e.message })); return; }
  state.graphs[meta.id] = api;
  await loadObjInfo([...Object.values(api).map(n => n.class_type), "LoadImage"]);
  const model = buildModel(api);
  state.wf = { id: meta.id, meta, api, model, widgets: [] };
  renderNav();
  renderForm(overrides);
}

function renderForm(overrides) {
  const w = state.wf, m = w.model, d = describe(w.api);
  const saved = overrides || formState();
  const vals = saved.values || {};
  w.widgets = [];
  const add = (f, widget) => {
    if (f.key in vals) { try { widget.set(vals[f.key]); } catch { /* stale value */ } }
    w.widgets.push(Object.assign(widget, { key: f.key, nid: f.nid, input: f.input, field: f }));
    return widget.elem;
  };
  const onChange = () => saveForm();
  const fieldsGrid = fields => {
    const grid = el("div", { class: "grid2" });
    for (const f of fields) grid.append(add(f, fieldWidget(f, onChange)));
    return grid;
  };

  $("#wf-head").replaceChildren(
    el("h2", { text: w.meta.title }),
    el("p", { text: w.meta.description || "", hidden: !w.meta.description }),
    el("div", { class: "chips" },
      el("span", { class: "chip" + (d.noAI ? "" : " ai"), text: d.noAI ? "No AI" : d.models.join(" + ") || "AI" }),
      el("span", { class: "chip", title: "Typical time on this Mac", text: d.eta }),
      el("span", { class: "chip", title: "Workflow file", text: w.id })));
  const looks = lookPresets([...m.mainNodes, ...m.advNodes]);
  if (looks.btn) $("#wf-head").append(el("div", { class: "preset-head look-head" }, looks.btn), looks.panel);

  const parts = [];
  if (m.images.length) parts.push(el("fieldset", {}, el("legend", { text: m.images.length > 1 ? "Images" : "Image" }),
    m.images.map(f => add(f, imageWidget(f, onChange)))));
  if (m.prompts.length) parts.push(el("fieldset", {}, el("legend", { text: "Prompt" }), fieldsGrid(m.prompts)));
  // one preset menu per node (a node split into "X" and "X — more" gets it on its first section)
  const withPresets = new Set();
  const presetUI = node => {
    if (withPresets.has(node.nid)) return { btn: null, panel: null };
    withPresets.add(node.nid);
    return sectionPresets(node);
  };
  for (const node of m.mainNodes) {
    const ui = presetUI(node);
    parts.push(el("fieldset", {}, el("legend", {}, el("span", { text: node.label }), ui.btn), ui.panel, fieldsGrid(node.fields)));
  }

  // settings: primitive ints, seed, fps
  const settings = el("fieldset", {}, el("legend", { text: "Settings" }));
  if (m.settings.length) settings.append(fieldsGrid(m.settings));
  w.seed = seedWidget(m, saved, onChange);
  if (w.seed.elem) settings.append(w.seed.elem);
  w.fps = null;
  if (m.fps.length) {
    const f = { key: "__fps", nid: m.fps[0].nid, cls: "SK_SavePixelArt", input: "fps", value: m.fps[0].value, label: "Animation FPS" };
    w.fps = fieldWidget(f, onChange);
    if (saved.fps != null) w.fps.set(saved.fps);
    (m.noAI ? settings : null)?.append(w.fps.elem);
  }
  if (settings.children.length > 1) parts.push(settings);

  const advNodes = m.advNodes.slice();
  if (m.advNodes.length || (w.fps && !m.noAI)) {
    const body = el("div", { class: "adv-body" });
    if (w.fps && !m.noAI) body.append(el("details", { class: "node" }, el("summary", { text: "Output" }), w.fps.elem));
    for (const node of advNodes) {
      const ui = presetUI(node);
      body.append(el("details", { class: "node" },
        el("summary", {}, node.label, el("span", { class: "cls", text: objInfo[node.cls]?.display_name || node.cls })),
        ui.btn ? el("div", { class: "preset-head" }, ui.btn) : null, ui.panel,
        fieldsGrid(node.fields)));
    }
    const count = advNodes.reduce((a, n) => a + n.fields.length, 0) + (w.fps && !m.noAI ? 1 : 0);
    parts.push(el("details", { class: "adv" }, el("summary", { text: `Advanced settings (${count})` }), body));
  }
  if (!parts.length) parts.push(el("p", { class: "hint", text: "This workflow has no editable inputs." }));
  $("#wf-fields").replaceChildren(...parts);
  $("#wf-fields").classList.toggle("hd", isHD(w.id));
  $("#run-btn").firstChild.textContent = d.noAI ? "Run " : "Generate ";
  setStatus("");

  if (state.pendingImage) {
    const img = w.widgets.find(x => x.field.kind === "image");
    if (img) { img.set(state.pendingImage); saveForm(); }
    state.pendingImage = null;
  }
}

// ---- presets: section settings reusable in every tool that has the same section type -------------
// Stored per node class ("sk.presets"), and "looks" that bundle several sections ("sk.looks").
// not part of a preset: the character itself, and the model a tool is built around
const PRESET_SKIP = new Set(["subject", "character", "model"]);
// inputs a preset leaves alone while the graph wires the node to a reference (palette locks, edit round-trips)
const KEEP_WHEN_LINKED = { SK_Palette: ["palette_image", ["source", "colors"]], SK_PixelPerfect: ["reference", ["mode"]] };
const SUBJECT_TIP = "Subject wording that worked: ask for the character from image 1 drawn exactly like the sprite in image 2, "
  + "with the same small size, super-deformed proportions (big head about one third of the body height, short legs), "
  + "thick dark outlines, saturated cel shading and pixel density, then list the outfit details to keep.";
const BUILTIN_PRESETS = {
  SK_StylePreset: [
    { name: "FFBE-style chibi 96×120 (tested)", values: { style: "HD-2D character (Octopath-like)", art_width: 96, art_height: 120, colors: 40, background: "auto" },
      note: "Tested in 03 with a style reference sprite. " + SUBJECT_TIP },
    { name: "GBA chibi 32×32", values: { style: "character (front, RPG)", art_width: 32, art_height: 32, colors: 15, background: "auto",
      extra: "Game Boy Advance era JRPG sprite, vibrant saturated colors, clean dark outlines, soft cel shading" } },
    { name: "FFT-style unit 32×48", values: { style: "HD-2D character (Octopath-like)", art_width: 32, art_height: 48, colors: 15, background: "auto",
      extra: "tactics RPG unit sprite, isometric three-quarter view, muted earthy desaturated palette, soft painterly shading, thin dark brown outlines" } },
    { name: "HD anime JRPG 80×112", values: { style: "high-detail modern (indie)", art_width: 80, art_height: 112, colors: 48, background: "white",
      extra: "anime JRPG tactics unit sprite, full body, three-quarter front view, about 4 heads tall, crisp dark outlines, cel shading with soft highlights, battle-ready stance, centered" } },
  ],
  SK_PixelPerfect: [
    { name: "Detailed (64 colour clusters)", values: { method: "dominant", pre_colors: 64 } },
    { name: "Smooth shading (median)", values: { method: "median", pre_colors: 48 } },
  ],
  SK_Palette: [
    { name: "Clean, no dither", values: { dither: "none", shared_palette: true } },
    { name: "Rich 48 colours", values: { source: "auto", colors: 48, dither: "none" } },
    { name: "PICO-8", values: { source: "preset", preset: "PICO-8", colors: 16, dither: "none" } },
    { name: "Game Boy + Bayer", values: { source: "preset", preset: "Game Boy (DMG)", colors: 4, dither: "bayer-4", dither_strength: 0.4 } },
  ],
  SK_BackgroundRemove: [
    { name: "Clear enclosed gaps (global key, tested)", values: { mode: "global_key", tolerance: 0.06 },
      note: "Tested in 04: removes backdrop pockets (inside a staff ring, between arm and body) and keeps the outlines. "
        + "Anything painted in exactly the backdrop colour is removed too." },
  ],
  SK_Cleanup: [
    { name: "Keep detail", values: { speck_size: 2, max_contrast: 0.12, outline: "none" } },
    { name: "Soft auto outline", values: { outline: "outer_auto", outline_diagonal: false, darken: 0.45 } },
    { name: "Dark outline", values: { outline: "outer_color", outline_color: "#140c1c", outline_diagonal: false } },
    { name: "Heavy despeckle", values: { speck_size: 4, max_contrast: 0.2 } },
  ],
};
const BUILTIN_LOOKS = [
  { name: "FFBE-style chibi sprite (tested)", note: "Style 96×120, 40 colours, 64 colour clusters, speck size 2, no outline, no dither. " + SUBJECT_TIP,
    sections: { SK_StylePreset: BUILTIN_PRESETS.SK_StylePreset[0].values, SK_PixelPerfect: { method: "dominant", pre_colors: 64 },
      SK_Cleanup: { speck_size: 2, max_contrast: 0.12, outline: "none" }, SK_Palette: { dither: "none" } } },
];

const sectionName = node => node.label.replace(/ — more$/, "");
const typeName = cls => NODE_LABELS[cls] || objInfo[cls]?.display_name || cls;
function nodeValues(nid) {
  const v = {};
  for (const x of state.wf.widgets) if (x.nid === nid && !PRESET_SKIP.has(x.input)) v[x.input] = x.get();
  return v;
}
function applyToNode(nid, values) {
  const n = state.wf.api[nid];
  const [link, guarded] = KEEP_WHEN_LINKED[canon(n.class_type)] || [null, []];
  const locked = !!link && isLink(n.inputs[link]);
  let applied = 0, kept = 0;
  for (const x of state.wf.widgets) {
    if (x.nid !== nid || PRESET_SKIP.has(x.input) || !(x.input in values)) continue;
    if (locked && guarded.includes(x.input)) { kept++; continue; }
    x.set(values[x.input]);
    applied++;
  }
  return { applied, kept };
}
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;

function presetPanel(opts) {
  const panel = el("div", { class: "preset-panel", hidden: true });
  const btn = el("button", { type: "button", class: "small ghost preset-btn", "aria-expanded": "false" });
  const paint = () => {
    const items = opts.items();
    btn.textContent = `${opts.label} (${items.length}) ▾`;
    if (panel.hidden) return;
    const nameIn = el("input", { type: "text", placeholder: "Name for the current settings", "aria-label": `${opts.label} name`, maxlength: 60 });
    const save = () => {
      const name = nameIn.value.trim();
      if (!name) { nameIn.focus(); return; }
      opts.save(name);
      paint();
    };
    nameIn.addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); save(); } });
    panel.replaceChildren(...[
      items.length ? el("ul", { class: "preset-list" }, items.map(it => el("li", { title: it.note || "" },
        el("span", { class: "pname" }, it.name, it.builtin ? el("span", { class: "tag", text: "built-in" }) : null,
          it.note ? el("span", { class: "pnote", text: it.note }) : null),
        el("button", { type: "button", class: "small", text: "Apply", onclick: () => opts.apply(it) }),
        it.builtin ? null : el("button", {
          type: "button", class: "small ghost", text: "✕", title: "Delete", "aria-label": `Delete ${it.name}`,
          onclick: () => { opts.remove(it.name); paint(); },
        })))) : null,
      el("p", { class: "hint", text: opts.hint }),
      el("div", { class: "row" }, nameIn, el("button", { type: "button", class: "small", text: "Save current", onclick: save })),
    ].filter(Boolean));
  };
  btn.addEventListener("click", () => {
    panel.hidden = !panel.hidden;
    btn.setAttribute("aria-expanded", String(!panel.hidden));
    paint();
  });
  paint();
  return { btn, panel };
}

function sectionPresets(node) {
  const cls = node.cls, label = sectionName(node);
  return presetPanel({
    label: "Presets",
    hint: `Saves this section (not the subject or model). Use it in any tool with a ${typeName(cls)} section.`,
    items: () => [...(BUILTIN_PRESETS[cls] || []).map(p => ({ ...p, builtin: true })),
      ...Object.entries(store.get("sk.presets", {})[cls] || {}).sort(([a], [b]) => a.localeCompare(b))
        .map(([name, p]) => ({ name, values: p.values, note: p.from ? `saved in ${p.from}` : "" }))],
    apply: it => {
      const r = applyToNode(node.nid, it.values);
      saveForm();
      setStatus(`Applied "${it.name}" to ${label}: ${plural(r.applied, "setting")}`
        + (r.kept ? `, kept ${plural(r.kept, "setting")} that this tool wires to its reference image` : "") + ".");
    },
    save: name => {
      const all = store.get("sk.presets", {});
      (all[cls] ??= {})[name] = { values: nodeValues(node.nid), from: state.wf.id, saved: Date.now() };
      store.set("sk.presets", all);
      setStatus(`Saved preset "${name}" for ${typeName(cls)} sections.`);
    },
    remove: name => {
      const all = store.get("sk.presets", {});
      if (all[cls]) { delete all[cls][name]; if (!Object.keys(all[cls]).length) delete all[cls]; }
      store.set("sk.presets", all);
    },
  });
}

function lookPresets(nodes) {
  const byNid = new Map(nodes.map(n => [n.nid, n]));
  if (!byNid.size) return { btn: null, panel: null };
  return presetPanel({
    label: "Looks",
    hint: "A look applies several sections at once (style, pixel grid, palette, cleanup…) to every matching section of this tool.",
    items: () => [...BUILTIN_LOOKS.map(l => ({ ...l, builtin: true })),
      ...Object.entries(store.get("sk.looks", {})).sort(([a], [b]) => a.localeCompare(b))
        .map(([name, l]) => ({ name, sections: l.sections, note: l.from ? `saved in ${l.from}` : "" }))],
    apply: it => {
      const done = [], kept = [];
      for (const node of byNid.values()) {
        const values = it.sections[node.cls];
        if (!values) continue;
        const r = applyToNode(node.nid, values);
        if (r.applied) done.push(sectionName(node));
        if (r.kept) kept.push(sectionName(node));
      }
      saveForm();
      const missing = Object.keys(it.sections).filter(c => ![...byNid.values()].some(n => n.cls === c)).map(typeName);
      setStatus(done.length ? `Applied look "${it.name}" to ${done.join(", ")}.`
        + (kept.length ? ` Kept the reference lock in ${kept.join(", ")}.` : "")
        + (missing.length ? ` This tool has no ${missing.join(" / ")} section.` : "")
        : `This tool has none of the sections in "${it.name}".`);
    },
    save: name => {
      const sections = {};
      for (const node of byNid.values()) {
        const v = nodeValues(node.nid);
        if (Object.keys(v).length) sections[node.cls] = { ...(sections[node.cls] || {}), ...v };
      }
      const all = store.get("sk.looks", {});
      all[name] = { sections, from: state.wf.id, saved: Date.now() };
      store.set("sk.looks", all);
      setStatus(`Saved look "${name}" (${Object.keys(sections).map(typeName).join(", ")}).`);
    },
    remove: name => { const all = store.get("sk.looks", {}); delete all[name]; store.set("sk.looks", all); },
  });
}

function seedWidget(m, saved, onChange) {
  if (!m.seeds.length) return { elem: null, get: () => null, random: () => false, set() {} };
  let random = saved.seedRandom ?? true;
  const num = el("input", { id: "f-seed", type: "number", min: 0, step: 1, value: saved.seed ?? m.seeds[0].value });
  const dice = el("button", { type: "button", title: "New random seed", text: "🎲", onclick: () => { num.value = randSeed(); onChange(); } });
  const lock = el("button", { type: "button", class: "toggle-btn", "aria-pressed": String(!random) });
  const paint = () => {
    lock.setAttribute("aria-pressed", String(!random));
    lock.textContent = random ? "Random each run" : "🔒 Locked";
    lock.title = random ? "A new seed is drawn for every run (click to keep this seed)" : "This seed is reused (click to randomize every run)";
  };
  lock.addEventListener("click", () => { random = !random; paint(); onChange(); });
  num.addEventListener("change", onChange);
  paint();
  const elem = el("div", { class: "field" }, el("label", { for: "f-seed", text: "Seed" }),
    el("div", { class: "row seed-row" }, num, dice, lock),
    el("div", { class: "hint", text: `Applied to ${m.seeds.length} sampler seed${m.seeds.length > 1 ? "s" : ""}.` }));
  return {
    elem, random: () => random,
    get: () => Math.max(0, Math.min(SEED_MAX * SEED_MAX, Math.round(Number(num.value) || 0))),
    set: v => { num.value = v; },
    setRandom: r => { random = r; paint(); },
  };
}

function setStatus(text, err = false) {
  const s = $("#run-status");
  s.textContent = text;
  s.classList.toggle("err", err);
}

// ---- running ---------------------------------------------------------------------------------
function promptError(d, graph) {
  if (!d) return "ComfyUI rejected the workflow.";
  const err = d.error || {};
  const lines = [typeof err === "string" ? err : err.message || "ComfyUI rejected the workflow."];
  if (err.details) lines.push(err.details);
  for (const [nid, ne] of Object.entries(d.node_errors || {})) {
    const t = graph[nid]?._meta?.title || ne.class_type || nid;
    for (const e of ne.errors || []) lines.push(`• ${t} (#${nid}): ${e.message}${e.details ? " — " + e.details : ""}`);
  }
  return lines.join("\n");
}

function summarize(model, values) {
  for (const f of [...model.prompts, ...model.mainNodes.flatMap(n => n.fields)]) {
    const v = values[f.key];
    if (typeof v === "string" && v.trim() && PROMPTISH.has(f.input)) return v.trim().slice(0, 160);
  }
  const img = model.images[0];
  return img ? splitPath(String(values[img.key]))[1] : "";
}

let submitting = false;
async function submit(ev) {
  ev?.preventDefault();
  const w = state.wf;
  if (!w || submitting) return;
  submitting = true;
  const btn = $("#run-btn");
  btn.disabled = true;
  try {
    setStatus("Preparing…");
    await Promise.all(w.widgets.map(x => x.wait?.()));
    const graph = structuredClone(w.api);
    const values = {};
    for (const x of w.widgets) { const v = x.get(); graph[x.nid].inputs[x.input] = v; values[x.key] = v; }
    let seed = null;
    if (w.model.seeds.length) {
      if (w.seed.random()) w.seed.set(randSeed());
      seed = w.seed.get();
      for (const s of w.model.seeds) graph[s.nid].inputs[s.input] = seed;
    }
    if (w.fps) { const f = w.fps.get(); for (const s of w.model.fps) graph[s.nid].inputs.fps = f; }
    saveForm();
    setStatus("Queueing…");
    const res = await postJSON("/prompt", { prompt: graph, client_id: CLIENT_ID });
    if (!res.ok || !res.data?.prompt_id) throw new Error(promptError(res.data, graph));
    const titles = {};
    for (const nid of Object.keys(graph)) titles[nid] = nodeLabel(graph, nid);
    const job = {
      pid: res.data.prompt_id, wf: w.id, wfTitle: w.meta.title, created: Date.now(), status: "queued",
      seed, values, fps: w.fps?.get(), seedRandom: w.seed.random(), summary: summarize(w.model, values), titles,
      total: Object.keys(graph).length, outputs: [], texts: [],
    };
    state.jobs.unshift(job);
    saveJobs();
    renderFeed();
    renderNav();
    setStatus(`Queued (${job.pid.slice(0, 8)}). Progress is shown on the right.`);
    poll();
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    submitting = false;
    btn.disabled = false;
  }
}

async function cancelJob(job) {
  try {
    const q = await getJSON("/queue");
    if (q.queue_running.some(x => x[1] === job.pid)) await postJSON("/interrupt", { prompt_id: job.pid });
    else if (q.queue_pending.some(x => x[1] === job.pid)) {
      await postJSON("/queue", { delete: [job.pid] });
      Object.assign(job, { status: "cancelled", finished: Date.now() });
    }
  } catch (e) { job.error = "Cancel failed: " + e.message; }
  saveJobs();
  renderJob(job);
  renderNav();
  setTimeout(poll, 500);
}

// ---- job tracking (queue/history polling + websocket progress) ---------------------------------
const active = j => j.status === "queued" || j.status === "running";
const PERSIST = ["pid", "wf", "wfTitle", "created", "started", "finished", "status", "error", "seed", "values", "fps",
  "seedRandom", "summary", "titles", "total", "outputs", "texts", "collapsed"];
function saveJobs() {
  state.jobs = state.jobs.slice(0, 80);
  store.set("sk.jobs", state.jobs.map(j => Object.fromEntries(PERSIST.filter(k => k in j).map(k => [k, j[k]]))));
}

let polling = false;
async function poll() {
  if (polling) return;
  const live = state.jobs.filter(active);
  if (!live.length) return;
  polling = true;
  try {
    const q = await getJSON("/queue");
    const running = new Set(q.queue_running.map(x => x[1]));
    const pending = q.queue_pending.map(x => ({ n: x[0], id: x[1] })).sort((a, b) => a.n - b.n);
    state.queueRemaining = q.queue_running.length + q.queue_pending.length;
    paintQueue();
    for (const j of live) {
      if (running.has(j.pid)) {
        if (j.status !== "running") { j.status = "running"; j.started ??= Date.now(); }
        j.missing = 0;
      } else {
        const i = pending.findIndex(p => p.id === j.pid);
        if (i >= 0) { j.status = "queued"; j.ahead = i + q.queue_running.length; j.missing = 0; }
        else await checkHistory(j);
      }
      renderJob(j);
    }
    saveJobs();
    renderNav();
  } catch { /* ComfyUI down: the status pill says so; retry next tick */ }
  polling = false;
}

async function checkHistory(j) {
  const h = await getJSON("/history/" + encodeURIComponent(j.pid));
  const entry = h[j.pid];
  if (!entry) {
    j.missing = (j.missing || 0) + 1;
    if (j.missing > 4) Object.assign(j, { status: "error", error: "The job is no longer in ComfyUI's queue or history (server restarted?).", finished: Date.now() });
    return;
  }
  finishJob(j, entry);
}

function finishJob(j, entry) {
  const st = entry.status || {};
  const msgs = st.messages || [];
  const msg = t => msgs.find(m => m[0] === t)?.[1];
  const start = msg("execution_start"), err = msg("execution_error"), intr = msg("execution_interrupted");
  const end = msg("execution_success") || err || intr;
  collectOutputs(j, entry.outputs || {});
  if (start?.timestamp) j.started = start.timestamp;
  j.finished = end?.timestamp || Date.now();
  if (err) {
    j.status = "error";
    const t = j.titles?.[err.node_id] || err.node_type;
    const tb = (err.traceback || []).join("").trim().split("\n").slice(-3).join("\n");
    j.error = `${t} (#${err.node_id}, ${err.node_type}): ${err.exception_type ? err.exception_type + ": " : ""}${err.exception_message}`.trim()
      + (tb ? "\n\n" + tb : "");
  } else if (intr) j.status = "cancelled";
  else j.status = "success";
  delete j.node; delete j.step; delete j.max; delete j.preview;
}

function collectOutputs(j, outputs) {
  const imgs = [], texts = [];
  for (const nid of Object.keys(outputs).sort(byNumericId)) {
    const o = outputs[nid], label = j.titles?.[nid] || nid;
    for (const im of [...(o.images || []), ...(o.gifs || [])])
      if (im?.filename) imgs.push({ nid, label, filename: im.filename, subfolder: im.subfolder || "", type: im.type || "output" });
    if (o.text) texts.push({ nid, label, text: [].concat(o.text).join("\n") });
  }
  j.outputs = imgs;
  j.texts = texts;
}

let ws = null, wsJob = null;
function connectWS() {
  try { ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?clientId=${CLIENT_ID}`); }
  catch { return; }
  ws.binaryType = "arraybuffer";
  ws.onmessage = ev => {
    if (typeof ev.data !== "string") return onPreview(ev.data);
    let m;
    try { m = JSON.parse(ev.data); } catch { return; }
    const d = m.data || {};
    if (m.type === "status") { state.queueRemaining = d.status?.exec_info?.queue_remaining ?? state.queueRemaining; paintQueue(); return; }
    const j = d.prompt_id && state.jobs.find(x => x.pid === d.prompt_id);
    if (!j) return;
    j.done ??= new Set();
    switch (m.type) {
      case "execution_start": j.status = "running"; j.started = d.timestamp || Date.now(); wsJob = j; break;
      case "execution_cached": (d.nodes || []).forEach(n => j.done.add(n)); break;
      case "executing":
        if (d.node == null) { setTimeout(poll, 200); break; }
        if (j.node && j.node !== d.node) j.done.add(j.node);
        j.node = d.node; j.step = j.max = null; wsJob = j; j.status = "running";
        break;
      case "progress": j.step = d.value; j.max = d.max; if (d.node) j.node = d.node; break;
      case "executed":
        j.done.add(d.node);
        if (d.output) {
          const partial = {};
          for (const o of j.outputs || []) (partial[o.nid] ??= { images: [] }).images.push(o);
          for (const t of j.texts || []) (partial[t.nid] ??= {}).text = t.text;
          partial[d.node] = d.output;
          collectOutputs(j, partial);
        }
        break;
      case "execution_error": case "execution_interrupted": case "execution_success": setTimeout(poll, 200); break;
      default: return;
    }
    renderJob(j);
  };
  ws.onclose = () => { ws = null; setTimeout(connectWS, 3000); };
}

function onPreview(buf) {
  if (!wsJob || !active(wsJob)) return;
  const dv = new DataView(buf);
  const kind = dv.getUint32(0);
  let off = 8, mime = dv.getUint32(4) === 2 ? "image/png" : "image/jpeg";
  if (kind === 4) {  // preview with metadata
    const len = dv.getUint32(4);
    try {
      const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, len)));
      const j = state.jobs.find(x => x.pid === meta.prompt_id);
      if (j) wsJob = j;
      mime = meta.image_type || "image/jpeg";
    } catch { /* ignore */ }
    off = 8 + len;
  } else if (kind !== 1) return;
  if (wsJob.preview) URL.revokeObjectURL(wsJob.preview);
  wsJob.preview = URL.createObjectURL(new Blob([new Uint8Array(buf, off)], { type: mime }));
  renderJob(wsJob);
}

// ---- results feed ----------------------------------------------------------------------------
function renderFeed() {
  const feed = $("#feed");
  if (!state.jobs.length) { feed.replaceChildren($("#tpl-empty-feed").content.cloneNode(true)); return; }
  feed.replaceChildren(...state.jobs.map((j, i) => {
    if (j.collapsed == null && i >= 3 && !active(j)) j.collapsed = true;
    return jobCard(j);
  }));
}

function jobCard(j) {
  const card = el("article", { class: "job", id: "job-" + j.pid });
  card._parts = {
    head: el("div", { class: "job-head" }), progress: el("div", { class: "job-progress" }),
    thumbs: el("div", { class: "thumbs" }), error: el("div"), body: el("div", { class: "job-body" }),
  };
  card.append(card._parts.head, card._parts.progress, card._parts.thumbs, card._parts.error, card._parts.body);
  fillJob(card, j);
  return card;
}

function renderJob(j) {
  const card = document.getElementById("job-" + j.pid);
  if (card) fillJob(card, j);
  else renderFeed();
}

function fillJob(card, j) {
  const p = card._parts;
  card.classList.toggle("collapsed", !!j.collapsed);
  const stateText = { queued: "queued", running: "running", success: "done", error: "error", cancelled: "cancelled" }[j.status];
  const when = new Date(j.created).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  const dur = j.finished && j.started ? ` · took ${fmtDur(j.finished - j.started)}` : "";
  p.head.replaceChildren(
    el("div", {},
      el("div", { class: "title" }, `${wfNum(j.wf)} · ${j.wfTitle} `, el("span", { class: "job-state " + j.status, text: stateText })),
      el("div", { class: "sub", text: [j.summary, j.seed != null ? `seed ${j.seed}` : null, when + dur].filter(Boolean).join(" · ") })),
    el("div", { class: "actions" },
      active(j) ? el("button", { type: "button", class: "small danger", text: "Cancel", onclick: () => cancelJob(j) }) : null,
      el("button", { type: "button", class: "small", text: "Reuse settings", title: "Load this run's settings into the form", onclick: () => reuse(j) }),
      el("button", {
        type: "button", class: "small ghost", text: j.collapsed ? "Show" : "Hide",
        onclick: () => { j.collapsed = !j.collapsed; saveJobs(); fillJob(card, j); },
      }),
      active(j) ? null : el("button", {
        type: "button", class: "small ghost", text: "✕", title: "Remove from history (files stay on disk)",
        onclick: () => { state.jobs = state.jobs.filter(x => x !== j); saveJobs(); renderFeed(); },
      })));
  paintProgress(p.progress, j);
  p.error.replaceChildren(...(j.error ? [el("div", { class: "job-error", text: j.error })] : []));

  const sig = JSON.stringify([j.outputs?.length, j.texts?.length, j.collapsed, j.preview, j.status === "success"]);
  if (card._sig === sig) return;
  card._sig = sig;
  const shown = (j.outputs || []).filter(o => o.type === "output");
  p.thumbs.replaceChildren(...(j.collapsed ? shown.slice(0, 8).map(o => el("img", { src: viewUrl(o.filename, o.subfolder, o.type), alt: "", loading: "lazy", class: isHD(j.wf) ? "hd" : null })) : []));
  if (j.collapsed) { p.body.replaceChildren(); return; }
  p.body.replaceChildren(...outputsView(j));
}

function paintProgress(box, j) {
  if (!active(j)) { box.replaceChildren(); return; }
  const now = Date.now();
  let text, frac = null;
  if (j.status === "queued") {
    text = j.ahead ? `Waiting in the ComfyUI queue — ${j.ahead} job${j.ahead > 1 ? "s" : ""} ahead` : "Waiting in the ComfyUI queue";
    text += ` · ${fmtTime(now - j.created)}`;
  } else {
    const node = j.node ? (j.titles?.[j.node] || j.node) : null;
    const done = j.done?.size || 0;
    text = `Running${node ? ` · ${node}` : ""}${j.max ? ` · step ${j.step}/${j.max}` : ""} · ${fmtTime(now - (j.started || j.created))} elapsed`;
    if (j.max) frac = j.step / j.max;
    if (done && j.total) text += ` · node ${Math.min(done + 1, j.total)}/${j.total}`;
  }
  const bar = el("div", { class: "bar" + (frac == null ? " indet" : "") }, el("i", { style: frac == null ? {} : { width: `${Math.round(frac * 100)}%` } }));
  box.replaceChildren(el("div", { text }), bar);
  if (j.preview) box.append(el("img", { class: "live-preview", src: j.preview, alt: "Live preview" }));
}

function outputsView(j) {
  const out = [], finals = [], temps = [], hd = isHD(j.wf);
  for (const o of j.outputs || []) (o.type === "output" ? finals : temps).push(o);
  if (hd) {  // HD saves list only the sheet: add the animation (APNG keeps soft alpha) next to it
    for (const o of finals.slice()) {
      const b = o.filename.match(/^(.*_\d{5})_sheet_x\d+\.png$/)?.[1];
      if (b && !finals.some(f => f.filename.startsWith(b) && /\.(gif|webp)$/i.test(f.filename)))
        finals.splice(finals.indexOf(o) + 1, 0, { ...o, filename: `${b}.apng.png`, label: o.label + " (animated)", optional: true });
    }
  }
  if (finals.length) out.push(el("div", { class: "outputs" }, finals.map(o => outCard(o, true, hd))));
  else if (j.status === "success") out.push(el("p", { class: "hint", text: "This run produced no saved images." }));
  for (const t of j.texts || []) out.push(el("div", { class: "out-text" }, el("b", { text: t.label }), t.text));
  if (temps.length) {
    out.push(el("div", { class: "section-label", text: "Intermediate previews" }));
    out.push(el("div", { class: "outputs" }, temps.map(o => outCard(o, false, hd))));
  }
  applyZoom(out);
  return out;
}

const SCALE_RE = /_x(\d+)\.(?:png|gif|webp)$/i;
function outCard(o, final, hd = false) {
  const src = viewUrl(o.filename, o.subfolder, o.type);
  const factor = hd ? 1 : Number((o.filename.match(SCALE_RE) || [])[1] || 1);
  const img = el("img", { src, alt: o.label, loading: "lazy", draggable: "true", class: hd ? "hd" : null });
  img.dataset.factor = factor;
  img.dataset.zoomable = final && !hd ? "1" : "0";
  const dim = el("span", { class: "dim" });
  const dl = el("div", { class: "dl" }, el("a", { href: src, download: o.filename, text: "⤓ " + fileKind(o.filename) }));
  const parts = isPartsComposite(o) ? el("div", { class: "parts" }) : null;
  const card = el("div", { class: "out-card" + (parts ? " wide" : "") });
  img.addEventListener("load", () => {
    const w = Math.round(img.naturalWidth / factor), h = Math.round(img.naturalHeight / factor);
    dim.textContent = factor > 1 ? ` ${w}×${h} (shown file ${factor}×)` : ` ${w}×${h}`;
    if (w > 512 || h > 512) img.dataset.zoomable = "0";
    if (!hd && w < 200 && h < 200) img.classList.add("small");
    zoomImg(img);
  });
  img.addEventListener("error", () => { if (o.optional) card.remove(); });
  if (final && parts) addParts(dl, parts, o);
  else if (final) addRelated(dl, o);
  const cap = el("div", { class: "cap" }, el("span", { class: "name", text: o.label }), dim,
    el("div", { class: "dim", text: (o.subfolder ? o.subfolder + "/" : "") + o.filename }), dl, parts,
    final && /\.png$/i.test(o.filename) && !o.optional ? sendTo(o) : null);
  card.append(el("div", { class: "pic" }, img), cap);
  return card;
}

// SK_SaveParts: <folder>_NNN/ with composite.png, parts.json, NN_<part>.png and cropped/NN_<part>.png
const isPartsComposite = o => o.filename === "composite.png" && o.type === "output";
async function addParts(dl, box, o) {
  let meta;
  try { meta = await getJSON(viewUrl("parts.json", o.subfolder)); } catch { return; }
  dl.append(el("a", { href: viewUrl("parts.json", o.subfolder), download: "parts.json", text: "⤓ parts.json" }));
  box.append(...(meta.parts || []).map(p => {
    const full = viewUrl(p.file, o.subfolder), crop = viewUrl(p.file, o.subfolder + "/cropped");
    return el("figure", { title: `${p.name} · ${p.w}×${p.h} at ${p.x},${p.y} · z ${p.z}` },
      el("img", { src: crop, alt: p.name, loading: "lazy", class: "hd" }),
      el("figcaption", {}, el("span", { text: p.name }),
        el("a", { href: full, download: p.file, text: "⤓ layer" }), " ",
        el("a", { href: crop, download: p.file, text: "⤓ cropped" })));
  }));
}

function fileKind(fn) {
  const s = fn.match(/_x(\d+)(?:\.apng)?\.\w+$/)?.[1];
  if (/\.apng\.png$/i.test(fn)) return "APNG" + (s ? ` ${s}×` : " 1×");
  if (/_sheet\.json$/i.test(fn)) return "Aseprite JSON";
  if (/_sheet(_x\d+)?\.png$/i.test(fn)) return "sheet PNG" + (s ? ` ${s}×` : " 1×");
  const ext = fn.split(".").pop().toUpperCase();
  return ext + (s ? ` ${s}×` : ext === "GIF" || ext === "PNG" ? " 1×" : "");
}

// Files SK_SavePixelArt writes next to the ones it lists in the UI.
function relatedFiles(fn) {
  let m = fn.match(/^(.*_\d{5})_sheet_x(\d+)\.png$/);
  if (m) {
    const [, b, s] = m;
    return [`${b}_sheet.png`, `${b}_sheet.json`, `${b}.gif`, `${b}.apng.png`]
      .concat(s === "1" ? [] : [`${b}_x${s}.gif`, `${b}_x${s}.apng.png`]);
  }
  m = fn.match(/^(.*_\d{5}(?:_\d{2})?)_x\d+\.png$/);
  if (m) return [`${m[1]}.png`];
  return [];
}
const existsCache = new Map();
function fileExists(url) {
  if (!existsCache.has(url)) existsCache.set(url, fetch(url, { method: "HEAD" }).then(r => r.ok).catch(() => false));
  return existsCache.get(url);
}
async function addRelated(dl, o) {
  for (const fn of relatedFiles(o.filename)) {
    const url = viewUrl(fn, o.subfolder, "output");
    if (await fileExists(url)) dl.append(el("a", { href: url, download: fn, text: "⤓ " + fileKind(fn) }));
  }
}
async function nativeSource(o) {
  for (const fn of relatedFiles(o.filename).filter(f => f.endsWith(".png") && !f.endsWith(".apng.png"))) {
    const url = viewUrl(fn, o.subfolder, "output");
    if (await fileExists(url)) return { url, fn };
  }
  return { url: viewUrl(o.filename, o.subfolder, o.type), fn: o.filename };
}

function sendTo(o) {
  const targets = state.workflows.filter(w => Object.values(state.graphs[w.id] || {}).some(n => n.class_type === "LoadImage"));
  if (!targets.length) return null;
  const sel = el("select", { "aria-label": "Use this image as the input of another tool" },
    el("option", { value: "", text: "Use as input in…" }),
    targets.map(w => el("option", { value: w.id, text: `${wfNum(w.id)} · ${w.title}` })));
  sel.addEventListener("change", async () => {
    const id = sel.value;
    if (!id) return;
    sel.disabled = true;
    try {
      const src = await nativeSource(o);
      const blob = await (await fetch(src.url)).blob();
      state.pendingImage = await uploadBlob(new File([blob], src.fn, { type: blob.type || "image/png" }), src.fn);
      if (location.hash === "#/" + id) await selectWorkflow(id); else location.hash = "#/" + id;
    } catch (e) { setStatus("Could not send image: " + e.message, true); }
    sel.disabled = false;
    sel.value = "";
  });
  return el("div", { class: "send" }, sel);
}

function zoomImg(img) {
  const z = state.zoom, f = Number(img.dataset.factor || 1);
  if (z > 0 && img.dataset.zoomable === "1" && img.naturalWidth) {
    img.style.width = `${(img.naturalWidth / f) * z}px`;
    img.style.height = `${(img.naturalHeight / f) * z}px`;
    img.style.maxWidth = img.style.maxHeight = "none";
  } else {
    img.style.width = img.style.height = img.style.maxWidth = img.style.maxHeight = "";
  }
}
function applyZoom(roots = [$("#feed")]) {
  for (const r of roots) {
    if (!(r instanceof Element)) continue;
    const grids = r.matches(".outputs") ? [r] : r.querySelectorAll(".outputs");
    grids.forEach(g => g.classList.toggle("zoomed", state.zoom > 0));
    r.querySelectorAll(".out-card img").forEach(zoomImg);
  }
}

function reuse(j) {
  const target = { values: j.values || {}, seed: j.seed, seedRandom: false, fps: j.fps };
  store.set("sk.form." + j.wf, target);
  if (location.hash === "#/" + j.wf) selectWorkflow(j.wf, target);
  else location.hash = "#/" + j.wf;
}

// ---- header status ---------------------------------------------------------------------------
async function pingComfy() {
  const pill = $("#comfy-status");
  try {
    const s = await getJSON("/system_stats");
    pill.className = "status-pill ok";
    pill.textContent = "ComfyUI " + (s.system?.comfyui_version || "");
  } catch {
    pill.className = "status-pill down";
    pill.textContent = "ComfyUI offline";
  }
}
function paintQueue() {
  const n = state.queueRemaining;
  $("#queue-status").textContent = n == null ? "" : n ? `Queue: ${n}` : "Queue idle";
}

// ---- theme -----------------------------------------------------------------------------------
function initTheme() {
  const meta = document.querySelector('meta[name="color-scheme"]');
  const mq = matchMedia("(prefers-color-scheme: dark)");
  $("#theme-toggle").addEventListener("click", () => {
    const rendered = meta.content === "light dark" ? (mq.matches ? "dark" : "light") : meta.content;
    const target = rendered === "dark" ? "light" : "dark";
    // stored raw (not JSON) because the inline script in <head> reads it before first paint
    try {
      if (target === (mq.matches ? "dark" : "light")) localStorage.removeItem("sk.color-scheme");
      else localStorage.setItem("sk.color-scheme", target);
    } catch { /* not persisted, still applied */ }
    meta.content = target === (mq.matches ? "dark" : "light") ? "light dark" : target;
  });
  addEventListener("storage", e => { if (e.key === "sk.color-scheme") meta.content = e.newValue ?? "light dark"; });
}

// ---- boot ------------------------------------------------------------------------------------
async function boot() {
  initTheme();

  $("#wf-form").noValidate = true;
  $("#wf-form").addEventListener("submit", submit);
  $("#reset-btn").addEventListener("click", () => { if (!state.wf) return; store.del("sk.form." + state.wf.id); selectWorkflow(state.wf.id); });
  $("#wf-search").addEventListener("input", renderNav);
  document.addEventListener("keydown", e => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); submit(); }
  });
  document.addEventListener("paste", e => {
    const item = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith("image/"));
    if (!item) return;
    const accept = state.activeDrop || state.wf?.widgets.find(x => x.accept)?.accept;
    if (!accept) return;
    e.preventDefault();
    accept(item.getAsFile());
  });
  const zoom = $("#zoom"), zout = $("#zoom-out"), checker = $("#checker");
  const paintZoom = () => { zout.textContent = state.zoom ? `${state.zoom}×` : "Fit"; };
  zoom.value = state.zoom; paintZoom();
  zoom.addEventListener("input", () => { state.zoom = Number(zoom.value); store.set("sk.zoom", state.zoom); paintZoom(); applyZoom(); });
  checker.checked = state.checker;
  $("#feed").classList.toggle("checker", state.checker);
  checker.addEventListener("change", () => { state.checker = checker.checked; store.set("sk.checker", state.checker); $("#feed").classList.toggle("checker", state.checker); });
  $("#clear-history").addEventListener("click", () => { state.jobs = state.jobs.filter(active); saveJobs(); renderFeed(); });

  pingComfy();
  setInterval(pingComfy, 10000);
  connectWS();
  renderFeed();

  try { state.workflows = await getJSON("/sk/workflows"); }
  catch (e) { $("#wf-fields").replaceChildren(el("p", { class: "run-status err", text: "Could not list workflows: " + e.message })); return; }
  renderNav();
  const fromHash = () => decodeURIComponent(location.hash.replace(/^#\/?/, ""));
  addEventListener("hashchange", () => selectWorkflow(fromHash()));
  await selectWorkflow(fromHash() || state.workflows[0]?.id);
  // learn which workflows take images (for "Use as input in…"), then refresh cards that need it
  await Promise.all(state.workflows.filter(w => !state.graphs[w.id]).map(async w => {
    try { state.graphs[w.id] = await getJSON("/sk/workflow/" + encodeURIComponent(w.id)); } catch { /* skip */ }
  }));
  renderFeed();

  setInterval(poll, 2000);
  setInterval(() => {
    for (const j of state.jobs.filter(active)) {
      const card = document.getElementById("job-" + j.pid);
      if (card) paintProgress(card._parts.progress, j);
    }
  }, 1000);
  poll();
}

boot();
