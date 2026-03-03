const API_BASE = "http://127.0.0.1:8000";
const DEFAULT_CITY = "Karlsruhe";
const DEFAULT_LAT = 49.0096;
const DEFAULT_LON = 8.4053;

function formatLocal(isoUtc) {
  try {
    if (!isoUtc) return "—";
    const d = new Date(isoUtc);
    if (Number.isNaN(d.getTime())) return isoUtc;
    return d.toLocaleString(undefined, {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return isoUtc ?? "—";
  }
}

function setStatus(msg, isError = false) {
  const statusEl = document.getElementById("status");
  if (!statusEl) return;

  if (!msg) {
    statusEl.hidden = true;
    statusEl.textContent = "";
    statusEl.style.borderColor = "";
    statusEl.style.background = "";
    return;
  }

  statusEl.hidden = false;
  statusEl.textContent = msg;

  // simple styling toggle for non-error
  if (!isError) {
    statusEl.style.borderColor = "rgba(120,230,255,.25)";
    statusEl.style.background = "rgba(120,230,255,.10)";
  }
}

function animateNumber(el, toValue, decimals = 2, durationMs = 350) {
  if (!el) return;
  const from = parseFloat(el.textContent);
  const start = Number.isFinite(from) ? from : toValue;
  const startTime = performance.now();

  function step(now) {
    const t = Math.min(1, (now - startTime) / durationMs);
    const eased = 1 - Math.pow(1 - t, 3); // easeOutCubic
    const v = start + (toValue - start) * eased;
    el.textContent = v.toFixed(decimals);
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function setBackgroundByTemp(tempC) {
  // super simple “mood”: cold / mild / warm
  // we modify the big background element by swapping gradient via inline style
  const bg = document.querySelector(".bg");
  if (!bg || !Number.isFinite(tempC)) return;

  if (tempC <= 2) {
    bg.style.background =
      "radial-gradient(520px 520px at 20% 25%, rgba(120,230,255,.35), transparent 60%)," +
      "radial-gradient(520px 520px at 85% 35%, rgba(160,120,255,.28), transparent 60%)," +
      "radial-gradient(520px 520px at 70% 90%, rgba(90,140,255,.22), transparent 60%)";
  } else if (tempC >= 22) {
    bg.style.background =
      "radial-gradient(520px 520px at 20% 25%, rgba(255,190,120,.40), transparent 60%)," +
      "radial-gradient(520px 520px at 85% 35%, rgba(255,120,120,.28), transparent 60%)," +
      "radial-gradient(520px 520px at 70% 90%, rgba(255,220,120,.22), transparent 60%)";
  } else {
    bg.style.background =
      "radial-gradient(520px 520px at 20% 25%, rgba(120,230,255,.30), transparent 60%)," +
      "radial-gradient(520px 520px at 85% 35%, rgba(255,190,120,.22), transparent 60%)," +
      "radial-gradient(520px 520px at 70% 90%, rgba(160,120,255,.22), transparent 60%)";
  }
}

async function loadForecast() {
  const btn = document.getElementById("btnRefresh");

  const locationEl = document.getElementById("location");
  const obsTimeEl = document.getElementById("obsTime");
  const nowTempEl = document.getElementById("nowTemp");

  const fcTimeEl = document.getElementById("fcTime");
  const predTempEl = document.getElementById("predTemp");
  const deltaEl = document.getElementById("delta");

  const endpointEl = document.getElementById("endpoint");

  if (!locationEl || !obsTimeEl || !nowTempEl || !fcTimeEl || !predTempEl) {
    setStatus("Fehlende DOM-Elemente: überprüfe IDs in index.html.", true);
    return;
  }

  const url =
    `${API_BASE}/forecast-live-current?city=${encodeURIComponent(DEFAULT_CITY)}` +
    `&lat=${DEFAULT_LAT}&lon=${DEFAULT_LON}`;

  //endpointEl.textContent = url;

  try {
    if (btn) {
      btn.disabled = true;
      btn.style.opacity = "0.75";
    }
    setStatus("Loading live forecast…");

    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);

    const data = await res.json();

    // write location + times
    locationEl.textContent = data.location ?? DEFAULT_CITY;
    obsTimeEl.textContent = formatLocal(data.observation_time_utc);
    fcTimeEl.textContent = formatLocal(data.forecast_time_utc);

    // numbers (animated)
    const now = Number(data.temp_now_c);
    const pred = Number(data.temp_pred_t_plus_24_c);
   // const delta = Number(data.delta_c);

    // if (Number.isFinite(now)) animateNumber(nowTempEl, now, 2);
    // else nowTempEl.textContent = "—";

    // if (Number.isFinite(pred)) animateNumber(predTempEl, pred, 2);
    // else predTempEl.textContent = "—";

    const nowRounded = Number.isFinite(now) ? Math.round(now) : null;
    const predRounded = Number.isFinite(pred) ? Math.round(pred) : null;

    if (nowRounded !== null) animateNumber(nowTempEl, nowRounded, 0);
    else nowTempEl.textContent = "—";

    if (predRounded !== null) animateNumber(predTempEl, predRounded, 0);
    else predTempEl.textContent = "—";

    function setBackgroundByTemp(tempC) {
  // Wir ändern nur die Overlay-Intensität, nicht das Background-Image.
  // Damit bleibt das Schloss dauerhaft sichtbar.
  const root = document.documentElement;
  if (!Number.isFinite(tempC)) return;

  // cold / mild / warm -> nur Opacity vom Overlay variieren
  if (tempC <= 2) {
    root.style.setProperty("--bgMoodOpacity", "0.75");
  } else if (tempC >= 22) {
    root.style.setProperty("--bgMoodOpacity", "0.55");
  } else {
    root.style.setProperty("--bgMoodOpacity", "0.65");
  }
}

    setStatus(""); // clear
  } catch (err) {
    console.error(err);
    setStatus(
      `Fehler beim Laden der Prognose. Läuft dein Backend? (${String(err?.message ?? err)})`,
      true
    );
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.style.opacity = "1";
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("btnRefresh");
  if (btn) btn.addEventListener("click", loadForecast);
  loadForecast();
});