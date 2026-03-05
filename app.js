// app.js — Frontend client for Render-hosted FastAPI
// Adds: timeout + retry so Render Free "spin down" (cold start) won't break the UI.

const API_BASE = "https://weather-app-03v8.onrender.com";
const DEFAULT_CITY = "Karlsruhe";
const DEFAULT_LAT = 49.0096;
const DEFAULT_LON = 8.4053;

/** Format ISO-UTC string into local time string (best effort). */
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

  if (!isError) {
    statusEl.style.borderColor = "rgba(120,230,255,.25)";
    statusEl.style.background = "rgba(120,230,255,.10)";
  } else {
    statusEl.style.borderColor = "rgba(255,120,120,.30)";
    statusEl.style.background = "rgba(255,120,120,.10)";
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

/**
 * Mood overlay control (keeps your background image visible if you use CSS var).
 */
function setBackgroundByTemp(tempC) {
  const root = document.documentElement;
  if (!Number.isFinite(tempC)) return;

  if (tempC <= 2) root.style.setProperty("--bgMoodOpacity", "0.75");
  else if (tempC >= 22) root.style.setProperty("--bgMoodOpacity", "0.55");
  else root.style.setProperty("--bgMoodOpacity", "0.65");
}

/** Sleep helper */
function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/**
 * Fetch with timeout + retry (handles Render free-tier cold start).
 * - Retries for network errors, timeouts, and 5xx responses.
 * - Shows progress via setStatus().
 */
async function fetchWithRetry(url, opts = {}) {
  const {
    retries = 2, // total extra tries (2 => up to 3 attempts)
    timeoutMs = 60000,
    backoffMs = 3000,
  } = opts;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);

      if (attempt === 0) {
        setStatus("Loading live forecast…");
      } else {
        setStatus(
          `Backend wacht gerade auf… Versuch ${attempt + 1}/${retries + 1} (kann bis ~50s dauern)`
        );
      }

      const res = await fetch(url, {
        signal: controller.signal,
        headers: { accept: "application/json" },
      });

      clearTimeout(timer);

      // Retry on server errors (5xx). For 4xx, fail fast.
      if (!res.ok) {
        if (res.status >= 500 && res.status <= 599 && attempt < retries) {
          await sleep(backoffMs);
          continue;
        }
        throw new Error(`HTTP ${res.status} ${res.statusText}`);
      }

      return res;
    } catch (err) {
      const msg = String(err?.message ?? err);

      // Abort (timeout) or network failure -> retry if attempts left
      const isTimeout = msg.toLowerCase().includes("aborted");
      const isNetwork =
        msg.toLowerCase().includes("failed to fetch") ||
        msg.toLowerCase().includes("network") ||
        msg.toLowerCase().includes("load failed");

      if (attempt < retries && (isTimeout || isNetwork)) {
        await sleep(backoffMs);
        continue;
      }

      throw err;
    }
  }

  // should never reach
  throw new Error("Fetch failed after retries");
}

async function loadForecast() {
  const btn = document.getElementById("btnRefresh");

  const locationEl = document.getElementById("location");
  const obsTimeEl = document.getElementById("obsTime");
  const nowTempEl = document.getElementById("nowTemp");
  const fcTimeEl = document.getElementById("fcTime");
  const predTempEl = document.getElementById("predTemp");

  if (!locationEl || !obsTimeEl || !nowTempEl || !fcTimeEl || !predTempEl) {
    setStatus("Fehlende DOM-Elemente: überprüfe IDs in index.html.", true);
    return;
  }

  const url =
    `${API_BASE}/forecast-live-current?city=${encodeURIComponent(DEFAULT_CITY)}` +
    `&lat=${DEFAULT_LAT}&lon=${DEFAULT_LON}`;

  try {
    if (btn) {
      btn.disabled = true;
      btn.style.opacity = "0.75";
    }

    // Use retry-capable fetch
    const res = await fetchWithRetry(url, {
      retries: 2,
      timeoutMs: 60000, // give Render time to wake up
      backoffMs: 3000,
    });

    const data = await res.json();

    // write location + times
    locationEl.textContent = data.location ?? DEFAULT_CITY;
    obsTimeEl.textContent = formatLocal(data.observation_time_utc);
    fcTimeEl.textContent = formatLocal(data.forecast_time_utc);

    // numbers (rounded to int like your current version)
    const now = Number(data.temp_now_c);
    const pred = Number(data.temp_pred_t_plus_24_c);

    const nowRounded = Number.isFinite(now) ? Math.round(now) : null;
    const predRounded = Number.isFinite(pred) ? Math.round(pred) : null;

    if (nowRounded !== null) animateNumber(nowTempEl, nowRounded, 0);
    else nowTempEl.textContent = "—";

    if (predRounded !== null) animateNumber(predTempEl, predRounded, 0);
    else predTempEl.textContent = "—";

    // optional mood
    if (Number.isFinite(now)) setBackgroundByTemp(now);

    setStatus(""); // clear
  } catch (err) {
    console.error(err);
    setStatus(
      `Fehler beim Laden der Prognose. Backend evtl. im Cold-Start oder nicht erreichbar. (${String(
        err?.message ?? err
      )})`,
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