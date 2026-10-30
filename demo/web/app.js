// NovaFlow browser demo: talks to the Python simulator running in worker.js,
// plays each round on the animated map and updates the charts.
const LABELS = {
  centralized: "Centralised", fedavg_iid: "FedAvg (IID junctions)", fedavg_noniid: "FedAvg (non-IID junctions)",
  fedavg_noniid_int8: "FedAvg + INT8 (non-IID)", pulsefed_noniid: "PulseFed (non-IID)",
  pulsefed_v2_noniid: "PulseFed v2 (non-IID)",
};
const COLORS = {
  centralized: "#1f77b4", fedavg_iid: "#2ca02c", fedavg_noniid: "#d62728", fedavg_noniid_int8: "#9467bd",
  pulsefed_noniid: "#ff7f0e", pulsefed_v2_noniid: "#8c564b",
};
const STAGES = [
  { name: "centralized", title: "Stage 1 · Centralised",
    text: "<b>Stage 1 · Centralised.</b> Every junction uploads its raw data; one model trains on all of it. This is the accuracy reference." },
  { name: "fedavg_noniid", title: "Stage 2 · FedAvg",
    text: "<b>Stage 2 · FedAvg.</b> Raw data stays at the junctions. Each round the server sends the model down, junctions train locally and the server averages their updates. Every byte is counted." },
  { name: "pulsefed_noniid", title: "Stage 3 · PulseFed",
    text: "<b>Stage 3 · PulseFed (our idea).</b> Junctions whose model barely changed skip the next round (at least 2 always talk, nobody sleeps more than 2 rounds)." },
  { name: "pulsefed_v2_noniid", title: "Stage 4 · PulseFed v2",
    text: "<b>Stage 4 · PulseFed v2 (follow-up).</b> Same skipping, but the server reuses a sleeping junction's last update so every road group still counts." },
];
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let INFO = null, INTRO = "", busy = false;

// ---------------------------------------------------------------- worker
const worker = new Worker("worker.js");
const handlers = {};
let jobSeq = 0;
function banner(text, cls = "") { const b = $("banner"); b.innerHTML = text; b.className = "banner " + cls; }
worker.onmessage = (ev) => {
  const m = typeof ev.data === "string" ? JSON.parse(ev.data) : ev.data;
  if (m.type === "status") banner(m.text);
  else if (m.type === "ready") onReady(m);
  else if (m.type === "error") {
    banner("Something went wrong: " + m.text, "err");
    if (m.job && handlers[m.job]) handlers[m.job].fail(m.text);
  } else if (m.job && handlers[m.job]) handlers[m.job].push(m);
};

function startRun(name, rounds, seed, title) {
  const job = "j" + (++jobSeq), q = [];
  let wake = null, err = null;
  handlers[job] = {
    push: (m) => { q.push(m); if (wake) { wake(); wake = null; } },
    fail: (t) => { err = t; if (wake) { wake(); wake = null; } },
  };
  worker.postMessage({ cmd: "run", name, rounds, seed, title, job });
  return {
    async next() {
      while (!q.length && !err) await new Promise((r) => (wake = r));
      if (err) throw new Error(err);
      return q.shift();
    },
  };
}

// ---------------------------------------------------------------- charts
function lineChart(canvas, title, yTitle) {
  return new Chart(canvas, {
    type: "line", data: { labels: [], datasets: [] },
    options: {
      animation: false, responsive: true, maintainAspectRatio: false,
      plugins: { title: { display: true, text: title }, legend: { labels: { boxWidth: 12, font: { size: 11 } } } },
      scales: { x: { title: { display: true, text: "round" } }, y: { title: { display: true, text: yTitle } } },
    },
  });
}
function barChart(canvas) {
  return new Chart(canvas, {
    type: "bar",
    data: { labels: ["junction 0 (quietest)", "junction 1", "junction 2", "junction 3 (busiest)"], datasets: [] },
    options: {
      animation: false, responsive: true, maintainAspectRatio: false,
      plugins: { title: { display: true, text: "Who pays the price? Error on each junction's own roads" },
                 legend: { labels: { boxWidth: 12, font: { size: 11 } } } },
      scales: { y: { title: { display: true, text: "test MAE (lower is better)" }, beginAtZero: true } },
    },
  });
}

function makeView(prefix) {
  return {
    map: $(prefix + "-map"), status: $(prefix + "-status"), part: $(prefix + "-part"), table: $(prefix + "-table"),
    mae: lineChart($(prefix + "-mae"), "Accuracy on unseen weeks, live", "test MAE (lower is better)"),
    mb: lineChart($(prefix + "-mb"), "Communication cost, live", "cumulative MB sent (measured)"),
    fair: barChart($(prefix + "-fair")),
  };
}

function resetView(v, rounds) {
  const labels = Array.from({ length: rounds }, (_, i) => String(i + 1));
  for (const c of [v.mae, v.mb]) { c.data.labels = labels; c.data.datasets = []; }
  v.mae.data.datasets.push({ label: "Naive 'next = last'", data: labels.map(() => INFO.persist),
    borderColor: "#9ca3af", borderDash: [6, 4], pointRadius: 0, borderWidth: 1.5 });
  v.mb.data.datasets.push({ label: `raw counts sent once (${INFO.raw_mb.toFixed(2)} MB)`,
    data: labels.map(() => INFO.raw_mb), borderColor: "#111827", borderDash: [6, 4], pointRadius: 0, borderWidth: 1.5 });
  v.fair.data.datasets = [];
  v.mae.update(); v.mb.update(); v.fair.update();
  v.part.innerHTML = ""; if (v.table) v.table.innerHTML = "";
}

function series(chart, name) {
  let ds = chart.data.datasets.find((d) => d.key === name);
  if (!ds) {
    ds = { key: name, label: LABELS[name], data: [], borderColor: COLORS[name], backgroundColor: COLORS[name],
           pointRadius: 2, borderWidth: 2 };
    chart.data.datasets.push(ds);
  }
  return ds;
}

function participation(v, name, rows) {
  if (!name.startsWith("pulsefed")) return;
  const K = 4, colors = ["#f1f5f9", "#2a9d8f", "#f59e0b"];
  let html = `<b>${LABELS[name]}:</b> green = talked, amber = forced back in, grey = asleep · ` +
             `${rows.reduce((s, r) => s + r.active_clients.length, 0)} uploads vs ${K * rows.length} for FedAvg`;
  for (let k = 0; k < K; k++) {
    html += `<div class="row"><span class="lab">junction ${k}</span>`;
    for (const r of rows) {
      const st = r.active_clients.includes(k) ? ((r.forced_clients || []).includes(k) ? 2 : 1) : 0;
      html += `<span class="cell" style="background:${colors[st]}" title="round ${r.round}"></span>`;
    }
    html += "</div>";
  }
  v.part.innerHTML += `<div style="margin-top:8px">${html}</div>`;
}

function fmtMB(res) {
  return res.comm && res.comm.total_bytes ? (res.comm.total_bytes / 1e6).toFixed(2) + " MB"
                                           : `raw upload ${INFO.raw_mb.toFixed(2)} MB`;
}

// ---------------------------------------------------------------- playing a run
async function playRun(v, name, title, rounds, seed, frameMs, text, prefixText = "") {
  const run = startRun(name, rounds, seed, title), rows = [];
  while (true) {
    const m = await run.next();
    if (m.type === "round") {
      rows.push(m.row);
      v.map.innerHTML = m.scene;
      series(v.mae, name).data.push(m.row.test_mae);
      if (m.row.cumulative_bytes !== undefined) series(v.mb, name).data.push(m.row.cumulative_bytes / 1e6);
      v.mae.update(); v.mb.update();
      const mb = m.row.cumulative_bytes !== undefined ? ` · ${(m.row.cumulative_bytes / 1e6).toFixed(2)} MB sent` : "";
      v.status.innerHTML = `${text}<br>${prefixText}round ${m.row.round}/${rounds} · test MAE ${m.row.test_mae.toFixed(4)}${mb}`;
      await sleep(frameMs());
    } else if (m.type === "done") {
      v.map.innerHTML = m.scene;
      const pc = m.res.final.per_client || [];
      if (pc.length) {
        v.fair.data.datasets.push({ label: LABELS[name], data: pc.map((c) => c.mae), backgroundColor: COLORS[name] });
        v.fair.update();
      }
      participation(v, name, rows);
      v.status.innerHTML = `${text}<br>Finished · test MAE <b>${m.res.final.mae.toFixed(4)}</b>, RMSE ` +
        `<b>${m.res.final.rmse.toFixed(4)}</b> · data: <b>${fmtMB(m.res)}</b>. Badges show each junction's own error.`;
      await sleep(frameMs() * 1.5);
      return m.res;
    }
  }
}

function speed(nameAttr) { return () => Number(document.querySelector(`input[name=${nameAttr}]:checked`).value); }

async function runTour() {
  if (busy) return; busy = true; setButtons();
  const v = VIEWS.tour, rounds = Number($("tour-rounds").value), seed = Number($("tour-seed").value);
  resetView(v, rounds);
  const results = {};
  try {
    for (let i = 0; i < STAGES.length; i++) {
      const s = STAGES[i];
      results[s.name] = await playRun(v, s.name, s.title, rounds, seed, speed("tour-speed"), s.text, `Stage ${i + 1}/4 · `);
    }
    let html = "<tr><th>Method</th><th>Test MAE</th><th>Test RMSE</th><th>Data sent</th><th>Worst junction MAE</th></tr>";
    for (const s of STAGES) {
      const r = results[s.name], pc = (r.final.per_client || []).map((c) => c.mae);
      html += `<tr><td>${LABELS[s.name]}</td><td>${r.final.mae.toFixed(4)}</td><td>${r.final.rmse.toFixed(4)}</td>` +
              `<td>${fmtMB(r)}</td><td>${pc.length ? Math.max(...pc).toFixed(4) : ""}</td></tr>`;
    }
    html += `<tr><td>Naive 'next = last'</td><td>${INFO.persist.toFixed(4)}</td><td></td><td></td><td></td></tr>`;
    v.table.innerHTML = html;
    v.status.innerHTML = "<b>Tour complete.</b> Compare accuracy, bytes and fairness below. (Live runs use one seed and " +
      "fewer rounds; the <i>Full results</i> tab has the 3-seed, 30-round numbers.)";
  } catch (e) { v.status.textContent = "Run failed: " + e.message; }
  busy = false; setButtons();
}

async function runSingle() {
  if (busy) return; busy = true; setButtons();
  const v = VIEWS.single, name = $("single-method").value;
  const rounds = Number($("single-rounds").value), seed = Number($("single-seed").value);
  resetView(v, rounds);
  try { await playRun(v, name, LABELS[name], rounds, seed, speed("single-speed"), `<b>${LABELS[name]}</b>`); }
  catch (e) { v.status.textContent = "Run failed: " + e.message; }
  busy = false; setButtons();
}

// ---------------------------------------------------------------- page setup
function setButtons() { for (const id of ["tour-run", "single-run"]) $(id).disabled = !INFO || busy; }

let VIEWS = {};
function onReady(m) {
  INFO = m.info; INTRO = m.intro;
  banner("✅ Simulator loaded in your browser. Press <b>Run the tour</b> to start.", "ok");
  $("tour-map").innerHTML = INTRO; $("single-map").innerHTML = INTRO;
  const meta = INFO.meta || {};
  if (meta.num_sensors) $("m-sensors").textContent = meta.num_sensors;
  if (meta.num_train_steps) $("m-train").textContent = meta.num_train_steps;
  if (meta.num_test_steps) $("m-test").textContent = meta.num_test_steps;
  setButtons();
}

document.addEventListener("DOMContentLoaded", () => {
  for (const b of document.querySelectorAll(".tab-btn")) {
    b.onclick = () => {
      document.querySelectorAll(".tab-btn").forEach((x) => x.classList.toggle("active", x === b));
      document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab));
    };
  }
  for (const p of ["tour", "single"]) {
    $(p + "-rounds").oninput = (e) => ($(p + "-rounds-v").textContent = e.target.value);
  }
  $("single-method").innerHTML = Object.entries(LABELS)
    .map(([k, v]) => `<option value="${k}" ${k === "fedavg_noniid" ? "selected" : ""}>${v}</option>`).join("");
  VIEWS = { tour: makeView("tour"), single: makeView("single") };
  $("tour-run").onclick = runTour;
  $("single-run").onclick = runSingle;

  fetch("results_summary.json").then((r) => r.json()).then((s) => {
    let html = "<tr><th>Method</th><th>MAE (mean ± std)</th><th>RMSE (mean)</th><th>Total MB</th><th>Seeds</th></tr>";
    for (const r of s.rows) {
      html += `<tr><td>${r.method}</td><td>${r.mae}</td><td>${r.rmse}</td><td>${r.mb ?? ""}</td><td>${r.seeds}</td></tr>`;
    }
    $("full-table").innerHTML = html;
    $("gallery").innerHTML = s.figures.map((f) =>
      `<figure><img src="figures/${f.file}" alt="${f.caption}" loading="lazy"><figcaption>${f.caption}</figcaption></figure>`).join("");
  }).catch(() => { $("full-table").innerHTML = "<tr><td>Pre-computed results not found.</td></tr>"; });
});
