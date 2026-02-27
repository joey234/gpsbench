/* ===== Globals ===== */
const { models, taskMeta, providerColors, categories } = BENCHMARK_DATA;
let sortKey = "overall";
let radarChart = null;
let selectedModels = new Set(["gpt-5-mini", "gpt-5.1", "gemini-2.5-pro", "gpt-4.1"]);

const acc = v => v >= 70 ? "c-hi" : v >= 50 ? "c-md" : "c-lo";
const prov = m => {
  const c = providerColors[m.provider] || "#888";
  return `<span class="prov" style="background:${c}14;color:${c};border:1px solid ${c}30">${m.provider}</span>`;
};

/* ===== Tab System ===== */
const tabState = { overview: false, leaderboard: false };

function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
  document.getElementById(`panel-${tabId}`).classList.add('active');

  if (!tabState[tabId]) {
    initTabContent(tabId);
    tabState[tabId] = true;
  }

  history.replaceState(null, '', `#${tabId}`);
  window.scrollTo(0, 0);
}

function initTabContent(tabId) {
  if (tabId === 'overview') {
    renderTeaser();
    renderTaxonomy();
    bindOverviewEvents();
  } else if (tabId === 'leaderboard') {
    renderLeaderboard();
    renderTrackTables();
    renderHeatmap();
    renderChips();
    renderRadar();
    bindLeaderboardEvents();
  }
}

function resolveTabFromHash() {
  const hash = window.location.hash.replace('#', '');
  const leaderboardSections = ['leaderboard', 'tracks', 'heatmap', 'compare'];
  if (leaderboardSections.includes(hash)) return 'leaderboard';
  return 'overview';
}

/* ===== Init ===== */
document.addEventListener("DOMContentLoaded", () => {
  // Bind tab buttons
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // BibTeX link should switch to overview tab and scroll to citation
  document.getElementById('link-bibtex').addEventListener('click', (e) => {
    e.preventDefault();
    if (!tabState.overview) {
      switchTab('overview');
    } else {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      document.querySelector('[data-tab="overview"]').classList.add('active');
      document.getElementById('panel-overview').classList.add('active');
    }
    requestAnimationFrame(() => {
      document.getElementById('citation').scrollIntoView({ behavior: 'smooth' });
    });
  });

  // Init based on hash
  const initialTab = resolveTabFromHash();
  switchTab(initialTab);

  // If hash points to a specific section, scroll to it
  const hash = window.location.hash.replace('#', '');
  if (hash && hash !== initialTab) {
    requestAnimationFrame(() => {
      const el = document.getElementById(hash);
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    });
  }
});

/* ===== Overview Events ===== */
function bindOverviewEvents() {
  document.getElementById("copy-bib").addEventListener("click", () => {
    const text = document.getElementById("bib-text").textContent;
    navigator.clipboard.writeText(text).then(() => {
      const btn = document.getElementById("copy-bib");
      btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>';
      setTimeout(() => {
        btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>';
      }, 1500);
    });
  });
}

/* ===== Leaderboard Events ===== */
function bindLeaderboardEvents() {
  document.getElementById("search-input").addEventListener("input", renderLeaderboard);
  document.getElementById("provider-filter").addEventListener("change", renderLeaderboard);

  document.querySelectorAll(".lb-table th[data-sort]").forEach(th => {
    th.addEventListener("click", () => {
      document.querySelectorAll(".lb-table th").forEach(t => t.classList.remove("sort-active"));
      th.classList.add("sort-active");
      sortKey = th.dataset.sort;
      renderLeaderboard();
    });
  });
}

/* ===== Teaser Bar Chart ===== */
function renderTeaser() {
  const sorted = [...models].sort((a, b) => a.overall - b.overall);
  new Chart(document.getElementById("chart-teaser"), {
    type: "bar",
    data: {
      labels: sorted.map(m => m.name),
      datasets: [
        {
          label: "Pure GPS",
          data: sorted.map(m => m.pure_gps.accuracy),
          backgroundColor: "rgba(37,99,235,0.55)",
          borderColor: "rgba(37,99,235,0.85)",
          borderWidth: 1,
          borderRadius: 2,
        },
        {
          label: "Applied",
          data: sorted.map(m => m.applied.accuracy),
          backgroundColor: "rgba(124,58,237,0.45)",
          borderColor: "rgba(124,58,237,0.8)",
          borderWidth: 1,
          borderRadius: 2,
        }
      ]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top", labels: { color: "#5a6370", font: { size: 11 }, padding: 16, usePointStyle: true, pointStyleWidth: 8 } },
        tooltip: {
          backgroundColor: "#ffffff",
          titleColor: "#1a1f2e",
          bodyColor: "#5a6370",
          borderColor: "#e0e3e8",
          borderWidth: 1,
          callbacks: { label: c => `${c.dataset.label}: ${c.raw.toFixed(1)}%` }
        }
      },
      scales: {
        x: { stacked: false, beginAtZero: true, max: 100, ticks: { color: "#8a92a0", font: { size: 10 }, callback: v => v + "%" }, grid: { color: "#e0e3e830" } },
        y: { ticks: { color: "#3a4050", font: { size: 11, weight: "500" } }, grid: { display: false } }
      }
    }
  });
  document.getElementById("chart-teaser").parentElement.style.height = (sorted.length * 42 + 60) + "px";
}

/* ===== Taxonomy ===== */
function renderTaxonomy() {
  const container = document.getElementById("taxonomy-container");

  // Group tasks by track then by category
  const tracks = { pure_gps: {}, applied: {} };
  Object.entries(taskMeta).forEach(([key, task]) => {
    if (!task.category) return;
    if (!tracks[task.track]) return;
    if (!tracks[task.track][task.category]) tracks[task.track][task.category] = [];
    tracks[task.track][task.category].push({ key, ...task });
  });

  function renderTrackBranch(trackId, trackLabel, trackClass) {
    const trackCats = tracks[trackId];
    const catKeys = Object.keys(trackCats);
    const taskCount = catKeys.reduce((sum, ck) => sum + trackCats[ck].length, 0);
    const catDescription = trackId === 'pure_gps' ? 'Coordinate manipulation' : 'Geographic reasoning';

    let html = `<div class="tax-track">
      <div class="tax-node tax-node-track ${trackClass}">
        <strong>${trackLabel}</strong>
        <span>${taskCount} tasks &middot; ${catDescription}</span>
      </div>
      <div class="tax-categories">`;

    catKeys.forEach(catKey => {
      const cat = categories[catKey];
      const tasks = trackCats[catKey];
      html += `<div class="tax-category">
        <div class="tax-cat-header">
          <span class="tax-cat-name">${cat.name}</span>
          <span class="tax-cat-desc">${cat.description}</span>
        </div>
        <div class="tax-tasks">`;
      tasks.forEach(t => {
        html += `<div class="tax-task">
          <span class="task-name">${t.name}</span>
          <span class="task-desc">${t.description}</span>
          <span class="task-eval">${t.evaluationType}</span>
          <span class="task-meta">${t.samples}</span>
        </div>`;
      });
      html += `</div></div>`;
    });

    html += `</div></div>`;
    return html;
  }

  container.innerHTML = `<div class="tax-tree">
    <div class="tax-root">
      <div class="tax-node tax-node-root">
        <strong>GPSBench</strong>
        <span>17 tasks &middot; 57,800 samples</span>
      </div>
    </div>
    <div class="tax-branches">
      ${renderTrackBranch('pure_gps', 'Pure GPS Track', 'pure')}
      ${renderTrackBranch('applied', 'Applied Track', 'applied')}
    </div>
  </div>`;
}

/* ===== Leaderboard ===== */
function renderLeaderboard() {
  const q = document.getElementById("search-input").value.toLowerCase();
  const pf = document.getElementById("provider-filter").value;

  let list = models.filter(m => {
    const ms = m.name.toLowerCase().includes(q) || m.fullName.toLowerCase().includes(q);
    const mp = pf === "all" || m.provider === pf;
    return ms && mp;
  });

  list.sort((a, b) => {
    const va = sortKey === "overall" ? a.overall : sortKey === "pure_gps" ? a.pure_gps.accuracy : a.applied.accuracy;
    const vb = sortKey === "overall" ? b.overall : sortKey === "pure_gps" ? b.pure_gps.accuracy : b.applied.accuracy;
    return vb - va;
  });

  document.getElementById("lb-tbody").innerHTML = list.map((m, i) => {
    const r = i + 1;
    const rc = r <= 3 ? `rb-${r}` : "rb-n";
    const pw = m.pure_gps.accuracy / 100 * 80;
    const aw = m.applied.accuracy / 100 * 80;
    return `<tr>
      <td class="col-rank"><span class="rb ${rc}">${r}</span></td>
      <td class="col-model"><div class="mn">${m.name}</div><div class="mf">${m.fullName}</div></td>
      <td class="col-prov">${prov(m)}</td>
      <td class="col-num"><span class="acc ${acc(m.overall)}">${m.overall.toFixed(2)}%</span></td>
      <td class="col-num"><span class="acc ${acc(m.pure_gps.accuracy)}">${m.pure_gps.accuracy.toFixed(2)}%</span></td>
      <td class="col-num"><span class="acc ${acc(m.applied.accuracy)}">${m.applied.accuracy.toFixed(2)}%</span></td>
      <td class="col-bar">
        <div class="ibar">
          <div class="ibar-seg" style="width:${pw}px;background:rgba(37,99,235,0.5)"></div>
          <div class="ibar-seg" style="width:${aw}px;background:rgba(124,58,237,0.4)"></div>
          <span class="ibar-lbl">${m.overall.toFixed(1)}</span>
        </div>
      </td>
    </tr>`;
  }).join("");
}

/* ===== Track Tables ===== */
function renderTrackTables() {
  const sorted = [...models].sort((a, b) => b.pure_gps.accuracy - a.pure_gps.accuracy);
  const sortedA = [...models].sort((a, b) => b.applied.accuracy - a.applied.accuracy);

  const head = `<thead><tr><th>#</th><th>Model</th><th>Accuracy</th></tr></thead>`;

  document.getElementById("table-pure").innerHTML = head + "<tbody>" +
    sorted.map((m, i) => `<tr><td style="width:30px;text-align:center;color:var(--text3);font-family:var(--mono)">${i+1}</td><td>${m.name}</td><td><span class="acc ${acc(m.pure_gps.accuracy)}">${m.pure_gps.accuracy.toFixed(2)}%</span></td></tr>`).join("") + "</tbody>";

  document.getElementById("table-applied").innerHTML = head + "<tbody>" +
    sortedA.map((m, i) => `<tr><td style="width:30px;text-align:center;color:var(--text3);font-family:var(--mono)">${i+1}</td><td>${m.name}</td><td><span class="acc ${acc(m.applied.accuracy)}">${m.applied.accuracy.toFixed(2)}%</span></td></tr>`).join("") + "</tbody>";
}

/* ===== Heatmap ===== */
function renderHeatmap() {
  const sorted = [...models].sort((a, b) => b.overall - a.overall);
  const pureTasks = Object.entries(taskMeta).filter(([,v]) => v.track === "pure_gps");
  const appTasks = Object.entries(taskMeta).filter(([,v]) => v.track === "applied");

  const head = `<thead><tr><th>Task</th>${sorted.map(m => `<th>${m.name.replace(/ /g, '<br>')}</th>`).join("")}</tr></thead>`;

  function makeRows(tasks, label) {
    let html = `<tr class="hm-divider"><td colspan="${sorted.length + 1}">${label}</td></tr>`;
    tasks.forEach(([key, info]) => {
      html += `<tr><td>${info.name}</td>`;
      sorted.forEach(m => {
        const v = m.tasks[key];
        if (v === undefined) { html += `<td style="color:var(--text3)">—</td>`; return; }
        const norm = v / 100;
        const bg = cellColor(norm);
        html += `<td style="background:${bg};color:#1a1f2e">${v.toFixed(1)}</td>`;
      });
      html += `</tr>`;
    });
    return html;
  }

  document.getElementById("heatmap-table").innerHTML = head + "<tbody>" +
    makeRows(pureTasks, "Pure GPS Track") +
    makeRows(appTasks, "Applied Track") +
    "</tbody>";
}

function cellColor(norm) {
  if (norm < 0.5) {
    const t = norm / 0.5;
    const r = Math.round(254 + (253 - 254) * t);
    const g = Math.round(215 + (236 - 215) * t);
    const b = Math.round(215 + (200 - 215) * t);
    return `rgba(${r},${g},${b},0.8)`;
  } else {
    const t = (norm - 0.5) / 0.5;
    const r = Math.round(253 - (253 - 187) * t);
    const g = Math.round(236 + (247 - 236) * t);
    const b = Math.round(200 - (200 - 208) * t);
    return `rgba(${r},${g},${b},0.85)`;
  }
}

/* ===== Chips + Radar ===== */
function renderChips() {
  const bar = document.getElementById("chip-bar");
  bar.innerHTML = models.map(m =>
    `<button class="chip ${selectedModels.has(m.id) ? 'on' : ''}" data-id="${m.id}">${m.name}</button>`
  ).join("");

  bar.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const id = chip.dataset.id;
      if (selectedModels.has(id)) { if (selectedModels.size > 1) selectedModels.delete(id); }
      else { if (selectedModels.size < 6) selectedModels.add(id); }
      renderChips();
      renderRadar();
    });
  });
}

function renderRadar() {
  const sel = models.filter(m => selectedModels.has(m.id));
  const allTasks = Object.entries(taskMeta)
    .filter(([, v]) => v.track === 'pure_gps')
    .map(([k]) => k)
    .concat(
      Object.entries(taskMeta)
        .filter(([, v]) => v.track === 'applied')
        .map(([k]) => k)
    );
  const labels = allTasks.map(t => taskMeta[t]?.name || t);

  const colors = [
    {bg:"rgba(37,99,235,0.08)",b:"#2563eb"},
    {bg:"rgba(124,58,237,0.08)",b:"#7c3aed"},
    {bg:"rgba(5,150,105,0.08)",b:"#059669"},
    {bg:"rgba(217,119,6,0.08)",b:"#d97706"},
    {bg:"rgba(220,38,38,0.08)",b:"#dc2626"},
    {bg:"rgba(219,39,119,0.08)",b:"#db2777"},
  ];

  const datasets = sel.map((m, i) => ({
    label: m.name,
    data: allTasks.map(t => m.tasks[t] || 0),
    backgroundColor: colors[i%colors.length].bg,
    borderColor: colors[i%colors.length].b,
    borderWidth: 2,
    pointRadius: 2.5,
    pointBackgroundColor: colors[i%colors.length].b,
  }));

  if (radarChart) radarChart.destroy();
  radarChart = new Chart(document.getElementById("chart-radar"), {
    type: "radar",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { position:"top", labels: { color:"#5a6370", font:{size:10.5}, padding:12, usePointStyle:true, pointStyleWidth:7 } },
        tooltip: {
          backgroundColor:"#ffffff", titleColor:"#1a1f2e", bodyColor:"#5a6370",
          borderColor:"#e0e3e8", borderWidth:1,
          callbacks: { label: c => `${c.dataset.label}: ${c.raw.toFixed(1)}%` }
        }
      },
      scales: {
        r: {
          beginAtZero: true, max: 100,
          ticks: { stepSize:25, color:"#8a92a0", font:{size:8}, backdropColor:"transparent" },
          grid: { color:"#e0e3e860" },
          angleLines: { color:"#e0e3e860" },
          pointLabels: { color:"#5a6370", font:{size:8.5} }
        }
      }
    }
  });
}
