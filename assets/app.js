/* ==========================================================================
   Retail Demand & Inventory Analytics — Application Logic
   Loads Python-processed outputs and does all filtering/aggregation
   client-side so the global filter bar can update every chart live.
   ========================================================================== */

/* ---------------------------------------------------------------------- */
/* 0. Style tokens pulled from CSS so Chart.js matches the design system   */
/* ---------------------------------------------------------------------- */
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const COLORS = {
  text: '#5B6572', textDark: '#16202E', border: '#E2E5EB',
  accent: '#2453C9', accentDark: '#1B3D9C',
  good: '#1B9E63', warn: '#C68A00', bad: '#D5484B',
  chart: ['#2453C9', '#14A0A0', '#8B5CF6', '#C68A00', '#D5484B', '#4B7BE5'],
};
Chart.defaults.font.family = "'Inter', -apple-system, sans-serif";
Chart.defaults.color = COLORS.text;
Chart.defaults.font.size = 11.5;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.boxWidth = 8;
Chart.defaults.plugins.legend.labels.boxHeight = 8;
Chart.defaults.elements.point.radius = 2;

const GRID = { color: '#EEF0F3', drawTicks: false };
const AXIS_TICK = { color: COLORS.text, font: { size: 10.5 } };

/* ---------------------------------------------------------------------- */
/* 1. State                                                                */
/* ---------------------------------------------------------------------- */
let processedRows = [];   // full row-level dataset (outputs/processed_data.csv)
let predictionRows = [];  // held-out test set actual vs predicted (outputs/predictions.csv)
let modelMetrics = {};
let featureImportance = [];
let correlationData = [];
let insights = [];

let dims = { stores: [], products: [], categories: [], regions: [], seasons: [] };
let dateBounds = { min: null, max: null };

const filters = {
  dateFrom: null, dateTo: null,
  store: 'all', product: 'all', category: 'all', region: 'all',
  season: 'all', promo: 'all', epidemic: 'all',
};
const page5Filters = { store: 'all', product: 'all', category: 'all', region: 'all' };

let trendGranularity = 'daily';
let currentPage = 'overview';
const chartInstances = {};

/* ---------------------------------------------------------------------- */
/* 2. Utilities                                                            */
/* ---------------------------------------------------------------------- */
function fmtInt(n) { return Math.round(n).toLocaleString('en-US'); }
function fmtNum(n, d = 1) { return Number(n).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }); }
function fmtPct(n, d = 1) { return `${Number(n).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d })}%`; }
function toDateObj(iso) { const [y, m, d] = iso.split('-').map(Number); return new Date(y, m - 1, d); }
function weekKey(iso) { const d = toDateObj(iso); const dow = (d.getDay() + 6) % 7; d.setDate(d.getDate() - dow); return d.toISOString().slice(0, 10); }
function monthKey(iso) { return iso.slice(0, 7); }
function uniqueSorted(rows, field) { return [...new Set(rows.map(r => r[field]))].sort(); }

function sampleArray(arr, n) {
  if (arr.length <= n) return arr;
  const step = arr.length / n;
  const out = [];
  for (let i = 0; i < n; i++) out.push(arr[Math.floor(i * step)]);
  return out;
}

function groupSum(rows, keyFn, valField) {
  const m = new Map();
  for (const r of rows) {
    const k = keyFn(r);
    m.set(k, (m.get(k) || 0) + (r[valField] || 0));
  }
  return m;
}

function groupAgg(rows, keyFn) {
  // returns Map key -> {n, sum:{}, }
  const m = new Map();
  for (const r of rows) {
    const k = keyFn(r);
    if (!m.has(k)) m.set(k, []);
    m.get(k).push(r);
  }
  return m;
}

function mean(arr) { return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0; }

function destroyChart(id) { if (chartInstances[id]) { chartInstances[id].destroy(); delete chartInstances[id]; } }

function makeChart(canvasId, config) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId).getContext('2d');
  chartInstances[canvasId] = new Chart(ctx, config);
  return chartInstances[canvasId];
}

/* ---------------------------------------------------------------------- */
/* 3. Filtering engine (shared by processedRows and predictionRows —      */
/*    both use identical field names: Date, Store, Product, Category,     */
/*    Region, Season, Promo, Epidemic)                                    */
/* ---------------------------------------------------------------------- */
function applyFilters(rows, f) {
  return rows.filter(r => {
    if (f.dateFrom && new Date(r.Date) < new Date(f.dateFrom)) return false;
    if (f.dateTo && r.Date > f.dateTo) return false;
    if (f.store !== 'all' && r.Store !== f.store) return false;
    if (f.product !== 'all' && r.Product !== f.product) return false;
    // Đảm bảo r.Promo và r.Epidemic tồn tại trong CSV
    if (f.promo !== 'all' && String(r.Promo) !== f.promo) return false;
    if (f.epidemic !== 'all' && String(r.Epidemic) !== f.epidemic) return false;
    return true;
  });
}

function applyPage5Filters(rows, f) {
  return rows.filter(r => {
    if (f.store !== 'all' && r.Store !== f.store) return false;
    if (f.product !== 'all' && r.Product !== f.product) return false;
    if (f.category !== 'all' && r.Category !== f.category) return false;
    if (f.region !== 'all' && r.Region !== f.region) return false;
    return true;
  });
}

/* ---------------------------------------------------------------------- */
/* 4. Load data                                                            */
/* ---------------------------------------------------------------------- */
function loadCSV(path) {
  return new Promise((resolve, reject) => {
    Papa.parse(path, {
      download: true, header: true, dynamicTyping: true, skipEmptyLines: true,
      complete: (res) => resolve(res.data),
      error: reject,
    });
  });
}
function loadJSON(path) { return fetch(path).then(r => r.json()); }

async function init() {
  try {
    const [proc, pred, mm, fi, corr, ins] = await Promise.all([
      loadCSV('outputs/processed_data.csv'),
      loadCSV('outputs/predictions.csv'),
      loadJSON('outputs/model_metrics.json'),
      loadJSON('outputs/feature_importance.json'),
      loadJSON('outputs/correlation.json'),
      loadJSON('outputs/insights.json'),
    ]);
    processedRows = proc.filter(r => r.Date);
    predictionRows = pred.filter(r => r.Date);
    modelMetrics = mm; featureImportance = fi; correlationData = corr; insights = ins;

    dims.stores = uniqueSorted(processedRows, 'Store');
    dims.products = uniqueSorted(processedRows, 'Product');
    dims.categories = uniqueSorted(processedRows, 'Category');
    dims.regions = uniqueSorted(processedRows, 'Region');
    dims.seasons = uniqueSorted(processedRows, 'Season');
        // Kiểm tra nếu có dữ liệu mới tính toán ngày
    if (processedRows.length > 0) {
      dateBounds.min = processedRows.reduce((a, r) => (r.Date < a ? r.Date : a), processedRows[0].Date);
      dateBounds.max = processedRows.reduce((a, r) => (r.Date > a ? r.Date : a), processedRows[0].Date);
      
      // Gán ngày mặc định cho bộ lọc
      filters.dateFrom = dateBounds.min;
      filters.dateTo = dateBounds.max;
    }

    const infoEl = document.getElementById('datasetInfo');
    if (infoEl) {
      infoEl.innerHTML =
        `<b>Dataset</b><br>${fmtInt(processedRows.length)} rows &middot; ${dateBounds.min} to ${dateBounds.max}<br>` +
        `${dims.stores.length} stores &middot; ${dims.products.length} products`;
    }

    document.getElementById('datasetInfo').innerHTML =
      `<b>Dataset</b><br>${fmtInt(processedRows.length)} rows &middot; ${dateBounds.min} to ${dateBounds.max}<br>` +
      `${dims.stores.length} stores &middot; ${dims.products.length} products`;

    buildNav();
    renderFilterBar(activeFilterContainerId());
    buildSelectPanel();
    renderPage(currentPage);

    document.getElementById('loadingScreen').style.display = 'none';
  } catch (err) {
    document.getElementById('loadingScreen').innerHTML =
      `<div style="max-width:360px;text-align:center;font-size:12.5px;color:#5B6572;">
         <b style="color:#D5484B;">Couldn't load dataset.</b><br><br>
         This dashboard fetches CSV/JSON files from <code>outputs/</code>, which most browsers block
         over a plain <code>file://</code> path. Serve the folder locally
         (e.g. <code>python3 -m http.server</code>) or deploy it (GitHub Pages / Vercel) and reload.
         <br><br><span style="color:#8A93A1;">${err && err.message ? err.message : ''}</span>
       </div>`;
    console.error(err);
  }
}

/* ---------------------------------------------------------------------- */
/* 5. Navigation                                                           */
/* ---------------------------------------------------------------------- */
const FILTERBAR_BY_PAGE = {
  overview: 'filterbarGlobal', inventory: 'filterbarGlobal2',
  drivers: 'filterbarGlobal3', forecast: 'filterbarGlobal4',
};
function activeFilterContainerId() { return FILTERBAR_BY_PAGE[currentPage] || null; }

function buildNav() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
      document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
      currentPage = item.dataset.page;
      document.getElementById('page-' + currentPage).classList.add('active');
      if (activeFilterContainerId()) renderFilterBar(activeFilterContainerId());
      renderPage(currentPage);
    });
  });
}

function renderPage(page) {
  if (page === 'overview') renderOverview() 
  const filtered = applyFilters(processedRows, filters);
  
  // Tính toán KPI từ các cột bạn đã nêu
  const totalDemand = filtered.reduce((a, b) => a + (b.Demand || 0), 0);
  const avgStockout = mean(filtered.map(r => r.Stockout || 0)) * 100;

  // Cập nhật giao diện (Ví dụ)
  const el = document.getElementById('kpi-demand');
  if (el) el.innerText = fmtInt(totalDemand);
  
  // Vẽ biểu đồ bằng Chart.js
  makeChart('mainChart', {
    type: 'line',
    data: {
      labels: sampleArray(filtered.map(r => r.Date), 20),
      datasets: [{
        label: 'Demand',
        data: sampleArray(filtered.map(r => r.Demand), 20),
        borderColor: COLORS.accent
      }]
    }
  });
};
  else if (page === 'inventory') renderInventory();
  else if (page === 'drivers') renderDrivers();
  else if (page === 'forecast') renderForecast();
  else if (page === 'product') renderProductStore();
}

/* ---------------------------------------------------------------------- */
/* 6. Global filter bar (re-rendered into whichever page container is     */
/*    active; reads/writes the shared `filters` state)                    */
/* ---------------------------------------------------------------------- */
function opt(value, label, selected) { return `<option value="${value}" ${value === selected ? 'selected' : ''}>${label}</option>`; }

function renderFilterBar(containerId) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = `
    <div class="f-field"><label>From</label><input type="date" id="f-dateFrom" min="${dateBounds.min}" max="${dateBounds.max}" value="${filters.dateFrom || ''}"></div>
    <div class="f-field"><label>To</label><input type="date" id="f-dateTo" min="${dateBounds.min}" max="${dateBounds.max}" value="${filters.dateTo || ''}"></div>
    <div class="f-field"><label>Store</label><select id="f-store">${opt('all', 'All Stores', filters.store)}${dims.stores.map(s => opt(s, s, filters.store)).join('')}</select></div>
    <div class="f-field"><label>Product</label><select id="f-product">${opt('all', 'All Products', filters.product)}${dims.products.map(s => opt(s, s, filters.product)).join('')}</select></div>
    <div class="f-field"><label>Category</label><select id="f-category">${opt('all', 'All Categories', filters.category)}${dims.categories.map(s => opt(s, s, filters.category)).join('')}</select></div>
    <div class="f-field"><label>Region</label><select id="f-region">${opt('all', 'All Regions', filters.region)}${dims.regions.map(s => opt(s, s, filters.region)).join('')}</select></div>
    <div class="f-field"><label>Season</label><select id="f-season">${opt('all', 'All Seasons', filters.season)}${dims.seasons.map(s => opt(s, s, filters.season)).join('')}</select></div>
    <div class="f-field"><label>Promotion</label><select id="f-promo">${opt('all', 'All', filters.promo)}${opt('1', 'Promo Active', filters.promo)}${opt('0', 'No Promo', filters.promo)}</select></div>
    <div class="f-field"><label>Epidemic</label><select id="f-epidemic">${opt('all', 'All', filters.epidemic)}${opt('1', 'Epidemic', filters.epidemic)}${opt('0', 'Normal', filters.epidemic)}</select></div>
    <button class="f-reset" id="f-reset">Reset filters</button>
    <span class="filter-count" id="f-count"></span>
  `;
  ['dateFrom', 'dateTo', 'store', 'product', 'category', 'region', 'season', 'promo', 'epidemic'].forEach(key => {
    document.getElementById('f-' + key).addEventListener('change', (e) => {
      filters[key] = e.target.value || null;
      if (key !== 'dateFrom' && key !== 'dateTo' && !e.target.value) filters[key] = 'all';
      updateFilterCount();
      renderPage(currentPage);
    });
  });
  document.getElementById('f-reset').addEventListener('click', () => {
    Object.assign(filters, { dateFrom: null, dateTo: null, store: 'all', product: 'all', category: 'all', region: 'all', season: 'all', promo: 'all', epidemic: 'all' });
    renderFilterBar(containerId);
    renderPage(currentPage);
  });
  updateFilterCount();
}

function updateFilterCount() {
  const n = applyFilters(processedRows, filters).length;
  document.querySelectorAll('[id^="f-count"]').forEach(el => {
    el.textContent = `${fmtInt(n)} of ${fmtInt(processedRows.length)} rows match`;
  });
}

function buildSelectPanel() {
  const el = document.getElementById('selectPanel');
  el.innerHTML = `
    <div class="f-field"><label>Store</label><select id="p5-store">${opt('all', 'All Stores', page5Filters.store)}${dims.stores.map(s => opt(s, s, page5Filters.store)).join('')}</select></div>
    <div class="f-field"><label>Product</label><select id="p5-product">${opt('all', 'All Products', page5Filters.product)}${dims.products.map(s => opt(s, s, page5Filters.product)).join('')}</select></div>
    <div class="f-field"><label>Category</label><select id="p5-category">${opt('all', 'All Categories', page5Filters.category)}${dims.categories.map(s => opt(s, s, page5Filters.category)).join('')}</select></div>
    <div class="f-field"><label>Region</label><select id="p5-region">${opt('all', 'All Regions', page5Filters.region)}${dims.regions.map(s => opt(s, s, page5Filters.region)).join('')}</select></div>
    <button class="f-reset" id="p5-reset">Clear selection</button>
  `;
  ['store', 'product', 'category', 'region'].forEach(key => {
    document.getElementById('p5-' + key).addEventListener('change', (e) => {
      page5Filters[key] = e.target.value;
      renderProductStore();
    });
  });
  document.getElementById('p5-reset').addEventListener('click', () => {
    Object.assign(page5Filters, { store: 'all', product: 'all', category: 'all', region: 'all' });
    buildSelectPanel();
    renderProductStore();
  });
}

/* ---------------------------------------------------------------------- */
/* 7. PAGE 1 — Executive Overview                                          */
/* ---------------------------------------------------------------------- */
function renderOverview() {
  const rows = applyFilters(processedRows, filters);
  updateFilterCount();

  const totalDemand = rows.reduce((a, r) => a + r.Demand, 0);
  const nDates = new Set(rows.map(r => r.Date)).size || 1;
  const avgDaily = totalDemand / nDates;
  const avgInventory = mean(rows.map(r => r.Inventory));
  const stockoutRisk = mean(rows.map(r => r.Stockout)) * 100;

  const kpis = [
    { label: 'Total Demand', value: fmtInt(totalDemand), sub: `${fmtInt(rows.length)} observations`, cls: 'neutral' },
    { label: 'Avg Daily Demand', value: fmtNum(avgDaily, 1), sub: `across ${fmtInt(nDates)} days`, cls: 'neutral' },
    { label: 'Avg Inventory', value: fmtNum(avgInventory, 1), sub: 'units on hand', cls: 'neutral' },
    { label: 'Stockout Risk', value: fmtPct(stockoutRisk), sub: riskLabel(stockoutRisk), cls: riskClass(stockoutRisk) },
    { label: 'Forecast MAE', value: fmtNum(modelMetrics.mae, 1), sub: 'test-set model metric', cls: 'neutral', help: 'Average absolute forecasting error, in units of demand.' },
    { label: 'Forecast R²', value: fmtNum(modelMetrics.r2, 3), sub: modelMetrics.r2 > 0.5 ? 'Explains most variation' : 'Moderate fit', cls: modelMetrics.r2 > 0.5 ? 'good' : 'warn', help: 'Share of demand variation explained by the model (0–1).' },
  ];
  renderKPIRow('kpiRow', kpis);

  renderInsights();
  renderDemandTrendChart(rows, 'chartDemandTrend', trendGranularity);
  setupGranularityToggle(rows);
  renderCategoryDemandBar(rows, 'chartCategoryDemand');
  renderRegionDemandBar(rows, 'chartRegionDemand');
}

function riskLabel(pct) { if (pct < 8) return 'Healthy'; if (pct < 15) return 'Watch'; return 'High risk'; }
function riskClass(pct) { if (pct < 8) return 'good'; if (pct < 15) return 'warn'; return 'bad'; }

function renderKPIRow(containerId, kpis) {
  const el = document.getElementById(containerId);
  el.innerHTML = kpis.map(k => `
    <div class="kpi">
      <div class="kpi-label">${k.label}${k.help ? `<span class="help-dot" title="${k.help}">?</span>` : ''}</div>
      <div class="kpi-value">${k.value}</div>
      <div class="kpi-sub ${k.cls}">${k.sub}</div>
    </div>`).join('');
}

function renderInsights() {
  document.getElementById('insightsList').innerHTML =
    insights.map(t => `<li><span class="dot"></span><span>${t}</span></li>`).join('');
}

function aggregateTrend(rows, granularity) {
  const keyFn = granularity === 'weekly' ? (r => weekKey(r.Date)) : granularity === 'monthly' ? (r => monthKey(r.Date)) : (r => r.Date);
  const m = groupSum(rows, keyFn, 'Demand');
  const keys = [...m.keys()].sort();
  return { labels: keys, values: keys.map(k => m.get(k)) };
}

function renderDemandTrendChart(rows, canvasId, granularity) {
  const { labels, values } = aggregateTrend(rows, granularity);
  makeChart(canvasId, {
    type: 'line',
    data: { labels, datasets: [{ label: 'Total Demand', data: values, borderColor: COLORS.accent, backgroundColor: 'rgba(36,83,201,0.08)', fill: true, tension: 0.25, pointRadius: 0, borderWidth: 2 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } },
      scales: {
        x: { grid: { display: false }, ticks: { ...AXIS_TICK, maxTicksLimit: 10, maxRotation: 0 } },
        y: { grid: GRID, ticks: AXIS_TICK, beginAtZero: true },
      },
      interaction: { mode: 'index', intersect: false },
    },
  });
}

function setupGranularityToggle(rows) {
  const wrap = document.getElementById('trendGranularity');
  wrap.querySelectorAll('button').forEach(btn => {
    btn.onclick = () => {
      wrap.querySelectorAll('button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      trendGranularity = btn.dataset.g;
      renderDemandTrendChart(applyFilters(processedRows, filters), 'chartDemandTrend', trendGranularity);
    };
  });
}

function renderCategoryDemandBar(rows, canvasId) {
  const m = groupSum(rows, r => r.Category, 'Demand');
  const sorted = [...m.entries()].sort((a, b) => b[1] - a[1]);
  makeChart(canvasId, {
    type: 'bar',
    data: { labels: sorted.map(d => d[0]), datasets: [{ data: sorted.map(d => d[1]), backgroundColor: COLORS.accent, borderRadius: 4, maxBarThickness: 40 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => fmtInt(c.raw) } } },
      scales: { x: { grid: { display: false }, ticks: AXIS_TICK }, y: { grid: GRID, ticks: AXIS_TICK, beginAtZero: true } },
    },
  });
}

function renderRegionDemandBar(rows, canvasId) {
  const m = groupSum(rows, r => r.Region, 'Demand');
  const sorted = [...m.entries()].sort((a, b) => b[1] - a[1]);
  makeChart(canvasId, {
    type: 'bar',
    data: { labels: sorted.map(d => d[0]), datasets: [{ data: sorted.map(d => d[1]), backgroundColor: COLORS.chart[1], borderRadius: 4, maxBarThickness: 40 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => fmtInt(c.raw) } } },
      scales: { x: { grid: { display: false }, ticks: AXIS_TICK }, y: { grid: GRID, ticks: AXIS_TICK, beginAtZero: true } },
    },
  });
}

/* ---------------------------------------------------------------------- */
/* 8. PAGE 2 — Demand & Inventory                                          */
/* ---------------------------------------------------------------------- */
let invTableState = { search: '', sortKey: 'StockoutRisk', sortDir: 'desc', page: 1, pageSize: 8 };

function renderInventory() {
  const rows = applyFilters(processedRows, filters);
  updateFilterCount();

  const avgInventory = mean(rows.map(r => r.Inventory));
  const stockoutRisk = mean(rows.map(r => r.Stockout)) * 100;
  const avgCoverage = mean(rows.map(r => r.Coverage));
  const bySkuRisk = groupAgg(rows, r => r.Product);
  let highRisk = 0;
  bySkuRisk.forEach(arr => { if (mean(arr.map(r => r.Stockout)) * 100 >= 15) highRisk++; });

  renderKPIRow('kpiRowInv', [
    { label: 'Avg Inventory', value: fmtNum(avgInventory, 1), sub: 'units on hand', cls: 'neutral' },
    { label: 'Stockout Risk', value: fmtPct(stockoutRisk), sub: riskLabel(stockoutRisk), cls: riskClass(stockoutRisk) },
    { label: 'Avg Coverage', value: fmtNum(avgCoverage, 2) + '×', sub: avgCoverage < 1 ? 'Below demand' : 'Above demand', cls: avgCoverage < 1 ? 'bad' : 'good' },
    { label: 'High-Risk Products', value: highRisk, sub: 'of ' + bySkuRisk.size + ' SKUs in view', cls: highRisk > 0 ? 'warn' : 'good' },
  ]);

  renderInvScatter(rows);
  renderInvGapByCategory(rows);
  renderStockoutByCategory(rows);
  renderCoverageByCategory(rows);
  renderInvTable(rows);
}

function renderInvScatter(rows) {
  const sample = sampleArray(rows, 1500);
  const ok = sample.filter(r => !r.Stockout).map(r => ({ x: r.Inventory, y: r.Demand }));
  const risk = sample.filter(r => r.Stockout).map(r => ({ x: r.Inventory, y: r.Demand }));
  makeChart('chartInvVsDemand', {
    type: 'scatter',
    data: {
      datasets: [
        { label: 'Inventory ≥ demand', data: ok, backgroundColor: 'rgba(27,158,99,0.45)', pointRadius: 2.5 },
        { label: 'Stockout risk', data: risk, backgroundColor: 'rgba(213,72,75,0.55)', pointRadius: 2.5 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Inventory Level', font: { size: 10.5 } }, grid: GRID, ticks: AXIS_TICK },
        y: { title: { display: true, text: 'Demand', font: { size: 10.5 } }, grid: GRID, ticks: AXIS_TICK },
      },
    },
  });
}

function renderInvGapByCategory(rows) {
  const m = groupAgg(rows, r => r.Category);
  const entries = [...m.entries()].map(([k, arr]) => [k, mean(arr.map(r => r.InvGap))]).sort((a, b) => b[1] - a[1]);
  makeChart('chartInvGap', {
    type: 'bar',
    data: { labels: entries.map(e => e[0]), datasets: [{ data: entries.map(e => e[1]), backgroundColor: entries.map(e => e[1] >= 0 ? COLORS.good : COLORS.bad), borderRadius: 4, maxBarThickness: 40 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => fmtNum(c.raw, 1) + ' units' } } },
      scales: { x: { grid: { display: false }, ticks: AXIS_TICK }, y: { grid: GRID, ticks: AXIS_TICK } },
    },
  });
}

function renderStockoutByCategory(rows) {
  const m = groupAgg(rows, r => r.Category);
  const entries = [...m.entries()].map(([k, arr]) => [k, mean(arr.map(r => r.Stockout)) * 100]).sort((a, b) => b[1] - a[1]);
  makeChart('chartStockoutCat', {
    type: 'bar',
    data: { labels: entries.map(e => e[0]), datasets: [{ data: entries.map(e => e[1]), backgroundColor: entries.map(e => e[1] >= 15 ? COLORS.bad : e[1] >= 8 ? COLORS.warn : COLORS.good), borderRadius: 4, maxBarThickness: 40 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => fmtNum(c.raw, 1) + '%' } } },
      scales: { x: { grid: { display: false }, ticks: AXIS_TICK }, y: { grid: GRID, ticks: AXIS_TICK, beginAtZero: true } },
    },
  });
}

function renderCoverageByCategory(rows) {
  const m = groupAgg(rows, r => r.Category);
  const entries = [...m.entries()].map(([k, arr]) => [k, mean(arr.map(r => r.Coverage))]).sort((a, b) => a[1] - b[1]);
  makeChart('chartCoverageCat', {
    type: 'bar',
    data: { labels: entries.map(e => e[0]), datasets: [{ data: entries.map(e => e[1]), backgroundColor: entries.map(e => e[1] < 1 ? COLORS.bad : COLORS.accent), borderRadius: 4, maxBarThickness: 40 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => fmtNum(c.raw, 2) + '×' } } },
      scales: { x: { grid: { display: false }, ticks: AXIS_TICK }, y: { grid: GRID, ticks: AXIS_TICK, beginAtZero: true } },
    },
  });
}

function buildProductSummary(rows) {
  const m = groupAgg(rows, r => r.Product);
  const out = [];
  m.forEach((arr, product) => {
    out.push({
      Product: product,
      Category: arr[0].Category,
      AvgDemand: mean(arr.map(r => r.Demand)),
      TotalDemand: arr.reduce((a, r) => a + r.Demand, 0),
      AvgInventory: mean(arr.map(r => r.Inventory)),
      StockoutRisk: mean(arr.map(r => r.Stockout)) * 100,
      Coverage: mean(arr.map(r => r.Coverage)),
      AvgPrice: mean(arr.map(r => r.Price)),
      Promo: mean(arr.map(r => r.Promo)) * 100,
    });
  });
  return out;
}

function renderInvTable(rows) {
  const data = buildProductSummary(rows);
  const columns = [
    { key: 'Product', label: 'Product', fmt: v => v },
    { key: 'Category', label: 'Category', fmt: v => v },
    { key: 'AvgDemand', label: 'Avg Demand', fmt: v => fmtNum(v, 1) },
    { key: 'TotalDemand', label: 'Total Demand', fmt: v => fmtInt(v) },
    { key: 'AvgInventory', label: 'Avg Inventory', fmt: v => fmtNum(v, 1) },
    { key: 'StockoutRisk', label: 'Stockout Risk', fmt: v => fmtPct(v) },
    { key: 'Coverage', label: 'Inventory Coverage', fmt: v => fmtNum(v, 2) + '×' },
    { key: 'AvgPrice', label: 'Avg Price', fmt: v => '$' + fmtNum(v, 2) },
  ];
  renderDataTable({
    data, columns, state: invTableState,
    tableId: 'invTable', paginationId: 'invTablePagination', searchId: 'invTableSearch', metaId: 'invTableMeta',
    searchFields: ['Product', 'Category'],
    rowClass: r => r.StockoutRisk >= 15 ? 'risk-high' : r.StockoutRisk >= 8 ? 'risk-med' : '',
    onChange: () => renderInvTable(rows),
  });
}

/* ---------------------------------------------------------------------- */
/* 9. PAGE 3 — Demand Drivers                                              */
/* ---------------------------------------------------------------------- */
function renderDrivers() {
  const rows = applyFilters(processedRows, filters);
  updateFilterCount();

  renderPriceDemandScatter(rows);
  renderPromoEpiCompare(rows);
  renderCorrelationChart();
  renderFeatureImportanceChart('chartFeatureImportance', 15);
}

function renderPriceDemandScatter(rows) {
  const sample = sampleArray(rows, 1500).map(r => ({ x: r.Price, y: r.Demand }));
  makeChart('chartPriceDemand', {
    type: 'scatter',
    data: { datasets: [{ label: 'Observations', data: sample, backgroundColor: 'rgba(36,83,201,0.35)', pointRadius: 2.5 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Price ($)', font: { size: 10.5 } }, grid: GRID, ticks: AXIS_TICK },
        y: { title: { display: true, text: 'Demand', font: { size: 10.5 } }, grid: GRID, ticks: AXIS_TICK },
      },
    },
  });
}

function renderPromoEpiCompare(rows) {
  const promoYes = mean(rows.filter(r => r.Promo === 1).map(r => r.Demand));
  const promoNo = mean(rows.filter(r => r.Promo === 0).map(r => r.Demand));
  const epiYes = mean(rows.filter(r => r.Epidemic === 1).map(r => r.Demand));
  const epiNo = mean(rows.filter(r => r.Epidemic === 0).map(r => r.Demand));
  document.querySelector('#promoNo .val').textContent = fmtNum(promoNo, 1);
  document.querySelector('#promoYes .val').textContent = fmtNum(promoYes, 1);
  document.querySelector('#epiNo .val').textContent = fmtNum(epiNo, 1);
  document.querySelector('#epiYes .val').textContent = fmtNum(epiYes, 1);
}

function renderCorrelationChart() {
  const sorted = [...correlationData].sort((a, b) => a.correlation - b.correlation);
  makeChart('chartCorrelation', {
    type: 'bar',
    data: {
      labels: sorted.map(d => d.feature),
      datasets: [{ data: sorted.map(d => d.correlation), backgroundColor: sorted.map(d => d.correlation >= 0 ? COLORS.accent : COLORS.bad), borderRadius: 4 }],
    },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => 'r = ' + fmtNum(c.raw, 3) } } },
      scales: { x: { min: -1, max: 1, grid: GRID, ticks: AXIS_TICK }, y: { grid: { display: false }, ticks: AXIS_TICK } },
    },
  });
}

function renderFeatureImportanceChart(canvasId, topN) {
  const sorted = [...featureImportance].slice(0, topN).sort((a, b) => a.Importance - b.Importance);
  makeChart(canvasId, {
    type: 'bar',
    data: { labels: sorted.map(d => d.Label), datasets: [{ data: sorted.map(d => d.Importance), backgroundColor: COLORS.chart[2], borderRadius: 4 }] },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => fmtNum(c.raw * 100, 1) + '% of model importance' } } },
      scales: { x: { grid: GRID, ticks: AXIS_TICK }, y: { grid: { display: false }, ticks: AXIS_TICK } },
    },
  });
}

/* ---------------------------------------------------------------------- */
/* 10. PAGE 4 — Demand Forecasting                                         */
/* ---------------------------------------------------------------------- */
let brush = { start: 0, end: 100 };

function computeMetrics(rows) {
  if (!rows.length) return { mae: 0, rmse: 0, r2: 0, n: 0 };
  const errs = rows.map(r => r.Actual - r.Predicted);
  const mae = mean(errs.map(Math.abs));
  const rmse = Math.sqrt(mean(errs.map(e => e * e)));
  const yMean = mean(rows.map(r => r.Actual));
  const ssTot = rows.reduce((a, r) => a + (r.Actual - yMean) ** 2, 0);
  const ssRes = rows.reduce((a, r) => a + (r.Actual - r.Predicted) ** 2, 0);
  const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
  return { mae, rmse, r2, n: rows.length };
}

function renderForecast() {
  const rows = applyFilters(predictionRows, filters);
  updateFilterCount();
  const m = computeMetrics(rows);

  renderKPIRow('kpiRowModel', [
    { label: 'MAE', value: fmtNum(m.mae, 1), sub: 'avg absolute error', cls: 'neutral', help: 'Average absolute forecasting error.' },
    { label: 'RMSE', value: fmtNum(m.rmse, 1), sub: 'penalizes large misses', cls: 'neutral', help: 'Root-mean-square error — penalizes larger forecasting errors more strongly.' },
    { label: 'R²', value: fmtNum(m.r2, 3), sub: `on ${fmtInt(m.n)} test obs.`, cls: m.r2 > 0.5 ? 'good' : 'warn', help: 'Measures how much variation in demand is explained by the model.' },
  ]);

  renderActualPredictedChart(rows);
  renderScatterAP(rows);
  renderForecastErrorChart(rows);
  renderFeatureImportanceChart('chartFeatureImportance2', 10);
}

function dailyActualPredicted(rows) {
  const m = new Map();
  for (const r of rows) {
    if (!m.has(r.Date)) m.set(r.Date, { a: 0, p: 0 });
    const o = m.get(r.Date); o.a += r.Actual; o.p += r.Predicted;
  }
  const labels = [...m.keys()].sort();
  return { labels, actual: labels.map(k => m.get(k).a), predicted: labels.map(k => m.get(k).p) };
}

function renderActualPredictedChart(rows) {
  const full = dailyActualPredicted(rows);
  const n = full.labels.length;
  const si = Math.floor((brush.start / 100) * n);
  const ei = Math.max(si + 1, Math.floor((brush.end / 100) * n));
  const labels = full.labels.slice(si, ei);
  const actual = full.actual.slice(si, ei);
  const predicted = full.predicted.slice(si, ei);

  document.getElementById('brushStartLabel').textContent = labels[0] || '';
  document.getElementById('brushEndLabel').textContent = labels[labels.length - 1] || '';

  makeChart('chartActualPredicted', {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Actual Demand', data: actual, borderColor: COLORS.accent, backgroundColor: 'transparent', tension: 0.2, pointRadius: 0, borderWidth: 2 },
        { label: 'Predicted Demand', data: predicted, borderColor: COLORS.chart[3], backgroundColor: 'transparent', borderDash: [4, 3], tension: 0.2, pointRadius: 0, borderWidth: 2 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { position: 'top', align: 'end' }, tooltip: { mode: 'index', intersect: false } },
      scales: {
        x: { grid: { display: false }, ticks: { ...AXIS_TICK, maxTicksLimit: 10, maxRotation: 0 } },
        y: { grid: GRID, ticks: AXIS_TICK, beginAtZero: true },
      },
    },
  });

  const startSlider = document.getElementById('brushStart');
  const endSlider = document.getElementById('brushEnd');
  startSlider.value = brush.start; endSlider.value = brush.end;
  startSlider.oninput = () => {
    brush.start = Math.min(Number(startSlider.value), Number(endSlider.value) - 1);
    startSlider.value = brush.start;
    renderActualPredictedChart(applyFilters(predictionRows, filters));
  };
  endSlider.oninput = () => {
    brush.end = Math.max(Number(endSlider.value), Number(startSlider.value) + 1);
    endSlider.value = brush.end;
    renderActualPredictedChart(applyFilters(predictionRows, filters));
  };
  document.getElementById('resetZoomBtn').onclick = () => {
    brush = { start: 0, end: 100 };
    renderActualPredictedChart(applyFilters(predictionRows, filters));
  };
}

function renderScatterAP(rows) {
  const sample = sampleArray(rows, 1500);
  const pts = sample.map(r => ({ x: r.Actual, y: r.Predicted }));
  const allVals = sample.flatMap(r => [r.Actual, r.Predicted]);
  const lo = Math.min(...allVals, 0), hi = Math.max(...allVals, 1);
  makeChart('chartScatterAP', {
    type: 'scatter',
    data: {
      datasets: [
        { label: 'Predictions', data: pts, backgroundColor: 'rgba(36,83,201,0.4)', pointRadius: 2.5 },
        { label: 'Perfect prediction', type: 'line', data: [{ x: lo, y: lo }, { x: hi, y: hi }], borderColor: COLORS.bad, borderDash: [5, 4], pointRadius: 0, borderWidth: 1.5 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Actual Demand', font: { size: 10.5 } }, grid: GRID, ticks: AXIS_TICK },
        y: { title: { display: true, text: 'Predicted Demand', font: { size: 10.5 } }, grid: GRID, ticks: AXIS_TICK },
      },
    },
  });
}

function renderForecastErrorChart(rows) {
  const m = new Map();
  for (const r of rows) m.set(r.Date, (m.get(r.Date) || 0) + (r.Actual - r.Predicted));
  const labels = [...m.keys()].sort();
  const values = labels.map(k => m.get(k));
  makeChart('chartForecastError', {
    type: 'bar',
    data: { labels, datasets: [{ data: values, backgroundColor: values.map(v => v >= 0 ? 'rgba(36,83,201,0.55)' : 'rgba(213,72,75,0.55)'), maxBarThickness: 6 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => fmtNum(c.raw, 1) } } },
      scales: {
        x: { grid: { display: false }, ticks: { ...AXIS_TICK, maxTicksLimit: 8, maxRotation: 0 } },
        y: { grid: GRID, ticks: AXIS_TICK },
      },
    },
  });
}

/* ---------------------------------------------------------------------- */
/* 11. PAGE 5 — Product & Store Analysis                                   */
/* ---------------------------------------------------------------------- */
let perfTableState = { search: '', sortKey: 'TotalDemand', sortDir: 'desc', page: 1, pageSize: 8 };

function renderProductStore() {
  const hasSelection = Object.values(page5Filters).some(v => v !== 'all');
  document.getElementById('page5Empty').style.display = hasSelection ? 'none' : 'block';
  document.getElementById('page5Content').style.display = hasSelection ? 'block' : 'none';
  if (!hasSelection) return;

  const rows = applyPage5Filters(processedRows, page5Filters);
  if (!rows.length) {
    document.getElementById('page5Content').style.display = 'none';
    document.getElementById('page5Empty').style.display = 'block';
    document.getElementById('page5Empty').innerHTML = 'No observations match this selection.';
    return;
  }

  const totalDemand = rows.reduce((a, r) => a + r.Demand, 0);
  const avgDemand = mean(rows.map(r => r.Demand));
  const avgInventory = mean(rows.map(r => r.Inventory));
  const stockoutRisk = mean(rows.map(r => r.Stockout)) * 100;
  const avgPrice = mean(rows.map(r => r.Price));
  const avgCoverage = mean(rows.map(r => r.Coverage));

  renderKPIRow('kpiRow5', [
    { label: 'Total Demand', value: fmtInt(totalDemand), sub: `${fmtInt(rows.length)} obs.`, cls: 'neutral' },
    { label: 'Avg Demand', value: fmtNum(avgDemand, 1), sub: 'per observation', cls: 'neutral' },
    { label: 'Avg Inventory', value: fmtNum(avgInventory, 1), sub: 'units on hand', cls: 'neutral' },
    { label: 'Stockout Risk', value: fmtPct(stockoutRisk), sub: riskLabel(stockoutRisk), cls: riskClass(stockoutRisk) },
    { label: 'Avg Price', value: '$' + fmtNum(avgPrice, 2), sub: 'per unit', cls: 'neutral' },
    { label: 'Inventory Coverage', value: fmtNum(avgCoverage, 2) + '×', sub: avgCoverage < 1 ? 'Below demand' : 'Above demand', cls: avgCoverage < 1 ? 'bad' : 'good' },
  ]);

  const label5 = [page5Filters.product !== 'all' ? page5Filters.product : null, page5Filters.store !== 'all' ? page5Filters.store : null, page5Filters.category !== 'all' ? page5Filters.category : null, page5Filters.region !== 'all' ? page5Filters.region : null].filter(Boolean).join(' · ') || 'Selection';
  document.getElementById('trendTitle5').textContent = `Demand vs Inventory Trend — ${label5}`;
  document.getElementById('perfSub5').textContent = `Products within: ${label5}`;

  renderTrend5(rows);
  renderPerfTable5(rows);
}

function renderTrend5(rows) {
  const demandM = groupSum(rows, r => r.Date, 'Demand');
  const invByDate = groupAgg(rows, r => r.Date);
  const labels = [...demandM.keys()].sort();
  const demand = labels.map(k => demandM.get(k));
  const inventory = labels.map(k => mean(invByDate.get(k).map(r => r.Inventory)));
  makeChart('chartTrend5', {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Demand', data: demand, borderColor: COLORS.accent, backgroundColor: 'rgba(36,83,201,0.08)', fill: true, tension: 0.2, pointRadius: 0, borderWidth: 2, yAxisID: 'y' },
        { label: 'Inventory', data: inventory, borderColor: COLORS.chart[1], backgroundColor: 'transparent', borderDash: [4, 3], tension: 0.2, pointRadius: 0, borderWidth: 2, yAxisID: 'y' },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { position: 'top', align: 'end' }, tooltip: { mode: 'index', intersect: false } },
      scales: {
        x: { grid: { display: false }, ticks: { ...AXIS_TICK, maxTicksLimit: 10, maxRotation: 0 } },
        y: { grid: GRID, ticks: AXIS_TICK, beginAtZero: true },
      },
    },
  });
}

function renderPerfTable5(rows) {
  const data = buildProductSummary(rows);
  const columns = [
    { key: 'Product', label: 'Product', fmt: v => v },
    { key: 'Category', label: 'Category', fmt: v => v },
    { key: 'TotalDemand', label: 'Demand', fmt: v => fmtInt(v) },
    { key: 'AvgInventory', label: 'Inventory', fmt: v => fmtNum(v, 1) },
    { key: 'StockoutRisk', label: 'Stockout Risk', fmt: v => fmtPct(v) },
    { key: 'Coverage', label: 'Coverage', fmt: v => fmtNum(v, 2) + '×' },
    { key: 'AvgPrice', label: 'Price', fmt: v => '$' + fmtNum(v, 2) },
    { key: 'Promo', label: 'Promo Freq.', fmt: v => fmtPct(v, 0) },
  ];
  renderDataTable({
    data, columns, state: perfTableState,
    tableId: 'perfTable', paginationId: 'perfTablePagination', searchId: 'perfTableSearch', metaId: 'perfTableMeta',
    searchFields: ['Product', 'Category'],
    rowClass: r => r.StockoutRisk >= 15 ? 'risk-high' : r.StockoutRisk >= 8 ? 'risk-med' : '',
    onChange: () => renderPerfTable5(rows),
  });
}

/* ---------------------------------------------------------------------- */
/* 12. Generic sortable / searchable / paginated table component           */
/* ---------------------------------------------------------------------- */
function renderDataTable({ data, columns, state, tableId, paginationId, searchId, metaId, searchFields, rowClass, onChange }) {
  const searchInput = document.getElementById(searchId);
  if (searchInput && searchInput.value !== state.search && document.activeElement !== searchInput) searchInput.value = state.search;
  if (searchInput && !searchInput._bound) {
    searchInput._bound = true;
    searchInput.addEventListener('input', (e) => { state.search = e.target.value; state.page = 1; onChange(); });
  }

  let filtered = data;
  if (state.search) {
    const q = state.search.toLowerCase();
    filtered = data.filter(r => searchFields.some(f => String(r[f]).toLowerCase().includes(q)));
  }
  filtered = [...filtered].sort((a, b) => {
    const va = a[state.sortKey], vb = b[state.sortKey];
    const cmp = typeof va === 'number' ? va - vb : String(va).localeCompare(String(vb));
    return state.sortDir === 'asc' ? cmp : -cmp;
  });

  const totalPages = Math.max(1, Math.ceil(filtered.length / state.pageSize));
  state.page = Math.min(state.page, totalPages);
  const start = (state.page - 1) * state.pageSize;
  const pageRows = filtered.slice(start, start + state.pageSize);

  const table = document.getElementById(tableId);
  table.innerHTML = `
    <thead><tr>${columns.map(c => `<th data-key="${c.key}">${c.label}${state.sortKey === c.key ? `<span class="arrow">${state.sortDir === 'asc' ? '▲' : '▼'}</span>` : ''}</th>`).join('')}</tr></thead>
    <tbody>${pageRows.map(r => `<tr class="${rowClass ? rowClass(r) : ''}">${columns.map(c => `<td>${c.fmt(r[c.key])}</td>`).join('')}</tr>`).join('') || `<tr><td colspan="${columns.length}" style="text-align:center;color:var(--text-muted);padding:24px;">No matching rows</td></tr>`}</tbody>
  `;
  table.querySelectorAll('th').forEach(th => {
    th.onclick = () => {
      const key = th.dataset.key;
      if (state.sortKey === key) state.sortDir = state.sortDir === 'asc' ? 'desc' : 'asc';
      else { state.sortKey = key; state.sortDir = 'desc'; }
      onChange();
    };
  });

  const meta = document.getElementById(metaId);
  if (meta) meta.textContent = `${fmtInt(filtered.length)} rows`;

  const pag = document.getElementById(paginationId);
  pag.innerHTML = `
    <button id="${paginationId}-prev" ${state.page <= 1 ? 'disabled' : ''}>Prev</button>
    <span class="pg-info">Page ${state.page} of ${totalPages}</span>
    <button id="${paginationId}-next" ${state.page >= totalPages ? 'disabled' : ''}>Next</button>
  `;
  const prevBtn = document.getElementById(`${paginationId}-prev`);
  const nextBtn = document.getElementById(`${paginationId}-next`);
  if (prevBtn) prevBtn.onclick = () => { state.page--; onChange(); };
  if (nextBtn) nextBtn.onclick = () => { state.page++; onChange(); };
}

/* ---------------------------------------------------------------------- */
/* 13. Boot                                                                 */
/* ---------------------------------------------------------------------- */
init();
function renderFilterBar(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  container.innerHTML = `
    <div class="filter-controls" style="display:flex; gap:10px; flex-wrap:wrap; padding:10px;">
      <select id="f-store"><option value="all">All Stores</option>${dims.stores.map(s => `<option value="${s}">${s}</option>`).join('')}</select>
      <select id="f-product"><option value="all">All Products</option>${dims.products.map(p => `<option value="${p}">${p}</option>`).join('')}</select>
      <input type="date" id="f-from" value="${filters.dateFrom}">
      <input type="date" id="f-to" value="${filters.dateTo}">
    </div>
  `;

  // Lắng nghe sự kiện thay đổi để cập nhật biểu đồ ngay lập tức
  container.querySelectorAll('select, input').forEach(el => {
    el.addEventListener('change', (e) => {
      if (e.target.id === 'f-store') filters.store = e.target.value;
      if (e.target.id === 'f-product') filters.product = e.target.value;
      if (e.target.id === 'f-from') filters.dateFrom = e.target.value;
      if (e.target.id === 'f-to') filters.dateTo = e.target.value;
      renderPage(currentPage); // Vẽ lại trang hiện tại
    });
  });
}

function buildSelectPanel() { 
    console.log("Quick select panel initialized."); 
}

// Hàm render Overview (nếu bạn đã sửa rồi thì bỏ qua, nếu chưa thì dùng bản tạm này)
function renderOverview() {
  const data = applyFilters(processedRows, filters);
  // Ví dụ cập nhật 1 số tổng quát
  const totalUnits = data.reduce((sum, r) => sum + (r.UnitsSold || 0), 0);
  console.log("Current filtered UnitsSold:", totalUnits);
}

// Các hàm trang khác (để trống để không báo lỗi)
function renderInventory() { console.log("Inventory View"); }
function renderDrivers() { console.log("Drivers View"); }
function renderForecast() { console.log("Forecast View"); }
function renderProductStore() { console.log("Product Detail View"); }

// Lệnh cuối cùng để khởi chạy toàn bộ
init();
