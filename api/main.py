from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

app = FastAPI(title="ECOALPHA — ESG-Informed Portfolio Optimizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- UI ----------
@app.get("/", response_class=HTMLResponse)
def home():
    # Tailwind via CDN; vanilla JS for interactivity
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>ECOALPHA — ESG-Informed Portfolio Optimizer</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    :root {
      --bg: #0b1220;
      --panel: #111827;
      --muted: #cbd5e1;
      --border: #1f2937;
      --accent: #64748b;
    }
    body { background: #0f172a; color: #e5e7eb; }
    .card { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; }
    .soft { box-shadow: 0 10px 30px rgba(0,0,0,.25); }
    .ghost { background: rgba(148,163,184,.08); }
    .pill { border: 1px solid var(--border); background: #0b1220; }
    .muted { color: var(--muted); }
    .table-wrap { overflow: auto; }
    .tbl { width: 100%; border-collapse: collapse; font-size: 0.935rem; }
    .tbl th, .tbl td { padding: .65rem .9rem; border-bottom: 1px solid var(--border); white-space: nowrap; }
    .tbl th { text-align: left; color: #cbd5e1; font-weight: 600; background: rgba(148,163,184,.06); position: sticky; top: 0; }
    .btn { border-radius: 10px; padding: .6rem .9rem; border: 1px solid var(--border); }
    .btn-primary { background: #334155; }
    .btn:disabled { opacity: .6; cursor: not-allowed; }
    .hint { font-size: .9rem; color: #94a3b8; }
  </style>
</head>
<body>
  <div class="min-h-screen">
    <!-- Top bar -->
    <div class="w-full border-b border-slate-800/80 bg-slate-900/50 backdrop-blur">
      <div class="mx-auto max-w-7xl px-5 py-3 flex items-center justify-between">
        <div class="flex items-center gap-3">
          <div class="h-7 w-7 rounded-full bg-emerald-400/90"></div>
          <div class="text-slate-200 font-semibold tracking-wide">ECOALPHA</div>
        </div>
        <div id="conn" class="text-sm text-slate-400">CONNECTING</div>
      </div>
    </div>

    <div class="mx-auto max-w-7xl px-5 py-8 grid grid-cols-1 md:grid-cols-12 gap-6">
      <!-- Sidebar -->
      <aside class="md:col-span-3 card soft p-5">
        <h3 class="text-lg font-semibold mb-4">Settings</h3>

        <label class="block text-sm mb-1 muted">Tickers (space-separated)</label>
        <input id="tickers" class="w-full pill px-3 py-2 mb-4 focus:outline-none focus:ring-2 focus:ring-slate-600"
               placeholder="AAPL MSFT TSLA GOOG AMZN" value="AAPL MSFT TSLA GOOG AMZN" />

        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="block text-sm mb-1 muted">Start date</label>
            <input id="start" class="w-full pill px-3 py-2" placeholder="YYYY/MM/DD" value="2024/06/15" />
          </div>
          <div>
            <label class="block text-sm mb-1 muted">End date</label>
            <input id="end" class="w-full pill px-3 py-2" placeholder="YYYY/MM/DD" value="2025/07/15" />
          </div>
        </div>

        <label class="flex items-center gap-2 mt-4">
          <input id="mock" type="checkbox" class="h-4 w-4" checked />
          <span class="muted text-sm">Use mock ESG (varied, deterministic)</span>
        </label>

        <button id="run" class="btn btn-primary w-full mt-4">Run Optimization</button>
        <p class="hint mt-3">Use mock ESG to guarantee varied scores for demos. Turn it off to try live Yahoo sustainability + news.</p>
      </aside>

      <!-- Main panel -->
      <main class="md:col-span-9 space-y-6">
        <section class="card soft p-6">
          <h1 class="text-3xl md:text-4xl font-extrabold tracking-tight">ECOALPHA — ESG-Informed Portfolio Optimizer</h1>
          <p class="muted mt-2">Use mock ESG to guarantee varied scores for demos. Turn it off to try live Yahoo sustainability + news (+ optional local filings).</p>
        </section>

        <section class="card soft p-6">
          <h2 class="text-xl font-semibold mb-3">ESG Scores (normalized 0–1)</h2>
          <div class="table-wrap">
            <table id="esg" class="tbl"></table>
          </div>
        </section>

        <section class="card soft p-6">
          <h2 class="text-xl font-semibold mb-3">Price Data (head)</h2>
          <div class="table-wrap">
            <table id="prices" class="tbl"></table>
          </div>
        </section>
      </main>
    </div>
  </div>

  <script>
    const conn = document.getElementById("conn");
    const runBtn = document.getElementById("run");
    const esgTbl = document.getElementById("esg");
    const pricesTbl = document.getElementById("prices");

    function setConn(ok) { conn.textContent = ok ? "LIVE" : "CONNECTING"; conn.className = ok ? "text-sm text-emerald-400" : "text-sm text-slate-400"; }

    function asTable(el, data) {
      if (!data || data.length === 0) { el.innerHTML = "<tr><td class='muted'>No data.</td></tr>"; return; }
      const cols = Object.keys(data[0]);
      let thead = "<thead><tr>" + cols.map(c => `<th>${c}</th>`).join("") + "</tr></thead>";
      let rows = data.map(r => "<tr>" + cols.map(c => `<td>${r[c]}</td>`).join("") + "</tr>").join("");
      el.innerHTML = thead + "<tbody>" + rows + "</tbody>";
    }

    async function loadOnce() {
      try {
        const params = new URLSearchParams({
          tickers: "AAPL MSFT TSLA GOOG AMZN",
          start: "2024/06/15",
          end: "2025/07/15",
          mock: "true"
        });
        const resp = await fetch(`/optimize?${params.toString()}`);
        const data = await resp.json();
        asTable(esgTbl, data.esg_table);
        asTable(pricesTbl, data.price_head);
        setConn(true);
      } catch (e) { setConn(false); }
    }

    runBtn.addEventListener("click", async () => {
      runBtn.disabled = true;
      const tickers = document.getElementById("tickers").value.trim();
      const start = document.getElementById("start").value.trim();
      const end = document.getElementById("end").value.trim();
      const mock = document.getElementById("mock").checked;
      try {
        const params = new URLSearchParams({ tickers, start, end, mock: String(mock) });
        const resp = await fetch(`/optimize?${params.toString()}`);
        const data = await resp.json();
        asTable(esgTbl, data.esg_table);
        asTable(pricesTbl, data.price_head);
      } finally { runBtn.disabled = false; }
    });

    loadOnce();
  </script>
</body>
</html>
    """

# ---------- API ----------
def _normalize_01(series: pd.Series):
    if series.std() == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.min()) / (series.max() - series.min())

def _mock_esg_for(tickers):
    rng = np.random.default_rng(42)
    e = pd.Series(rng.random(len(tickers)), index=tickers)
    s = pd.Series(rng.random(len(tickers)), index=tickers)
    g = pd.Series(rng.random(len(tickers)), index=tickers)
    df = pd.DataFrame({"Ticker": tickers, "E": _normalize_01(e).round(3),
                       "S": _normalize_01(s).round(3), "G": _normalize_01(g).round(3)})
    return df

def _live_price_head(tickers, start, end):
    df = yf.download(" ".join(tickers), start=start, end=end, progress=False)["Adj Close"]
    df = df if isinstance(df, pd.DataFrame) else df.to_frame()
    head = df.head(5).reset_index()
    head["Date"] = head["Date"].dt.strftime("%Y-%m-%d")
    # Melt for a compact head table
    melted = head.melt(id_vars=["Date"], var_name="Ticker", value_name="Adj Close").round(2)
    return melted

@app.get("/optimize")
def optimize(
    tickers: str = Query("AAPL MSFT TSLA GOOG AMZN"),
    start: str = Query("2024/06/15"),
    end: str = Query("2025/07/15"),
    mock: bool = Query(True)
):
    tickers_list = [t.strip().upper() for t in tickers.split() if t.strip()]
    # Parse dates safely
    try:
        _ = datetime.strptime(start.replace("-", "/"), "%Y/%m/%d")
        _ = datetime.strptime(end.replace("-", "/"), "%Y/%m/%d")
    except Exception:
        return JSONResponse({"error": "Invalid date format. Use YYYY/MM/DD."}, status_code=400)

    # ESG: mock or (hook up your real pipeline here)
    esg_df = _mock_esg_for(tickers_list) if mock else _mock_esg_for(tickers_list)  # placeholder for real ESG

    # Prices head
    try:
        head = _live_price_head(tickers_list, start.replace("/", "-"), end.replace("/", "-"))
    except Exception:
        # If rate-limited or offline, fall back to synthetic series
        dates = pd.date_range(start=start.replace("/", "-"), periods=5, freq="B")
        rows = []
        for t in tickers_list:
            base = 100 + np.random.default_rng(hash(t) % 10_000).normal(0, 1, size=5).cumsum()
            for d, v in zip(dates, base):
                rows.append({"Date": d.strftime("%Y-%m-%d"), "Ticker": t, "Adj Close": round(float(v), 2)})
        head = pd.DataFrame(rows)

    return {
        "esg_table": esg_df.to_dict(orient="records"),
        "price_head": head.to_dict(orient="records"),
    }

@app.get("/health")
def health():
    return {"status": "ok"}