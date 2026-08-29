# Retail Demand & Inventory Analytics Dashboard

An interactive, Power BI–style HTML dashboard for a retail demand-forecasting
project: Executive Overview → Demand & Inventory → Demand Drivers → Demand
Forecasting → Product & Store Analysis.

## Project structure

```
index.html            # the dashboard (open this)
assets/
  style.css            # design system (tokens, layout, components)
  app.js                # data loading, global filters, all charts & tables
data/
  sales_data.csv        # raw source data
outputs/                # Python-generated, browser-ready data (see below)
  processed_data.csv    # full row-level dataset + engineered features
  predictions.csv       # held-out test-set actual vs predicted demand
  model_metrics.json    # MAE / RMSE / R² and train/test split info
  feature_importance.json
  correlation.json
  insights.json
notebooks/
  Inventory_demand_prediction.ipynb
process_data.py         # regenerates everything in outputs/ from data/sales_data.csv
```

## How it works

`process_data.py` does the heavy lifting in Python: cleaning, feature
engineering (calendar features, lags, rolling means, inventory gap/coverage,
stockout flag), a time-based train/test split, and a `RandomForestRegressor`
demand-forecast model. It exports compact CSV/JSON files into `outputs/`.

`index.html` + `assets/app.js` load those files with PapaParse/fetch **once**,
then do all filtering and aggregation **in the browser** — that's what makes
the global filter bar (Date, Store, Product, Category, Region, Season,
Promotion, Epidemic) update every KPI, chart and table live without a backend.

To regenerate the outputs after changing the data or model:

```bash
pip install pandas numpy scikit-learn
python3 process_data.py
```

## Running locally

Because the dashboard `fetch`/loads files from `outputs/`, opening
`index.html` directly via `file://` will be blocked by the browser's CORS
policy. Serve the folder instead:

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

## Deploying

**GitHub Pages**
1. Push this whole folder to a GitHub repo.
2. Repo Settings → Pages → Deploy from branch → `main` / root.
3. Your dashboard will be live at `https://<user>.github.io/<repo>/`.

**Vercel**
1. Import the repo at vercel.com/new (framework preset: "Other" / static).
2. No build command needed — it's a static site. Deploy.

## Notes / assumptions

- Page 5's "Selection Panel" (Store/Product/Category/Region) is intentionally
  independent of the global filter bar on Pages 1–4, per the brief's separate
  treatment of drill-down selection vs. global filters.
- "Forecast MAE / R²" on the Executive Overview are the fixed model metrics
  from the held-out test set (retraining per filter combination isn't
  feasible client-side). The Demand Forecasting page's MAE/RMSE/R² KPIs,
  by contrast, **do** recompute live from whichever test-set rows match the
  current filters.
- Scatter charts sample up to ~1,500 points per view for readability and
  performance; all bar/line/table aggregates use the full filtered dataset.
