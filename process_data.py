"""
Retail Demand & Inventory Analytics — Data Processing Pipeline
================================================================
Reads data/sales_data.csv, performs cleaning + feature engineering,
trains a Random Forest demand-forecasting model (time-based split),
and exports compact, browser-ready files into outputs/ for the
HTML dashboard (index.html) to consume.

Run:  python3 process_data.py
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option("display.width", 140)

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------
# 1. LOAD + CLEAN
# ------------------------------------------------------------------
df = pd.read_csv("data/sales_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.drop_duplicates()
df = df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)

print("Rows:", len(df), "| Date range:", df["Date"].min().date(), "to", df["Date"].max().date())

# ------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ------------------------------------------------------------------
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
df["DayOfWeek"] = df["Date"].dt.dayofweek

df["Price Gap"] = df["Price"] - df["Competitor Pricing"]
df["Inventory Gap"] = df["Inventory Level"] - df["Demand"]
df["Stockout Risk Flag"] = (df["Inventory Level"] < df["Demand"]).astype(int)
df["Inventory Coverage"] = df["Inventory Level"] / df["Demand"].replace(0, np.nan)
df["Inventory Coverage"] = df["Inventory Coverage"].clip(upper=5)  # cap extreme ratios for charting

grp = df.groupby(["Store ID", "Product ID"])["Demand"]
df["Demand_Lag_1"] = grp.shift(1)
df["Demand_Lag_7"] = grp.shift(7)
df["Demand_Lag_28"] = grp.shift(28)
df["Demand_Rolling_7"] = df.groupby(["Store ID", "Product ID"])["Demand"].transform(
    lambda x: x.shift(1).rolling(7).mean()
)
df["Demand_Rolling_28"] = df.groupby(["Store ID", "Product ID"])["Demand"].transform(
    lambda x: x.shift(1).rolling(28).mean()
)

# ------------------------------------------------------------------
# 3. FORECASTING MODEL (Random Forest, time-based split)
# ------------------------------------------------------------------
model_df = df.dropna(
    subset=["Demand_Lag_1", "Demand_Lag_7", "Demand_Lag_28", "Demand_Rolling_7", "Demand_Rolling_28"]
).copy()

num_features = [
    "Inventory Level", "Price", "Discount", "Promotion", "Competitor Pricing", "Epidemic",
    "Year", "Month", "Week", "DayOfWeek",
    "Demand_Lag_1", "Demand_Lag_7", "Demand_Lag_28", "Demand_Rolling_7", "Demand_Rolling_28",
]
cat_features = ["Store ID", "Product ID", "Category", "Region", "Weather Condition", "Seasonality"]

enc_df = pd.get_dummies(model_df, columns=cat_features, drop_first=True)
encoded_features = [c for c in enc_df.columns if c in num_features or
                     any(c.startswith(x + "_") for x in cat_features)]

X = enc_df[encoded_features]
y = enc_df["Demand"]

split_date = model_df["Date"].quantile(0.8)
train_mask = model_df["Date"].values <= split_date
test_mask = ~train_mask

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

model = RandomForestRegressor(
    n_estimators=200, max_depth=14, min_samples_leaf=3, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)

print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}")

test_meta = model_df.loc[
    test_mask,
    ["Date", "Store ID", "Product ID", "Category", "Region", "Seasonality", "Promotion", "Epidemic", "Demand"],
].copy()
test_meta["Predicted Demand"] = np.round(y_pred, 2)
test_meta["Error"] = test_meta["Demand"] - test_meta["Predicted Demand"]

feat_imp = (
    pd.DataFrame({"Feature": encoded_features, "Importance": model.feature_importances_})
    .sort_values("Importance", ascending=False)
    .head(15)
    .reset_index(drop=True)
)

# ------------------------------------------------------------------
# 4. EXPORT — model metrics
# ------------------------------------------------------------------
with open(f"{OUT}/model_metrics.json", "w") as f:
    json.dump({
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4),
        "train_start": str(model_df.loc[train_mask, "Date"].min().date()),
        "train_end": str(model_df.loc[train_mask, "Date"].max().date()),
        "test_start": str(model_df.loc[test_mask, "Date"].min().date()),
        "test_end": str(model_df.loc[test_mask, "Date"].max().date()),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_estimators": 200,
        "model": "RandomForestRegressor",
    }, f, indent=2)

# ------------------------------------------------------------------
# 5. EXPORT — feature importance
# ------------------------------------------------------------------
def pretty_feature(name):
    mapping = {
        "Inventory Level": "Inventory Level", "Price": "Price", "Discount": "Discount",
        "Promotion": "Promotion", "Competitor Pricing": "Competitor Pricing", "Epidemic": "Epidemic",
        "Year": "Year", "Month": "Month", "Week": "Week of Year", "DayOfWeek": "Day of Week",
        "Demand_Lag_1": "Demand (1 day ago)", "Demand_Lag_7": "Demand (7 days ago)",
        "Demand_Lag_28": "Demand (28 days ago)", "Demand_Rolling_7": "Rolling Avg Demand (7d)",
        "Demand_Rolling_28": "Rolling Avg Demand (28d)",
    }
    if name in mapping:
        return mapping[name]
    for pre in cat_features:
        if name.startswith(pre + "_"):
            return f"{pre}: {name[len(pre)+1:]}"
    return name

feat_imp["Label"] = feat_imp["Feature"].apply(pretty_feature)
feat_imp[["Label", "Importance"]].round({"Importance": 5}).to_json(
    f"{OUT}/feature_importance.json", orient="records", indent=2
)

# ------------------------------------------------------------------
# 6. EXPORT — predictions (Page 4: Forecasting)
# ------------------------------------------------------------------
test_meta_out = test_meta.copy()
test_meta_out["Date"] = test_meta_out["Date"].dt.strftime("%Y-%m-%d")
test_meta_out.rename(columns={
    "Store ID": "Store", "Product ID": "Product", "Seasonality": "Season", "Promotion": "Promo",
    "Demand": "Actual", "Predicted Demand": "Predicted",
}, inplace=True)
test_meta_out["Predicted"] = test_meta_out["Predicted"].round(1)
test_meta_out["Error"] = test_meta_out["Error"].round(1)
test_meta_out = test_meta_out[
    ["Date", "Store", "Product", "Category", "Region", "Season", "Promo", "Epidemic", "Actual", "Predicted", "Error"]
]
test_meta_out.to_csv(f"{OUT}/predictions.csv", index=False)

# ------------------------------------------------------------------
# 7. EXPORT — correlation matrix (Demand vs key numeric drivers)
# ------------------------------------------------------------------
corr_cols = ["Demand", "Inventory Level", "Price", "Discount", "Promotion",
             "Competitor Pricing", "Epidemic", "Units Sold", "Units Ordered"]
corr = df[corr_cols].corr()["Demand"].drop("Demand").sort_values(key=abs, ascending=False)
corr_out = [{"feature": k, "correlation": round(float(v), 4)} for k, v in corr.items()]
with open(f"{OUT}/correlation.json", "w") as f:
    json.dump(corr_out, f, indent=2)

# ------------------------------------------------------------------
# 8. EXPORT — full processed row-level dataset (compact CSV for client-side filtering)
#    Pages 1, 2, 3, 5 filter/aggregate this in the browser.
# ------------------------------------------------------------------
export_cols = {
    "Date": "Date", "Store ID": "Store", "Product ID": "Product", "Category": "Category",
    "Region": "Region", "Seasonality": "Season", "Promotion": "Promo", "Epidemic": "Epidemic",
    "Weather Condition": "Weather", "Inventory Level": "Inventory", "Units Sold": "UnitsSold",
    "Units Ordered": "UnitsOrdered", "Price": "Price", "Discount": "Discount",
    "Competitor Pricing": "CompPrice", "Demand": "Demand", "Inventory Gap": "InvGap",
    "Stockout Risk Flag": "Stockout", "Inventory Coverage": "Coverage",
}
export_df = df[list(export_cols.keys())].rename(columns=export_cols)
export_df["Date"] = export_df["Date"].dt.strftime("%Y-%m-%d")
for c in ["Inventory", "UnitsSold", "UnitsOrdered", "Demand", "InvGap", "Stockout"]:
    export_df[c] = export_df[c].astype(int)
export_df["Price"] = export_df["Price"].round(2)
export_df["CompPrice"] = export_df["CompPrice"].round(2)
export_df["Coverage"] = export_df["Coverage"].round(3)
export_df.to_csv(f"{OUT}/processed_data.csv", index=False)

# ------------------------------------------------------------------
# 9. EXPORT — product summary table (Page 2 + Page 5)
# ------------------------------------------------------------------
product_summary = (
    df.groupby(["Product ID", "Category"])
    .agg(
        AvgDemand=("Demand", "mean"),
        TotalDemand=("Demand", "sum"),
        AvgInventory=("Inventory Level", "mean"),
        StockoutRisk=("Stockout Risk Flag", "mean"),
        AvgCoverage=("Inventory Coverage", "mean"),
        AvgPrice=("Price", "mean"),
    )
    .reset_index()
    .rename(columns={"Product ID": "Product", "Category": "Category"})
)
product_summary["AvgDemand"] = product_summary["AvgDemand"].round(1)
product_summary["AvgInventory"] = product_summary["AvgInventory"].round(1)
product_summary["StockoutRisk"] = (product_summary["StockoutRisk"] * 100).round(1)
product_summary["AvgCoverage"] = product_summary["AvgCoverage"].round(2)
product_summary["AvgPrice"] = product_summary["AvgPrice"].round(2)
product_summary.to_json(f"{OUT}/product_summary.json", orient="records")

# ------------------------------------------------------------------
# 10. EXPORT — auto-generated business insights
# ------------------------------------------------------------------
top_cat = df.groupby("Category")["Demand"].mean().idxmax()
top_region = df.groupby("Region")["Demand"].mean().idxmax()
risk_cat = df.groupby("Category")["Stockout Risk Flag"].mean().idxmax()
risk_cat_val = df.groupby("Category")["Stockout Risk Flag"].mean().max() * 100
epi_diff = df.groupby("Epidemic")["Demand"].mean()
epi_pct = (epi_diff.get(1, 0) - epi_diff.get(0, 0)) / epi_diff.get(0, 1) * 100
promo_diff = df.groupby("Promotion")["Demand"].mean()
promo_pct = (promo_diff.get(1, 0) - promo_diff.get(0, 0)) / promo_diff.get(0, 1) * 100
overall_stockout = df["Stockout Risk Flag"].mean() * 100

insights = [
    f"{top_cat} is the highest-demand category, averaging {df.groupby('Category')['Demand'].mean().max():.0f} units per order line.",
    f"The {top_region} region generates the highest average demand across all stores.",
    f"{risk_cat} carries the highest stockout risk at {risk_cat_val:.1f}% of observations, and needs closer inventory attention.",
    f"Demand during epidemic periods is {epi_pct:+.1f}% compared to normal periods.",
    f"Promotional periods show {promo_pct:+.1f}% average demand versus non-promotional periods.",
    f"Overall stockout risk sits at {overall_stockout:.1f}% of all store-product-day observations.",
]
with open(f"{OUT}/insights.json", "w") as f:
    json.dump(insights, f, indent=2)

print("\nAll outputs written to ./outputs/")
for fn in sorted(os.listdir(OUT)):
    size_kb = os.path.getsize(os.path.join(OUT, fn)) / 1024
    print(f"  {fn:28s} {size_kb:8.1f} KB")
