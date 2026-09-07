from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {
    "calendar.csv": {"date", "wm_yr_wk", "d"},
    "sell_prices.csv": {"store_id", "item_id", "wm_yr_wk", "sell_price"},
}


def validate_raw_files(raw_dir: Path) -> None:
    sales = raw_dir / "sales_train_evaluation.csv"
    if not sales.exists():
        raise FileNotFoundError(f"Missing {sales.name}. Place the M5 CSV files in {raw_dir}")
    for filename, cols in REQUIRED.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {filename}. Place the M5 CSV files in {raw_dir}")
        actual = set(pd.read_csv(path, nrows=2).columns)
        missing = cols - actual
        if missing:
            raise ValueError(f"{filename} missing columns: {sorted(missing)}")


def _pick_series(sales: pd.DataFrame, max_series: int) -> pd.DataFrame:
    """Choose a deterministic spread of active series across velocity levels.

    Slow-selling analysis must not sample only top sellers. Within each
    category/store group we sort by recent units and take evenly spaced ranks.
    """
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    recent = day_cols[-56:]
    sales = sales.copy()
    sales["recent_units"] = sales[recent].sum(axis=1)
    active = sales[sales["recent_units"] > 0].copy()
    if len(active) <= max_series:
        return active.drop(columns="recent_units")

    groups = list(active.groupby(["cat_id", "store_id"], sort=True))
    base = max_series // len(groups)
    remainder = max_series % len(groups)
    picked = []
    for i, (_, group) in enumerate(groups):
        n = min(len(group), base + (1 if i < remainder else 0))
        if n <= 0:
            continue
        group = group.sort_values(["recent_units", "id"])
        positions = np.linspace(0, len(group) - 1, n, dtype=int)
        picked.append(group.iloc[positions])
    chosen = pd.concat(picked) if picked else active.head(0)
    if len(chosen) < max_series:
        extra = active.loc[~active.index.isin(chosen.index)].sort_values(["cat_id", "store_id", "recent_units", "id"]).head(max_series - len(chosen))
        chosen = pd.concat([chosen, extra])
    return chosen.head(max_series).drop(columns="recent_units")


def load_m5_long(raw_dir: Path, max_series: int = 500) -> pd.DataFrame:
    validate_raw_files(raw_dir)
    sales = pd.read_csv(raw_dir / "sales_train_evaluation.csv")
    sales = _pick_series(sales, max_series)
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    long = sales[id_cols + day_cols].melt(id_vars=id_cols, var_name="d", value_name="sales")

    calendar = pd.read_csv(raw_dir / "calendar.csv")
    keep = [c for c in ["d", "date", "wm_yr_wk", "wday", "month", "year", "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"] if c in calendar.columns]
    long = long.merge(calendar[keep], on="d", how="left")
    long["date"] = pd.to_datetime(long["date"])

    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    snap_cols = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    long["snap"] = [long.loc[i, snap_cols.get(state, "snap_CA")] if snap_cols.get(state) in long.columns else 0 for i, state in enumerate(long["state_id"])]
    long["event"] = long.get("event_name_1", pd.Series(index=long.index, dtype="object")).notna().astype("int8")
    long = long.sort_values(["id", "date"]).reset_index(drop=True)
    return long


def make_demo_data(n_series: int = 12, n_days: int = 240, seed: int = 42) -> pd.DataFrame:
    """Synthetic data only for tests/smoke demos; never used for portfolio metrics."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for s in range(n_series):
        base = rng.uniform(1.5, 8)
        price0 = rng.uniform(2, 12)
        trend = rng.uniform(-0.004, 0.002)
        for t, date in enumerate(dates):
            markdown = 0.10 if (t > 150 and s % 3 == 0) else 0
            price = price0 * (1 - markdown)
            seasonal = 1 + 0.18 * np.sin(2 * np.pi * date.dayofweek / 7)
            demand = max(0.05, base * seasonal * (1 + trend * t) * (1 + 1.4 * markdown))
            rows.append({
                "id": f"ITEM_{s:03d}_STORE_1_evaluation", "item_id": f"ITEM_{s:03d}",
                "dept_id": "D1", "cat_id": "FOODS" if s % 2 == 0 else "HOUSEHOLD",
                "store_id": "STORE_1", "state_id": "CA", "date": date,
                "sales": int(rng.poisson(demand)), "sell_price": round(price, 2),
                "wday": date.dayofweek + 1, "month": date.month, "year": date.year,
                "snap": int(date.dayofweek in [4, 5]), "event": int(date.day == 1),
            })
    return pd.DataFrame(rows).sort_values(["id", "date"]).reset_index(drop=True)
