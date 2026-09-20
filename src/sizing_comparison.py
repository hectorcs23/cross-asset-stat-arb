"""
sizing_comparison.py
====================

Compares the current equal-weight position sizing (portfolio_value / MAX_POSITIONS)
against four dynamic sizing schemes, all run through the same entry/exit logic
as paper_trader.py to isolate the effect of sizing alone.

Sizing variants
---------------
  0. EqualWeight     — baseline: w_i = gross_leverage / N_active         (100k/8 mode)
  1. SignalWeight    — w_i ∝ |z_i|                                       (strength only)
  2. InverseVol      — w_i ∝ 1 / σ_i (60d rolling)                       (risk parity)
  3. Combined        — w_i ∝ |z_i| / σ_i                                 (Kelly-flavored)
  4. VolTargeted     — Combined rescaled so portfolio σ ≈ TARGET_VOL     (adaptive leverage)

All share the same ensemble z-scores, entry/exit thresholds, holding cap,
per-position cap (20%), and transaction cost (5 bps).

Input: z_scores.pkl (prices + z_coint + z_pca + z_factor, cached).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Configuration — matches paper_trader.py
# ---------------------------------------------------------------------
CAPITAL             = 100_000.0
ENTRY_Z             = 0.3
EXIT_Z              = 0.05
MAX_POSITIONS       = 8
MAX_HOLDING_DAYS    = 20
GROSS_LEVERAGE      = 1.0
MAX_POSITION_PCT    = 0.20
TXN_COST_BPS        = 5.0
VOL_LOOKBACK        = 60        # days for rolling per-asset vol
TARGET_VOL_ANN      = 0.10      # 10% annualized portfolio vol target
ANN                 = 252


# ---------------------------------------------------------------------
# Ensemble z-score builder (vectorized across dates)
# ---------------------------------------------------------------------
def build_ensemble_panel(z_dict: Dict[str, pd.DataFrame],
                         require_unanimous_sign: bool = True) -> pd.DataFrame:
    """
    Vectorized version of cross_asset_statarb.ensemble_signal applied to every date.
    Returns a DataFrame (T × N) of composite z-scores.
    """
    # Align all z-score panels on common index/columns
    frames = list(z_dict.values())
    idx = frames[0].index
    cols = frames[0].columns
    for f in frames[1:]:
        idx = idx.intersection(f.index)
        cols = cols.intersection(f.columns)
    frames = [f.reindex(index=idx, columns=cols) for f in frames]

    # Stack frames into a 3D array: (models, T, N)
    arr = np.stack([f.values for f in frames], axis=0)  # (M, T, N)

    # Require unanimous sign across models (ignoring NaNs)
    signs = np.sign(arr)
    with np.errstate(invalid="ignore"):
        all_pos = np.all((signs > 0) | np.isnan(arr), axis=0) & np.any(signs > 0, axis=0)
        all_neg = np.all((signs < 0) | np.isnan(arr), axis=0) & np.any(signs < 0, axis=0)
    agree = all_pos | all_neg if require_unanimous_sign else np.ones_like(all_pos, dtype=bool)

    # Rank within each day across cross-section (per model), then average ranks
    # We use a simple mean of the z-scores as composite magnitude (sign from sign_agree)
    # This matches ensemble_signal's intent: rank-pct mean centered at 0, rescaled by |mean|.
    # We'll replicate it date-by-date for fidelity.
    T, N = arr.shape[1], arr.shape[2]
    composite = np.full((T, N), np.nan)

    for t in range(T):
        row = arr[:, t, :]           # (M, N)
        valid = ~np.isnan(row).all(axis=0)
        if not valid.any():
            continue
        sub = row[:, valid]          # (M, n_valid)
        if require_unanimous_sign:
            keep = agree[t, valid]
            if not keep.any():
                continue
        else:
            keep = np.ones(sub.shape[1], dtype=bool)

        # Rank-pct per model (nan-aware), mean across models, center at 0
        ranks = np.full_like(sub, np.nan)
        for m in range(sub.shape[0]):
            v = sub[m]
            mask = ~np.isnan(v)
            if mask.sum() == 0:
                continue
            order = v[mask].argsort().argsort()  # ranks 0..k-1
            pct = (order + 1) / (mask.sum() + 1)
            ranks[m, mask] = pct
        rank_mean = np.nanmean(ranks, axis=0) - 0.5
        scale = 2 * np.nanmean(np.abs(sub), axis=0)
        comp = rank_mean * scale
        comp[~keep] = np.nan

        idx_valid = np.where(valid)[0]
        composite[t, idx_valid] = comp

    return pd.DataFrame(composite, index=idx, columns=cols)


# ---------------------------------------------------------------------
# Rolling volatility panel
# ---------------------------------------------------------------------
def rolling_vol(prices: pd.DataFrame, window: int = VOL_LOOKBACK) -> pd.DataFrame:
    """Daily log-return std, rolling window, annualized (per asset)."""
    r = np.log(prices).diff()
    return r.rolling(window, min_periods=max(10, window // 3)).std() * np.sqrt(ANN)


# ---------------------------------------------------------------------
# Sizing functions
#   Inputs:
#     active: dict {ticker: {side: +/-1, days: n}}
#     z_row:  pd.Series of latest ensemble z (index = tickers)
#     vol_row: pd.Series of latest annualized vol (index = tickers)
#     port_ret_recent: pd.Series of recent portfolio returns (for vol targeting)
#   Output:
#     dict {ticker: signed weight}   (weights are fractions of portfolio)
#
#   All sizing respects:
#     - gross leverage cap (Σ|w| ≤ GROSS_LEVERAGE)
#     - per-position cap (|w_i| ≤ MAX_POSITION_PCT)
# ---------------------------------------------------------------------
def _normalize(raw: Dict[str, float], gross: float, cap: float) -> Dict[str, float]:
    """Scale so Σ|w| = gross, then enforce per-position cap and re-normalize."""
    if not raw:
        return {}
    s = sum(abs(v) for v in raw.values())
    if s == 0:
        return {k: 0.0 for k in raw}
    w = {k: v * gross / s for k, v in raw.items()}
    # Cap
    capped = {k: np.sign(v) * min(abs(v), cap) for k, v in w.items()}
    # After capping, Σ|w| may shrink; that's acceptable (we accept lower gross rather than exceed cap)
    return capped


def size_equal(active, z_row, vol_row, port_ret_recent):
    """Baseline: equal weight across active positions."""
    if not active:
        return {}
    w_each = GROSS_LEVERAGE / len(active)
    return {t: info["side"] * min(w_each, MAX_POSITION_PCT) for t, info in active.items()}


def size_signal(active, z_row, vol_row, port_ret_recent):
    """w_i ∝ |z_i|."""
    raw = {}
    for t, info in active.items():
        z = z_row.get(t, np.nan)
        if pd.isna(z) or z == 0:
            continue
        raw[t] = info["side"] * abs(z)
    return _normalize(raw, GROSS_LEVERAGE, MAX_POSITION_PCT)


def size_invvol(active, z_row, vol_row, port_ret_recent):
    """w_i ∝ 1 / σ_i (risk parity per leg)."""
    raw = {}
    for t, info in active.items():
        v = vol_row.get(t, np.nan)
        if pd.isna(v) or v <= 0:
            continue
        raw[t] = info["side"] * (1.0 / v)
    return _normalize(raw, GROSS_LEVERAGE, MAX_POSITION_PCT)


def size_combined(active, z_row, vol_row, port_ret_recent):
    """w_i ∝ |z_i| / σ_i."""
    raw = {}
    for t, info in active.items():
        z = z_row.get(t, np.nan)
        v = vol_row.get(t, np.nan)
        if pd.isna(z) or pd.isna(v) or v <= 0:
            continue
        raw[t] = info["side"] * (abs(z) / v)
    return _normalize(raw, GROSS_LEVERAGE, MAX_POSITION_PCT)


def size_voltargeted(active, z_row, vol_row, port_ret_recent):
    """Combined, then rescale gross to hit TARGET_VOL_ANN (using realized portfolio vol)."""
    base = size_combined(active, z_row, vol_row, port_ret_recent)
    if not base:
        return base
    # Estimate current portfolio vol from recent realized returns; fall back to
    # a weighted-sum estimate using per-asset vols (correlation=0 approximation).
    if port_ret_recent is not None and len(port_ret_recent.dropna()) >= 20:
        cur_vol = port_ret_recent.dropna().std() * np.sqrt(ANN)
    else:
        # Conservative proxy: Σ|w_i|·σ_i (upper bound assuming perfect correlation)
        cur_vol = sum(abs(w) * vol_row.get(t, 0.2) for t, w in base.items())
    if cur_vol <= 0:
        return base
    scale = TARGET_VOL_ANN / cur_vol
    scale = float(np.clip(scale, 0.25, 2.0))   # never go below 0.25× or above 2×
    scaled = {t: w * scale for t, w in base.items()}
    # Re-apply per-position cap
    return {t: np.sign(w) * min(abs(w), MAX_POSITION_PCT) for t, w in scaled.items()}


SIZERS: Dict[str, Callable] = {
    "EqualWeight  (baseline, current bot)": size_equal,
    "SignalWeight (w ~ |z|)":                size_signal,
    "InverseVol   (w ~ 1/sigma)":            size_invvol,
    "Combined     (w ~ |z|/sigma)":          size_combined,
    "VolTargeted  (Combined, sigma_p->10%)": size_voltargeted,
}


# ---------------------------------------------------------------------
# Backtest engine (identical entry/exit across variants; only sizing differs)
# ---------------------------------------------------------------------
@dataclass
class Result:
    name: str
    equity: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    turnover: pd.Series


def backtest(prices: pd.DataFrame,
             z_ens: pd.DataFrame,
             vol: pd.DataFrame,
             sizing_fn: Callable,
             name: str) -> Result:
    dates = z_ens.index.intersection(prices.index).intersection(vol.index)
    z = z_ens.reindex(dates)
    P = prices.reindex(dates)
    V = vol.reindex(dates)
    R = np.log(P).diff()

    positions = pd.DataFrame(0.0, index=dates, columns=P.columns)
    active: Dict[str, dict] = {}
    port_ret_hist: list = []

    for ti in range(1, len(dates)):
        today, yday = dates[ti], dates[ti - 1]
        z_y = z.loc[yday].dropna()

        # Exit
        to_close = []
        for t, info in active.items():
            info["days"] += 1
            zn = z.loc[yday, t] if t in z.columns else np.nan
            if (pd.notna(zn) and abs(zn) < EXIT_Z) or info["days"] >= MAX_HOLDING_DAYS:
                to_close.append(t)
        for t in to_close:
            active.pop(t)

        # Entry
        longs = z_y[z_y < -ENTRY_Z].sort_values().index.tolist()
        shorts = z_y[z_y > +ENTRY_Z].sort_values(ascending=False).index.tolist()
        slots = MAX_POSITIONS - len(active)
        for t in longs:
            if slots <= 0: break
            if t not in active:
                active[t] = {"side": +1, "days": 0}
                slots -= 1
        for t in shorts:
            if slots <= 0: break
            if t not in active:
                active[t] = {"side": -1, "days": 0}
                slots -= 1

        # Sizing
        recent_ret = pd.Series(port_ret_hist[-VOL_LOOKBACK:]) if port_ret_hist else None
        weights = sizing_fn(active, z_y, V.loc[yday], recent_ret)
        for t, w in weights.items():
            positions.loc[today, t] = w

        # Record today's portfolio return for vol targeting feedback
        w_lag = positions.iloc[ti - 1] if ti >= 1 else positions.iloc[0]
        r_t = (w_lag * R.loc[today]).sum()
        port_ret_hist.append(r_t)

    # P&L with txn cost
    w_lag = positions.shift(1).fillna(0)
    gross = (w_lag * R).sum(axis=1)
    turnover = (positions - positions.shift(1).fillna(0)).abs().sum(axis=1)
    txn = turnover * (TXN_COST_BPS / 1e4)
    port_ret = gross - txn

    equity = CAPITAL * np.exp(port_ret.cumsum())
    return Result(name=name, equity=equity, returns=port_ret,
                  positions=positions, turnover=turnover)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def metrics(r: Result) -> dict:
    ret = r.returns.dropna()
    eq = r.equity.dropna()
    if len(ret) == 0 or eq.iloc[-1] == 0:
        return {}
    mean, std = ret.mean(), ret.std()
    downside = ret[ret < 0].std()
    sharpe = mean / std * np.sqrt(ANN) if std > 0 else 0.0
    sortino = mean / downside * np.sqrt(ANN) if downside and downside > 0 else np.nan
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (ANN / len(ret)) - 1
    cummax = eq.cummax()
    dd = (eq - cummax) / cummax
    maxdd = dd.min()
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    hit = (ret > 0).sum() / (ret != 0).sum() if (ret != 0).any() else np.nan
    avg_gross = r.positions.abs().sum(axis=1).mean()
    avg_turn = r.turnover.mean()
    return {
        "CAGR":       cagr,
        "Vol (ann)":  std * np.sqrt(ANN),
        "Sharpe":     sharpe,
        "Sortino":    sortino,
        "MaxDD":      maxdd,
        "Calmar":     calmar,
        "HitRate":    hit,
        "AvgGross":   avg_gross,
        "AvgTurn":    avg_turn,
        "FinalEq":    eq.iloc[-1],
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    cache_path = Path("z_scores.pkl")
    print(f"Loading cached data from {cache_path}...")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    prices = cache["prices"]
    z_dict = {"coint": cache["z_coint"], "pca": cache["z_pca"], "factor": cache["z_factor"]}
    print(f"  prices: {prices.shape[0]} days × {prices.shape[1]} tickers")
    print(f"  date range: {prices.index[0].date()} → {prices.index[-1].date()}")

    print("\nBuilding ensemble z-score panel (vectorized)...")
    z_ens = build_ensemble_panel(z_dict, require_unanimous_sign=True)
    print(f"  ensemble shape: {z_ens.shape}, non-null cells: {z_ens.notna().sum().sum():,}")

    print("\nComputing rolling volatility (60d, annualized)...")
    vol = rolling_vol(prices, VOL_LOOKBACK)

    # Evaluate on out-of-sample period (leave first ~2 years as burn-in / lookback)
    start = pd.Timestamp("2017-01-01")
    prices_oos = prices.loc[start:]
    z_oos = z_ens.loc[start:]
    vol_oos = vol.loc[start:]
    print(f"\nOOS window: {prices_oos.index[0].date()} → {prices_oos.index[-1].date()}   ({len(prices_oos)} days)")

    results = []
    print("\n" + "="*76)
    print(f"{'Strategy':<42s}  {'CAGR':>7s}  {'Vol':>6s}  {'Sharpe':>7s}  {'MaxDD':>7s}")
    print("="*76)
    for name, fn in SIZERS.items():
        res = backtest(prices_oos, z_oos, vol_oos, fn, name)
        m = metrics(res)
        results.append((res, m))
        print(f"{name:<42s}  {m['CAGR']*100:+6.1f}%  {m['Vol (ann)']*100:5.1f}%  "
              f"{m['Sharpe']:+6.2f}  {m['MaxDD']*100:+6.1f}%")
    print("="*76)

    # Full metrics table
    print("\nFull metrics:")
    df_metrics = pd.DataFrame({r.name: m for r, m in results}).T
    df_metrics["CAGR"]      = df_metrics["CAGR"].map(lambda x: f"{x*100:+.2f}%")
    df_metrics["Vol (ann)"] = df_metrics["Vol (ann)"].map(lambda x: f"{x*100:.2f}%")
    df_metrics["MaxDD"]     = df_metrics["MaxDD"].map(lambda x: f"{x*100:+.2f}%")
    df_metrics["HitRate"]   = df_metrics["HitRate"].map(lambda x: f"{x*100:.1f}%")
    df_metrics["AvgGross"]  = df_metrics["AvgGross"].map(lambda x: f"{x:.2f}")
    df_metrics["AvgTurn"]   = df_metrics["AvgTurn"].map(lambda x: f"{x:.3f}")
    df_metrics["Sharpe"]    = df_metrics["Sharpe"].map(lambda x: f"{x:+.2f}")
    df_metrics["Sortino"]   = df_metrics["Sortino"].map(lambda x: f"{x:+.2f}")
    df_metrics["Calmar"]    = df_metrics["Calmar"].map(lambda x: f"{x:+.2f}")
    df_metrics["FinalEq"]   = df_metrics["FinalEq"].map(lambda x: f"${x:,.0f}")
    print(df_metrics.to_string())

    # Save CSV
    pd.DataFrame({r.name: m for r, m in results}).T.to_csv("sizing_metrics.csv")
    print("\nSaved: sizing_metrics.csv")

    # Plots
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax1, ax2, ax3, ax4 = axes.flat

    # 1) Equity curves
    for res, _ in results:
        ax1.plot(res.equity.index, res.equity.values, label=res.name, linewidth=1.3)
    ax1.set_title("Equity curves — same signals, different sizing")
    ax1.set_ylabel("Equity ($)"); ax1.set_yscale("log")
    ax1.legend(fontsize=8, loc="upper left"); ax1.grid(alpha=0.3)

    # 2) Drawdown
    for res, _ in results:
        eq = res.equity
        dd = (eq - eq.cummax()) / eq.cummax()
        ax2.plot(dd.index, dd.values * 100, label=res.name, linewidth=1.1)
    ax2.set_title("Drawdown (%)")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=8, loc="lower left"); ax2.grid(alpha=0.3)

    # 3) 252d rolling Sharpe
    for res, _ in results:
        rs = res.returns.rolling(252).mean() / res.returns.rolling(252).std() * np.sqrt(ANN)
        ax3.plot(rs.index, rs.values, label=res.name, linewidth=1.1)
    ax3.axhline(0, color="k", linewidth=0.5, alpha=0.5)
    ax3.set_title("Rolling 252-day Sharpe")
    ax3.set_ylabel("Sharpe")
    ax3.legend(fontsize=8, loc="best"); ax3.grid(alpha=0.3)

    # 4) Gross leverage used over time
    for res, _ in results:
        g = res.positions.abs().sum(axis=1).rolling(21).mean()
        ax4.plot(g.index, g.values, label=res.name, linewidth=1.1)
    ax4.axhline(GROSS_LEVERAGE, color="k", linestyle="--", linewidth=0.7, alpha=0.6, label="target=1.0")
    ax4.set_title("Gross leverage (21d MA)")
    ax4.set_ylabel("Σ|w_i|")
    ax4.legend(fontsize=8, loc="best"); ax4.grid(alpha=0.3)

    fig.suptitle(f"Position sizing comparison — OOS {prices_oos.index[0].year}–{prices_oos.index[-1].year}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("sizing_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: sizing_comparison.png")

    return results, df_metrics


if __name__ == "__main__":
    main()
