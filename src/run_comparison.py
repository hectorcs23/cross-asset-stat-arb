"""
run_comparison.py
=================

End-to-end comparison of the three mispricing frameworks on a macro
cross-asset universe. Reproduces the setup used in the project (S. Mejía & H. Campbell).

Usage:
    pip install yfinance statsmodels numpy pandas scipy matplotlib
    python run_comparison.py

Outputs:
    - Console summary of each framework's Sharpe / drawdown
    - equity_curves.png
    - correlations.png (Pearson correlation between framework signals)
    - factor_exposures.csv (economic exposures from Framework 3)
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from cross_asset_statarb import (
    flat_universe, fetch_prices,
    CointegrationModel, PCAResidualModel, FactorResidualModel,
    build_all_z_scores, backtest_signal, ensemble_signal,
)


def main():
    # ---- 1. Data ----
    tickers = flat_universe()
    print(f"Universe: {len(tickers)} tickers")
    prices = fetch_prices(tickers, start="2015-01-01")
    print(f"Price panel: {prices.shape[0]} days × {prices.shape[1]} tickers")

    # ---- 2. Split: in-sample for fitting, out-of-sample for honest backtest ----
    split = int(len(prices) * 0.6)
    in_sample = prices.iloc[:split]
    out_sample = prices.iloc[split - 252:]   # leave a lookback buffer

    print(f"\nIn-sample (fit):  {in_sample.index[0].date()}  →  {in_sample.index[-1].date()}")
    print(f"Out-of-sample:    {out_sample.index[0].date()}  →  {out_sample.index[-1].date()}")

    # ---- 3. Fit each framework on FULL history (rolling) ----
    print("\n--- Fitting frameworks ---")
    z_coint = build_all_z_scores(CointegrationModel, prices,
                                  fdr_alpha=0.05, half_life_bounds=(2, 60))
    z_pca = build_all_z_scores(PCAResidualModel, prices,
                                lookback=252, refit_every=21,
                                half_life_bounds=(2, 60))
    z_factor = build_all_z_scores(FactorResidualModel, prices,
                                   lookback=252, half_life_bounds=(2, 60))

    # ---- 4. Restrict to OOS for backtest ----
    oos_idx = out_sample.index
    z_coint_oos = z_coint.reindex(oos_idx)
    z_pca_oos = z_pca.reindex(oos_idx)
    z_factor_oos = z_factor.reindex(oos_idx)
    prices_oos = prices.reindex(oos_idx)

    # ---- 5. Backtest each ----
    print("\n--- Backtests (out-of-sample) ---")
    results = {}
    for name, z in [("Cointegration",      z_coint_oos),
                    ("PCA/Avellaneda-Lee", z_pca_oos),
                    ("Factor residuals",    z_factor_oos)]:
        res = backtest_signal(prices_oos, z,
                              entry_z=1.5, exit_z=0.25,
                              max_positions=10, holding_cap_days=20,
                              transaction_cost_bps=5.0)
        results[name] = res
        print(f"{name:<22s}  Sharpe={res.sharpe:+.2f}  "
              f"MaxDD={res.max_drawdown*100:+.1f}%  "
              f"HitRate={res.hit_rate*100:.1f}%  "
              f"Trades={res.n_trades}")

    # ---- 6. Ensemble ----
    print("\n--- Ensemble (rank-agreement, unanimous sign) ---")
    z_ensemble = pd.DataFrame(index=oos_idx, columns=prices.columns, dtype=float)
    for date in oos_idx:
        s = ensemble_signal({
            "coint":  z_coint_oos.loc[date] if date in z_coint_oos.index else pd.Series(dtype=float),
            "pca":    z_pca_oos.loc[date] if date in z_pca_oos.index else pd.Series(dtype=float),
            "factor": z_factor_oos.loc[date] if date in z_factor_oos.index else pd.Series(dtype=float),
        }, require_unanimous_sign=True)
        z_ensemble.loc[date, s.index] = s.values

    res_ens = backtest_signal(prices_oos, z_ensemble,
                              entry_z=0.3, exit_z=0.05,   # ensemble scale differs
                              max_positions=8, holding_cap_days=20,
                              transaction_cost_bps=5.0)
    results["Ensemble (unanimous)"] = res_ens
    print(f"{'Ensemble':<22s}  Sharpe={res_ens.sharpe:+.2f}  "
          f"MaxDD={res_ens.max_drawdown*100:+.1f}%  "
          f"HitRate={res_ens.hit_rate*100:.1f}%  "
          f"Trades={res_ens.n_trades}")

    # ---- 7. Plot ----
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, r in results.items():
            ax.plot(r.equity, label=name, linewidth=1.5)
        ax.set_title("Equity curves — cross-asset stat arb (out-of-sample)")
        ax.set_ylabel("Equity ($)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("equity_curves.png", dpi=150)
        print("Saved: equity_curves.png")

        # Signal correlation between frameworks
        def flatten(df): return df.stack().dropna()
        sig_df = pd.DataFrame({
            "coint":  flatten(z_coint_oos),
            "pca":    flatten(z_pca_oos),
            "factor": flatten(z_factor_oos),
        }).dropna()
        print("\nCorrelation between framework signals:")
        print(sig_df.corr().round(3))
    except ImportError:
        print("(matplotlib not installed — skipping plots)")

    # ---- 8. Factor exposures from the interpretable framework ----
    fm = FactorResidualModel(lookback=252).fit(prices)
    if fm.exposures_ is not None:
        print("\nTop factor exposures (from Framework 3, final window):")
        print(fm.exposures_.abs().sum(axis=1).sort_values(ascending=False).head(10))
        fm.exposures_.to_csv("factor_exposures.csv")
        print("Saved: factor_exposures.csv")


if __name__ == "__main__":
    main()
