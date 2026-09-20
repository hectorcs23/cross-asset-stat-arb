"""
run_robust_analysis.py
======================
Análisis robusto del modelo cross-asset stat arb:

  1. Sube el bar del ensemble: grid de umbrales de entrada
  2. Sensibilidad al costo transaccional
  3. Bootstrap por bloques del Sharpe (intervalo de confianza)
  4. Walk-forward estricto (ventanas rodantes de 1 año)
  5. Ensemble SIN cointegración (solo PCA + Factor)
  6. Visualizaciones listas para presentación

Cache: los z-scores se guardan en cache/z_scores.pkl tras la primera corrida.
Para reajustar desde cero, borra la carpeta cache/ o pasa --refit.

Uso:
    python run_robust_analysis.py
    python run_robust_analysis.py --refit

Salidas en figs/:
    01_equity_curves.png         — curvas de capital + drawdown
    02_rolling_sharpe.png        — Sharpe rodante 252 días
    03_signal_correlation.png    — correlación PCA vs Factor
    04_sensitivity_heatmap.png   — Sharpe(z_entry, tcost)
    05_bootstrap_sharpe.png      — distribución bootstrap del Sharpe
    06_walk_forward.png          — Sharpe por ventana walk-forward
    07_factor_exposures.png      — heatmap de betas factoriales
    08_pnl_distribution.png      — distribución de PnL diario
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cross_asset_statarb import (
    CointegrationModel, FactorResidualModel, PCAResidualModel,
    backtest_signal, build_all_z_scores, ensemble_signal,
    fetch_prices, flat_universe,
)


# =====================================================================
# Config
# =====================================================================

CACHE_DIR = Path("cache")
FIGS_DIR = Path("figs")
CACHE_DIR.mkdir(exist_ok=True)
FIGS_DIR.mkdir(exist_ok=True)

# Matplotlib style — limpio para presentación académica
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "gray",
    "figure.titlesize": 13,
    "figure.titleweight": "bold",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})

COLORS = {
    "coint":    "#1f77b4",
    "pca":      "#ff7f0e",
    "factor":   "#2ca02c",
    "ensemble": "#d62728",
}
LABELS = {
    "coint":    "Cointegración",
    "pca":      "PCA / Avellaneda-Lee",
    "factor":   "Residuos factoriales",
    "ensemble": "Ensemble (PCA + Factor)",
}


# =====================================================================
# 1. Cacheo de fits
# =====================================================================

def load_or_fit(force_refit: bool = False) -> dict:
    """
    Ajusta los tres frameworks una sola vez y cachea los z-scores.
    Siguientes corridas cargan el cache en <1 segundo.
    """
    cache_file = CACHE_DIR / "z_scores.pkl"

    if cache_file.exists() and not force_refit:
        print(f"[cache] cargando z-scores desde {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("[fit] ajustando frameworks desde cero (puede tomar varios minutos)...")
    tickers = flat_universe()
    prices = fetch_prices(tickers, start="2015-01-01")
    print(f"  panel de precios: {prices.shape[0]} días × {prices.shape[1]} tickers")

    print("  → Framework 1: cointegración...")
    z_coint = build_all_z_scores(CointegrationModel, prices,
                                 fdr_alpha=0.05, half_life_bounds=(2, 60))
    print("  → Framework 2: PCA + RMT...")
    z_pca = build_all_z_scores(PCAResidualModel, prices,
                               lookback=252, refit_every=21,
                               half_life_bounds=(2, 60))
    print("  → Framework 3: factores económicos...")
    z_factor = build_all_z_scores(FactorResidualModel, prices,
                                  lookback=252, half_life_bounds=(2, 60))
    print("  → exposiciones factoriales finales...")
    fm = FactorResidualModel(lookback=252).fit(prices)

    data = {
        "prices":    prices,
        "z_coint":   z_coint,
        "z_pca":     z_pca,
        "z_factor":  z_factor,
        "exposures": fm.exposures_,
    }
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
    print(f"[cache] guardado en {cache_file}")
    return data


# =====================================================================
# 2. Ensemble PCA + Factor (sin cointegración)
# =====================================================================

def build_ensemble_z(z_pca: pd.DataFrame,
                     z_factor: pd.DataFrame,
                     dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Ensemble solo con PCA + Factor — cointegración dio 0 pares, no aporta."""
    cols = z_pca.columns.union(z_factor.columns)
    z_ens = pd.DataFrame(index=dates, columns=cols, dtype=float)
    for date in dates:
        sigs = {}
        if date in z_pca.index:
            sigs["pca"] = z_pca.loc[date]
        if date in z_factor.index:
            sigs["factor"] = z_factor.loc[date]
        s = ensemble_signal(sigs, require_unanimous_sign=True)
        z_ens.loc[date, s.index] = s.values
    return z_ens


# =====================================================================
# 3. Plots
# =====================================================================

def plot_equity_and_dd(results: dict, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8),
                                    sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    for name, r in results.items():
        ax1.plot(r.equity, label=LABELS.get(name, name),
                 color=COLORS.get(name, "gray"), linewidth=1.8)
    ax1.set_title("Curva de capital — estrategias cross-asset (out-of-sample)")
    ax1.set_ylabel("Capital (USD, escala log)")
    ax1.set_yscale("log")
    ax1.legend(loc="upper left")

    for name, r in results.items():
        cummax = r.equity.cummax()
        dd = (r.equity - cummax) / cummax * 100
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.25,
                         color=COLORS.get(name, "gray"))
        ax2.plot(dd, color=COLORS.get(name, "gray"), linewidth=1.0)
    ax2.set_title("Drawdown desde el pico (%)")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Fecha")
    ax2.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved: {out_path}")


def plot_rolling_sharpe(returns_dict: dict, out_path: Path, window: int = 252) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, r in returns_dict.items():
        mu = r.rolling(window).mean()
        sd = r.rolling(window).std()
        rs = (mu / sd) * np.sqrt(252)
        ax.plot(rs, label=LABELS.get(name, name),
                color=COLORS.get(name, "gray"), linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(1, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_title(f"Sharpe rodante de {window} días")
    ax.set_ylabel("Sharpe anualizado")
    ax.set_xlabel("Fecha")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved: {out_path}")


def plot_signal_correlation(z_dict: dict, out_path: Path) -> None:
    dfs = {}
    for name, z in z_dict.items():
        flat = z.stack().dropna()
        if len(flat) > 100:
            dfs[LABELS.get(name, name)] = flat
    if len(dfs) < 2:
        print("  [skip] no hay suficientes frameworks no vacíos para correlación")
        return
    combined = pd.DataFrame(dfs).dropna()
    corr = combined.corr()

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=30, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black", fontweight="bold")
    ax.set_title(f"Correlación entre señales de frameworks\n(n={len(combined):,} obs.)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved: {out_path}")


def plot_sensitivity(sharpe_grid: np.ndarray,
                     trade_grid: np.ndarray,
                     entry_zs: list, tcosts: list,
                     out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    vmax = max(abs(np.nanmin(sharpe_grid)), abs(np.nanmax(sharpe_grid)))
    if vmax == 0 or np.isnan(vmax):
        vmax = 1.0

    im1 = ax1.imshow(sharpe_grid, cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
    ax1.set_xticks(range(len(tcosts)))
    ax1.set_xticklabels([f"{t:.0f} bps" for t in tcosts])
    ax1.set_yticks(range(len(entry_zs)))
    ax1.set_yticklabels([f"{z:.1f}" for z in entry_zs])
    ax1.set_xlabel("Costo transaccional (por trade)")
    ax1.set_ylabel(r"Umbral de entrada $z_{\mathrm{in}}$")
    ax1.set_title("Sharpe ratio (anualizado)")
    for i in range(len(entry_zs)):
        for j in range(len(tcosts)):
            v = sharpe_grid[i, j]
            ax1.text(j, i, f"{v:.2f}",
                     ha="center", va="center",
                     color="white" if abs(v) > vmax * 0.55 else "black",
                     fontweight="bold", fontsize=9)
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(trade_grid, cmap="viridis", origin="lower", aspect="auto")
    ax2.set_xticks(range(len(tcosts)))
    ax2.set_xticklabels([f"{t:.0f} bps" for t in tcosts])
    ax2.set_yticks(range(len(entry_zs)))
    ax2.set_yticklabels([f"{z:.1f}" for z in entry_zs])
    ax2.set_xlabel("Costo transaccional (por trade)")
    ax2.set_ylabel(r"Umbral de entrada $z_{\mathrm{in}}$")
    ax2.set_title("Número de trades ejecutados")
    for i in range(len(entry_zs)):
        for j in range(len(tcosts)):
            ax2.text(j, i, f"{int(trade_grid[i, j])}",
                     ha="center", va="center", color="white",
                     fontweight="bold", fontsize=9)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle("Análisis de sensibilidad del ensemble (PCA + Factor)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved: {out_path}")


def plot_bootstrap(sharpe_samples: np.ndarray,
                   original_sharpe: float,
                   out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(sharpe_samples, bins=60, color="steelblue",
            edgecolor="white", alpha=0.85)

    p05, p50, p95 = np.percentile(sharpe_samples, [5, 50, 95])
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(original_sharpe, color="red", linewidth=2.2,
               label=f"Sharpe muestral: {original_sharpe:+.2f}")
    ax.axvline(p05, color="gray", linewidth=1.5, linestyle="--",
               label=f"P5:  {p05:+.2f}")
    ax.axvline(p95, color="gray", linewidth=1.5, linestyle="--",
               label=f"P95: {p95:+.2f}")

    frac_positive = (sharpe_samples > 0).mean() * 100
    ax.set_xlabel("Sharpe ratio (anualizado)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Bootstrap por bloques del Sharpe — ensemble PCA+Factor "
                 f"(n={len(sharpe_samples):,})\n"
                 f"IC 90%: [{p05:+.2f}, {p95:+.2f}]   |   "
                 f"P(Sharpe > 0) = {frac_positive:.1f}%")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved: {out_path}")


def plot_walk_forward(wf_df: pd.DataFrame, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    x = np.arange(len(wf_df))
    labels = [f"{s}\n→ {e}" for s, e in zip(wf_df["start"], wf_df["end"])]

    colors = ["#2ca02c" if s > 0 else "#d62728" for s in wf_df["sharpe"]]
    ax1.bar(x, wf_df["sharpe"], color=colors, alpha=0.75,
            edgecolor="black", linewidth=0.6)
    ax1.axhline(0, color="black", linewidth=0.8)
    mean_s = wf_df["sharpe"].mean()
    ax1.axhline(mean_s, color="steelblue", linewidth=1.5, linestyle="--",
                label=f"Media: {mean_s:+.2f}")
    ax1.set_ylabel("Sharpe (anualizado)")
    ax1.set_title("Walk-forward: Sharpe por ventana de 1 año (ensemble PCA+Factor)")
    ax1.legend()
    for i, v in enumerate(wf_df["sharpe"]):
        ax1.text(i, v + 0.05 if v >= 0 else v - 0.1, f"{v:+.2f}",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=9)

    ax2.bar(x, wf_df["n_trades"], color="steelblue", alpha=0.7,
            edgecolor="black", linewidth=0.6)
    ax2.set_ylabel("Trades")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved: {out_path}")


def plot_factor_exposures(exposures: pd.DataFrame,
                          out_path: Path, top_n: int = 20) -> None:
    if exposures is None or exposures.empty:
        print("  [skip] exposiciones vacías")
        return
    total = exposures.abs().sum(axis=1).sort_values(ascending=False)
    top_assets = total.head(top_n).index
    E = exposures.loc[top_assets]

    vmax = E.abs().max().max()
    fig, ax = plt.subplots(figsize=(8, 7.5))
    im = ax.imshow(E.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(E.columns)))
    ax.set_xticklabels(E.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(E.index)))
    ax.set_yticklabels(E.index)
    for i in range(len(E.index)):
        for j in range(len(E.columns)):
            v = E.iloc[i, j]
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > vmax * 0.55 else "black")
    ax.set_title(f"Exposiciones factoriales β — top {top_n} activos\n"
                 f"(regresión rodante 252 días, Framework 3)")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Activo")
    plt.colorbar(im, ax=ax, shrink=0.8, label="β (coeficiente de regresión)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved: {out_path}")


def plot_pnl_distribution(returns_dict: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(returns_dict), figsize=(4.5 * len(returns_dict), 4),
                             sharey=True)
    if len(returns_dict) == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, returns_dict.items()):
        rr = r[r != 0].values * 100  # en porcentaje
        ax.hist(rr, bins=60, color=COLORS.get(name, "gray"),
                edgecolor="white", alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline(rr.mean(), color="red", linewidth=1.5,
                   linestyle="--", label=f"Media: {rr.mean():+.3f}%")
        ax.set_title(LABELS.get(name, name))
        ax.set_xlabel("Retorno diario (%)")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("Frecuencia")
    fig.suptitle("Distribución de retornos diarios", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved: {out_path}")


# =====================================================================
# 4. Análisis de sensibilidad
# =====================================================================

def sensitivity_analysis(prices_oos, z_ensemble,
                         entry_zs: list, tcosts: list) -> tuple:
    """Grid (z_entry × tcost) → (Sharpe, n_trades)."""
    sharpe_grid = np.full((len(entry_zs), len(tcosts)), np.nan)
    trade_grid = np.zeros_like(sharpe_grid)
    for i, ez in enumerate(entry_zs):
        for j, tc in enumerate(tcosts):
            res = backtest_signal(prices_oos, z_ensemble,
                                  entry_z=ez, exit_z=0.05,
                                  max_positions=8, holding_cap_days=20,
                                  transaction_cost_bps=tc)
            sharpe_grid[i, j] = res.sharpe
            trade_grid[i, j] = res.n_trades
    return sharpe_grid, trade_grid


# =====================================================================
# 5. Bootstrap por bloques
# =====================================================================

def block_bootstrap_sharpe(returns: pd.Series,
                           block_size: int = 21,
                           n_boot: int = 10000,
                           seed: int = 42) -> np.ndarray:
    """
    Bootstrap por bloques móviles (preserva autocorrelación).
    Implementación vectorizada vía indexación avanzada.
    """
    rng = np.random.default_rng(seed)
    r = returns.dropna().values
    n = len(r)
    if n < block_size * 2:
        return np.array([])

    n_possible = n - block_size + 1
    n_blocks = int(np.ceil(n / block_size))
    offsets = np.arange(block_size)

    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n_possible, size=n_blocks)
        idx = (starts[:, None] + offsets[None, :]).ravel()[:n]
        sampled = r[idx]
        mu, sd = sampled.mean(), sampled.std(ddof=1)
        sharpes[b] = (mu / sd) * np.sqrt(252) if sd > 0 else 0.0
    return sharpes


# =====================================================================
# 6. Walk-forward
# =====================================================================

def walk_forward_analysis(data: dict,
                          train_years: int = 3,
                          test_years: int = 1,
                          entry_z: float = 0.3,
                          tcost_bps: float = 5.0) -> pd.DataFrame:
    """
    Walk-forward sobre z-scores pre-computados (ya son walk-forward-safe:
    los modelos rodantes solo usan datos estrictamente anteriores a cada fecha).
    """
    prices = data["prices"]
    z_pca = data["z_pca"]
    z_factor = data["z_factor"]

    start = prices.index[0] + pd.DateOffset(years=train_years)
    end = prices.index[-1]

    rows = []
    current = start
    while current + pd.DateOffset(years=test_years) <= end:
        win_start = current
        win_end = current + pd.DateOffset(years=test_years)
        dates = prices.loc[win_start:win_end].index
        z_ens = build_ensemble_z(z_pca, z_factor, dates)
        res = backtest_signal(prices.loc[win_start:win_end], z_ens,
                              entry_z=entry_z, exit_z=0.05,
                              max_positions=8, holding_cap_days=20,
                              transaction_cost_bps=tcost_bps)
        rows.append({
            "start":    win_start.strftime("%Y-%m"),
            "end":      win_end.strftime("%Y-%m"),
            "sharpe":   res.sharpe,
            "max_dd":   res.max_drawdown * 100,
            "n_trades": res.n_trades,
            "hit_rate": (res.hit_rate * 100) if not np.isnan(res.hit_rate) else 0.0,
        })
        current += pd.DateOffset(years=test_years)
    return pd.DataFrame(rows)


# =====================================================================
# 7. Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit", action="store_true",
                        help="Forzar reajuste en lugar de usar cache")
    args = parser.parse_args()

    print("=" * 65)
    print("ANÁLISIS ROBUSTO — Modelo cross-asset statistical arbitrage")
    print("=" * 65)

    # -------- Datos --------
    data = load_or_fit(force_refit=args.refit)
    prices   = data["prices"]
    z_pca    = data["z_pca"]
    z_factor = data["z_factor"]

    # Partición OOS limpia: 60% in-sample, 40% OOS, sin solapamiento
    split = int(len(prices) * 0.60)
    oos_start = prices.index[split]
    oos_end = prices.index[-1]
    oos_dates = prices.loc[oos_start:oos_end].index
    prices_oos = prices.loc[oos_start:oos_end]

    print(f"\nPeríodo in-sample:  {prices.index[0].date()} → {prices.index[split-1].date()}")
    print(f"Período OOS:        {oos_start.date()} → {oos_end.date()}"
          f"  ({len(oos_dates)} días, {len(oos_dates)/252:.1f} años)")

    # -------- Ensemble PCA + Factor (sin cointegración) --------
    print("\nConstruyendo ensemble (PCA + Factor, sin cointegración)...")
    z_ens = build_ensemble_z(z_pca, z_factor, oos_dates)

    # -------- Backtest base --------
    print("\nBacktest con parámetros default:")
    strategies = {
        "pca":      (z_pca.reindex(oos_dates),    1.5, 0.25, 10),
        "factor":   (z_factor.reindex(oos_dates), 1.5, 0.25, 10),
        "ensemble": (z_ens,                       0.3, 0.05, 8),
    }
    results = {}
    for name, (z, ez, xz, mp) in strategies.items():
        res = backtest_signal(prices_oos, z,
                              entry_z=ez, exit_z=xz,
                              max_positions=mp, holding_cap_days=20,
                              transaction_cost_bps=5.0)
        results[name] = res
        print(f"  {LABELS[name]:<32s}  Sharpe={res.sharpe:+.2f}  "
              f"MaxDD={res.max_drawdown*100:+6.1f}%  "
              f"Hit={res.hit_rate*100 if not np.isnan(res.hit_rate) else 0:5.1f}%  "
              f"N={res.n_trades}")

    # -------- Plot 1: equity + DD --------
    print("\n[plots] generando figuras...")
    plot_equity_and_dd(results, FIGS_DIR / "01_equity_curves.png")

    # -------- Plot 2: Sharpe rodante --------
    returns_dict = {k: v.returns for k, v in results.items()}
    plot_rolling_sharpe(returns_dict, FIGS_DIR / "02_rolling_sharpe.png")

    # -------- Plot 3: correlación de señales --------
    plot_signal_correlation({
        "pca":    z_pca.reindex(oos_dates),
        "factor": z_factor.reindex(oos_dates),
    }, FIGS_DIR / "03_signal_correlation.png")

    # -------- Plot 4: sensibilidad --------
    print("  análisis de sensibilidad...")
    entry_zs = [0.2, 0.3, 0.5, 0.7, 1.0]
    tcosts   = [0, 5, 10, 15, 20]
    sg, tg = sensitivity_analysis(prices_oos, z_ens, entry_zs, tcosts)
    plot_sensitivity(sg, tg, entry_zs, tcosts,
                     FIGS_DIR / "04_sensitivity_heatmap.png")

    # -------- Plot 5: bootstrap --------
    print("  bootstrap por bloques (n=10,000)...")
    boot = block_bootstrap_sharpe(results["ensemble"].returns,
                                  block_size=21, n_boot=10000, seed=42)
    plot_bootstrap(boot, results["ensemble"].sharpe,
                   FIGS_DIR / "05_bootstrap_sharpe.png")

    # -------- Plot 6: walk-forward --------
    print("  walk-forward (ventanas de 1 año)...")
    wf = walk_forward_analysis(data, train_years=3, test_years=1)
    print("\nResultados walk-forward:")
    print(wf.to_string(index=False))
    wf.to_csv("walk_forward_results.csv", index=False)
    plot_walk_forward(wf, FIGS_DIR / "06_walk_forward.png")

    # -------- Plot 7: exposiciones factoriales --------
    plot_factor_exposures(data["exposures"],
                          FIGS_DIR / "07_factor_exposures.png", top_n=20)

    # -------- Plot 8: distribución de PnL --------
    plot_pnl_distribution(returns_dict, FIGS_DIR / "08_pnl_distribution.png")

    # -------- Resumen ejecutivo --------
    p05, p50, p95 = np.percentile(boot, [5, 50, 95])
    frac_pos = (boot > 0).mean() * 100
    wf_positive = (wf["sharpe"] > 0).sum()

    print("\n" + "=" * 65)
    print("RESUMEN EJECUTIVO")
    print("=" * 65)
    print(f"Ensemble (PCA + Factor), período OOS {oos_start.date()} → {oos_end.date()}")
    print(f"  Sharpe muestral:       {results['ensemble'].sharpe:+.2f}")
    print(f"  Max drawdown:          {results['ensemble'].max_drawdown*100:+.1f}%")
    print(f"  Hit rate:              {results['ensemble'].hit_rate*100:.1f}%")
    print(f"  Número de trades:      {results['ensemble'].n_trades}")
    print(f"\nInferencia estadística (bootstrap por bloques, n=10,000):")
    print(f"  IC 90% del Sharpe:     [{p05:+.2f}, {p95:+.2f}]")
    print(f"  P(Sharpe > 0):         {frac_pos:.1f}%")
    print(f"\nEstabilidad temporal (walk-forward, {len(wf)} ventanas de 1 año):")
    print(f"  Ventanas con Sharpe>0: {wf_positive}/{len(wf)}")
    print(f"  Sharpe medio:          {wf['sharpe'].mean():+.2f}")
    print(f"  Sharpe std:            {wf['sharpe'].std():.2f}")
    print("\n" + "=" * 65)
    print(f"Figuras guardadas en: {FIGS_DIR.resolve()}")
    print(f"Cache en:             {CACHE_DIR.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
