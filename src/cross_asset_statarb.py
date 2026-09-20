"""
cross_asset_statarb.py
======================

Cross-Asset Statistical Arbitrage — three mispricing frameworks compared
on a macro universe (equity indexes, commodities, bonds, FX, inflation).

Frameworks
----------
1. CointegrationModel          — Engle-Granger pairwise + Ornstein-Uhlenbeck,
                                 with Benjamini-Hochberg FDR correction over
                                 all C(N, 2) pairs.
2. PCAResidualModel            — Avellaneda-Lee style: standardize returns,
                                 RMT-denoise the correlation matrix, project
                                 onto signal eigenportfolios, cumulate and
                                 OU-calibrate the per-asset residual.
3. FactorResidualModel         — Pre-specified economic factors (market,
                                 dollar, rates, credit, inflation, oil,
                                 gold). Rolling regression; trade residual.

All three return a standardized z-score per asset per day. The ensemble
combines them by rank-agreement (requiring directional consensus filters
out most false positives).

Mathematical choices
--------------------
* Marchenko-Pastur upper edge λ₊ = (1 + √(N/T))² separates signal from
  noise eigenvalues. Only eigenvalues above λ₊ are retained as factors.
* Exact OU discretization: ε_{t+1} = a + b·ε_t + η with
      θ = -ln(b),   τ_½ = ln(2)/θ,   σ_eq = √(Var(η)/(1-b²))
* Cointegration p-values corrected via Benjamini-Hochberg (FDR).
* Half-life filter [2, 60] days — strictly enforced.

Design
------
All model classes inherit from `MispricingModel` and implement `.fit(prices)`
and `.signal(date)`. This makes swapping and ensembling trivial.

Authors: Santiago Mejía Torres and Hector Campbell Salas, April 2026.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =====================================================================
# 1. Universe definition
# =====================================================================

UNIVERSE: Dict[str, List[str]] = {
    # Equity indexes — global beta
    "equity_indexes": ["SPY", "QQQ", "IWM", "DIA", "EEM", "EFA"],
    # Sector ETFs — intra-equity dispersion
    "equity_sectors": ["XLE", "XLF", "XLK", "XLI", "XLP",
                       "XLY", "XLV", "XLU", "XLB"],
    # Fixed income — duration & credit
    "bonds": ["TLT", "IEF", "SHY", "LQD", "HYG", "TIP"],
    # Precious metals
    "precious": ["GLD", "SLV"],
    # Industrial metals
    "industrial_metals": ["CPER"],        # copper
    # Energy
    "energy": ["USO", "UNG"],
    # Agriculture
    "agriculture": ["DBA", "CORN", "WEAT"],
    # FX proxies
    "fx": ["UUP", "FXE", "FXY"],
}

# Pre-specified factors for Framework 3 (economically meaningful)
FACTOR_PROXIES: Dict[str, str] = {
    "market":    "SPY",
    "dollar":    "UUP",
    "rates":     "TLT",   # long-duration treasuries
    "credit":    "HYG",   # high yield — contains credit spread info
    "inflation": "TIP",   # TIPS — breakeven / inflation expectations
    "oil":       "USO",
    "gold":      "GLD",
}


def flat_universe() -> List[str]:
    """All tickers, deduplicated and sorted."""
    return sorted({t for group in UNIVERSE.values() for t in group})


# =====================================================================
# 2. Data fetching
# =====================================================================

def fetch_prices(tickers: List[str],
                 start: str = "2018-01-01",
                 end: Optional[str] = None) -> pd.DataFrame:
    """
    Download adjusted close prices. Requires yfinance.

    Returns
    -------
    DataFrame of shape (T, N), indexed by date, columns = tickers.
    Tickers with excessive missing data are dropped with a warning.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("pip install yfinance") from e

    raw = yf.download(tickers, start=start, end=end,
                      progress=False, auto_adjust=True)
    prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    prices = prices.dropna(how="all")

    # Drop tickers with >5% missing observations
    missing = prices.isna().mean()
    too_sparse = missing[missing > 0.05].index.tolist()
    if too_sparse:
        print(f"[fetch_prices] dropping sparse tickers: {too_sparse}")
        prices = prices.drop(columns=too_sparse)

    return prices.ffill().dropna()


# =====================================================================
# 3. Ornstein-Uhlenbeck calibration
# =====================================================================

@dataclass
class OUParams:
    """Fitted parameters for dε = -θ(ε - μ)dt + σdW, Δt = 1."""
    mu: float          # long-run mean
    theta: float       # mean-reversion speed
    sigma: float       # instantaneous vol
    half_life: float   # ln(2) / θ
    sigma_eq: float    # steady-state std = σ/√(2θ)
    r2: float          # regression fit quality
    n_obs: int


def calibrate_ou(series: np.ndarray) -> Optional[OUParams]:
    """
    Fit OU parameters via exact discretization.

    ε_{t+1} = a + b·ε_t + η_t,   with  b = e^{-θ},  a = μ(1-b)

    Returns None if mean-reversion is not detected (b ≥ 1 or b ≤ 0).
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return None

    y_t = x[:-1]
    y_tp1 = x[1:]

    # OLS: y_tp1 = a + b*y_t
    X = np.column_stack([np.ones_like(y_t), y_t])
    try:
        coef, *_ = np.linalg.lstsq(X, y_tp1, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b = coef
    resid = y_tp1 - X @ coef

    # b must be in (0, 1) for stable mean-reverting OU
    if not (0 < b < 1):
        return None

    theta = -np.log(b)
    mu = a / (1 - b)
    var_eta = np.var(resid, ddof=1)
    # Variance of η relates to σ via: Var(η) = σ² (1-b²)/(2θ)
    sigma2 = var_eta * 2 * theta / (1 - b**2)
    sigma = np.sqrt(max(sigma2, 1e-12))
    sigma_eq = np.sqrt(var_eta / (1 - b**2))
    half_life = np.log(2) / theta

    ss_tot = np.var(y_tp1, ddof=1) * len(y_tp1)
    ss_res = np.sum(resid**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return OUParams(mu=mu, theta=theta, sigma=sigma,
                    half_life=half_life, sigma_eq=sigma_eq,
                    r2=r2, n_obs=len(y_t))


# =====================================================================
# 4. Random Matrix Theory — Marchenko-Pastur denoising
# =====================================================================

def marchenko_pastur_edges(n: int, t: int) -> Tuple[float, float]:
    """Upper/lower edges of the MP distribution for correlation noise."""
    q = n / t
    lam_minus = (1 - np.sqrt(q))**2
    lam_plus = (1 + np.sqrt(q))**2
    return lam_minus, lam_plus


def rmt_denoise_correlation(returns: pd.DataFrame) -> Tuple[np.ndarray, int]:
    """
    Clean a correlation matrix by keeping only eigenvalues above the MP bound.
    Noise eigenvalues are replaced by the average of the bulk (trace-preserving).

    Returns
    -------
    C_clean : (N, N) denoised correlation matrix
    k_signal : number of signal eigenvalues retained
    """
    R = returns.values
    T, N = R.shape
    # Standardize columns
    Z = (R - R.mean(axis=0)) / (R.std(axis=0, ddof=1) + 1e-12)
    C = Z.T @ Z / T

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(C)

    _, lam_plus = marchenko_pastur_edges(N, T)
    signal_mask = eigvals > lam_plus
    k_signal = int(signal_mask.sum())

    # Replace noise eigenvalues with their mean (preserves trace = N)
    noise_vals = eigvals[~signal_mask]
    if len(noise_vals) > 0:
        replacement = noise_vals.mean()
        eigvals_clean = np.where(signal_mask, eigvals, replacement)
    else:
        eigvals_clean = eigvals.copy()

    C_clean = (eigvecs * eigvals_clean) @ eigvecs.T
    # Force unit diagonal (it drifts slightly from eigenvalue replacement)
    d = np.sqrt(np.diag(C_clean))
    C_clean = C_clean / np.outer(d, d)
    return C_clean, k_signal


# =====================================================================
# 5. Benjamini-Hochberg FDR correction
# =====================================================================

def bh_fdr(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg step-up procedure.
    Returns boolean mask of hypotheses accepted at FDR level α.
    """
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = alpha * np.arange(1, n + 1) / n
    passing = ranked <= thresholds
    if not passing.any():
        return np.zeros_like(p, dtype=bool)
    max_k = np.max(np.where(passing)[0])
    cutoff = ranked[max_k]
    return p <= cutoff


# =====================================================================
# 6. Abstract base
# =====================================================================

class MispricingModel(ABC):
    """Common interface for all three frameworks."""

    @abstractmethod
    def fit(self, prices: pd.DataFrame) -> "MispricingModel":
        ...

    @abstractmethod
    def signal(self, date: pd.Timestamp) -> pd.Series:
        """Return z-score per asset on `date` (positive = rich, negative = cheap)."""
        ...


# =====================================================================
# 7. Framework 1 — Cointegration
# =====================================================================

@dataclass
class CointPair:
    asset_y: str
    asset_x: str
    beta: float
    ou: OUParams
    pvalue: float


class CointegrationModel(MispricingModel):
    """
    Engle-Granger pairwise cointegration, scanned across all C(N,2) pairs,
    FDR-corrected, filtered by OU half-life.

    Fitted pairs generate an asset-level signal by averaging the z-score
    contribution from every accepted pair that contains that asset.

    For a pair (Y, X) with spread s = log(P_y) - β·log(P_x) - μ:
      * z < 0 → spread is below mean → Y cheap relative to X → buy Y / sell X
    We report z per side: z(Y) = -z(spread)/2, z(X) = +z(spread)/2.
    """

    def __init__(self,
                 fdr_alpha: float = 0.05,
                 half_life_bounds: Tuple[float, float] = (2, 60),
                 min_r2: float = 0.0):
        self.fdr_alpha = fdr_alpha
        self.half_life_bounds = half_life_bounds
        self.min_r2 = min_r2
        self.pairs_: List[CointPair] = []
        self.prices_: Optional[pd.DataFrame] = None

    def fit(self, prices: pd.DataFrame) -> "CointegrationModel":
        from statsmodels.tsa.stattools import coint

        self.prices_ = prices.copy()
        log_p = np.log(prices)

        tickers = prices.columns.tolist()
        candidates: List[Tuple[str, str, float, float, np.ndarray]] = []
        pvalues: List[float] = []

        # Scan all unordered pairs
        for i, y in enumerate(tickers):
            for x in tickers[i + 1:]:
                py = log_p[y].values
                px = log_p[x].values
                try:
                    _, pval, _ = coint(py, px, trend="c")
                except Exception:
                    continue
                # hedge ratio via OLS
                X = np.column_stack([np.ones_like(px), px])
                coef, *_ = np.linalg.lstsq(X, py, rcond=None)
                beta = coef[1]
                spread = py - coef[0] - beta * px
                candidates.append((y, x, beta, pval, spread))
                pvalues.append(pval)

        if not candidates:
            return self

        # FDR-correct p-values
        accepted_mask = bh_fdr(np.array(pvalues), alpha=self.fdr_alpha)

        for (y, x, beta, pval, spread), accept in zip(candidates, accepted_mask):
            if not accept:
                continue
            ou = calibrate_ou(spread)
            if ou is None:
                continue
            if not (self.half_life_bounds[0] <= ou.half_life <= self.half_life_bounds[1]):
                continue
            if ou.r2 < self.min_r2:
                continue
            self.pairs_.append(CointPair(asset_y=y, asset_x=x,
                                         beta=beta, ou=ou, pvalue=pval))

        print(f"[Cointegration] {len(self.pairs_)} pairs survived "
              f"FDR @ α={self.fdr_alpha}, half-life in {self.half_life_bounds}")
        return self

    def signal(self, date: pd.Timestamp) -> pd.Series:
        if self.prices_ is None or date not in self.prices_.index:
            return pd.Series(dtype=float)

        log_p = np.log(self.prices_.loc[date])
        signals: Dict[str, List[float]] = {t: [] for t in self.prices_.columns}

        for p in self.pairs_:
            if p.asset_y not in log_p or p.asset_x not in log_p:
                continue
            spread_now = log_p[p.asset_y] - p.beta * log_p[p.asset_x]
            z = (spread_now - p.ou.mu) / p.ou.sigma_eq
            # High spread → Y rich, X cheap → z(Y) positive, z(X) negative
            signals[p.asset_y].append(z)
            signals[p.asset_x].append(-z)

        return pd.Series({t: np.mean(v) if v else np.nan
                          for t, v in signals.items()})


# =====================================================================
# 8. Framework 2 — PCA residuals (Avellaneda-Lee, RMT-denoised)
# =====================================================================

class PCAResidualModel(MispricingModel):
    """
    Avellaneda-Lee style statistical arbitrage.

    1. Standardize log-returns.
    2. Compute correlation, denoise with Marchenko-Pastur.
    3. Eigendecompose; keep k signal eigenvectors as "eigenportfolios".
    4. For each asset, rolling-window regression:
         R_i(t) = α_i + Σ β_ik · F_k(t) + ε_i(t)
       where F_k(t) is the k-th eigenportfolio return.
    5. Cumulate residual X_i(t) = Σ ε_i(s), fit OU.
    6. Signal = -z-score of X_i (large positive X → asset is rich; buy when z negative).
    """

    def __init__(self,
                 lookback: int = 252,
                 refit_every: int = 21,
                 half_life_bounds: Tuple[float, float] = (2, 60)):
        self.lookback = lookback
        self.refit_every = refit_every
        self.half_life_bounds = half_life_bounds
        self.returns_: Optional[pd.DataFrame] = None
        self.z_scores_: Optional[pd.DataFrame] = None

    def fit(self, prices: pd.DataFrame) -> "PCAResidualModel":
        self.returns_ = np.log(prices).diff().dropna()
        dates = self.returns_.index
        tickers = self.returns_.columns.tolist()

        z_out = pd.DataFrame(index=dates, columns=tickers, dtype=float)

        # Rolling window — recompute eigenportfolios every `refit_every` days
        last_fit_idx = -10**9
        eigvecs = None
        eigvals = None
        k_signal = 0

        for i, date in enumerate(dates):
            if i < self.lookback:
                continue
            window = self.returns_.iloc[i - self.lookback:i]

            # Refit factor basis periodically
            if i - last_fit_idx >= self.refit_every:
                C_clean, k_signal = rmt_denoise_correlation(window)
                eigvals_full, eigvecs_full = np.linalg.eigh(C_clean)
                # Top k eigenvectors
                if k_signal == 0:
                    k_signal = 1  # always keep at least the market
                eigvecs = eigvecs_full[:, -k_signal:]
                eigvals = eigvals_full[-k_signal:]
                last_fit_idx = i

            # Eigenportfolio returns: F = R @ eigvecs / σ  (normalized)
            sigmas = window.std(ddof=1).values + 1e-12
            R_std = (window.values - window.mean().values) / sigmas
            F = R_std @ eigvecs   # shape (lookback, k_signal)

            # Per-asset regression R_i on F → get residuals in the window
            X = np.column_stack([np.ones(self.lookback), F])
            XtX_inv = np.linalg.pinv(X.T @ X)
            betas = XtX_inv @ X.T @ R_std   # shape (1+k, N)
            residuals = R_std - X @ betas   # shape (lookback, N)

            # Cumulate residuals
            cum_res = np.cumsum(residuals, axis=0)

            # Per-asset OU on the cumulated residual
            for j, tkr in enumerate(tickers):
                ou = calibrate_ou(cum_res[:, j])
                if ou is None:
                    continue
                if not (self.half_life_bounds[0] <= ou.half_life <= self.half_life_bounds[1]):
                    continue
                last_val = cum_res[-1, j]
                z = (last_val - ou.mu) / ou.sigma_eq
                z_out.loc[date, tkr] = z

        self.z_scores_ = z_out
        print(f"[PCA/Avellaneda-Lee] avg signal eigenvalues retained: "
              f"{k_signal} of {len(tickers)}")
        return self

    def signal(self, date: pd.Timestamp) -> pd.Series:
        if self.z_scores_ is None or date not in self.z_scores_.index:
            return pd.Series(dtype=float)
        return self.z_scores_.loc[date].dropna()


# =====================================================================
# 9. Framework 3 — Cross-sectional factor residuals
# =====================================================================

class FactorResidualModel(MispricingModel):
    """
    Pre-specified economic factors (market, dollar, rates, credit,
    inflation, oil, gold). Rolling regression per asset; trade the residual.

    This is the most interpretable framework. β_i,oil tells you the asset's
    oil exposure — useful for risk decomposition far beyond the signal itself.
    """

    def __init__(self,
                 factor_tickers: Optional[Dict[str, str]] = None,
                 lookback: int = 252,
                 half_life_bounds: Tuple[float, float] = (2, 60)):
        self.factor_tickers = factor_tickers or FACTOR_PROXIES
        self.lookback = lookback
        self.half_life_bounds = half_life_bounds
        self.z_scores_: Optional[pd.DataFrame] = None
        self.exposures_: Optional[pd.DataFrame] = None

    def fit(self, prices: pd.DataFrame) -> "FactorResidualModel":
        factors = [t for t in self.factor_tickers.values() if t in prices.columns]
        if not factors:
            raise ValueError("None of the factor tickers are in the price panel.")

        returns = np.log(prices).diff().dropna()
        factor_ret = returns[factors]
        # Non-factor tickers are the traded universe
        asset_ret = returns.drop(columns=[c for c in factors if c in returns.columns],
                                 errors="ignore")
        # Include factors too — a factor can also be mispriced vs the rest
        asset_ret = returns

        dates = returns.index
        tickers = asset_ret.columns.tolist()
        z_out = pd.DataFrame(index=dates, columns=tickers, dtype=float)

        # Store last-fit exposures for interpretation
        final_betas: Dict[str, Dict[str, float]] = {}

        for i in range(self.lookback, len(dates)):
            window_ret = asset_ret.iloc[i - self.lookback:i]
            window_fac = factor_ret.iloc[i - self.lookback:i]

            F = np.column_stack([np.ones(self.lookback), window_fac.values])
            FtF_inv = np.linalg.pinv(F.T @ F)

            for j, tkr in enumerate(tickers):
                # Avoid regressing a factor against itself trivially
                if tkr in factors:
                    # Drop own column from regressors
                    own_idx = factors.index(tkr)
                    keep = [k for k in range(len(factors)) if k != own_idx]
                    Fj = np.column_stack([np.ones(self.lookback),
                                          window_fac.values[:, keep]])
                    FjtFj_inv = np.linalg.pinv(Fj.T @ Fj)
                    coef = FjtFj_inv @ Fj.T @ window_ret[tkr].values
                    resid = window_ret[tkr].values - Fj @ coef
                else:
                    coef = FtF_inv @ F.T @ window_ret[tkr].values
                    resid = window_ret[tkr].values - F @ coef

                cum_res = np.cumsum(resid)
                ou = calibrate_ou(cum_res)
                if ou is None:
                    continue
                if not (self.half_life_bounds[0] <= ou.half_life <= self.half_life_bounds[1]):
                    continue
                z = (cum_res[-1] - ou.mu) / ou.sigma_eq
                z_out.loc[dates[i], tkr] = z

                # Store final exposures
                if i == len(dates) - 1:
                    if tkr in factors:
                        own_idx = factors.index(tkr)
                        keep = [k for k in range(len(factors)) if k != own_idx]
                        names = [factors[k] for k in keep]
                    else:
                        names = factors
                    final_betas[tkr] = dict(zip(names, coef[1:]))

        self.z_scores_ = z_out
        self.exposures_ = pd.DataFrame(final_betas).T
        return self

    def signal(self, date: pd.Timestamp) -> pd.Series:
        if self.z_scores_ is None or date not in self.z_scores_.index:
            return pd.Series(dtype=float)
        return self.z_scores_.loc[date].dropna()


# =====================================================================
# 10. Ensemble by rank agreement
# =====================================================================

def ensemble_signal(signals: Dict[str, pd.Series],
                    require_unanimous_sign: bool = True) -> pd.Series:
    """
    Combine per-framework z-scores by rank-mean.

    If `require_unanimous_sign` is True, drop assets where the three models
    disagree on whether the asset is rich or cheap. This is the single most
    powerful noise filter in the system: each individual framework has
    plenty of false positives, but agreement across all three is rare by
    chance.
    """
    df = pd.DataFrame(signals).dropna(how="all")
    if df.empty:
        return pd.Series(dtype=float)

    if require_unanimous_sign:
        sign_agree = np.sign(df).apply(
            lambda row: (row.dropna() > 0).all() or (row.dropna() < 0).all(),
            axis=1
        )
        df = df[sign_agree]

    # Rank-based combination (robust to scale differences)
    ranks = df.rank(axis=0, pct=True)
    composite = ranks.mean(axis=1) - 0.5  # center around 0
    composite *= 2 * df.abs().mean(axis=1).fillna(0)  # rescale by magnitude
    return composite.dropna()


# =====================================================================
# 11. Backtester
# =====================================================================

@dataclass
class BacktestResult:
    returns: pd.Series
    positions: pd.DataFrame
    equity: pd.Series
    n_trades: int
    sharpe: float
    max_drawdown: float
    hit_rate: float


def backtest_signal(prices: pd.DataFrame,
                    z_scores: pd.DataFrame,
                    entry_z: float = 1.5,
                    exit_z: float = 0.25,
                    max_positions: int = 10,
                    holding_cap_days: int = 20,
                    transaction_cost_bps: float = 5.0,
                    capital: float = 100_000.0,
                    gross_leverage: float = 1.0) -> BacktestResult:
    """
    Simple threshold backtester:
      * Enter long when z < -entry_z, short when z > +entry_z
      * Exit when |z| < exit_z OR holding > holding_cap_days
      * Execute at next-day open (we use next close as proxy; modify to open
        if you have open data). Signals are lagged by one day.

    Position sizing: equal $ across current open positions, capped at
    `max_positions` on each side. Gross leverage controlled by `gross_leverage`.
    """
    dates = z_scores.index.intersection(prices.index)
    z = z_scores.reindex(dates)
    P = prices.reindex(dates)
    R = np.log(P).diff()

    positions = pd.DataFrame(0.0, index=dates, columns=P.columns)
    active: Dict[str, Dict] = {}  # ticker -> {'side': +/-1, 'entry_date': ..., 'days': n}

    for t_idx in range(1, len(dates)):
        today = dates[t_idx]
        yesterday = dates[t_idx - 1]
        z_yest = z.loc[yesterday].dropna()

        # --- Exit logic ---
        to_close = []
        for tkr, info in active.items():
            info["days"] += 1
            z_now = z.loc[yesterday, tkr] if tkr in z.columns else np.nan
            if (pd.notna(z_now) and abs(z_now) < exit_z) or info["days"] >= holding_cap_days:
                to_close.append(tkr)
        for tkr in to_close:
            active.pop(tkr)

        # --- Entry logic ---
        longs = z_yest[z_yest < -entry_z].sort_values().index.tolist()
        shorts = z_yest[z_yest > +entry_z].sort_values(ascending=False).index.tolist()

        slots_avail = max_positions - len(active)
        for tkr in longs:
            if slots_avail <= 0:
                break
            if tkr in active:
                continue
            active[tkr] = {"side": +1, "days": 0}
            slots_avail -= 1
        for tkr in shorts:
            if slots_avail <= 0:
                break
            if tkr in active:
                continue
            active[tkr] = {"side": -1, "days": 0}
            slots_avail -= 1

        # --- Weights: equal $ across active positions ---
        if active:
            w_each = gross_leverage / len(active)
            for tkr, info in active.items():
                positions.loc[today, tkr] = info["side"] * w_each

    # --- P&L calculation ---
    # Portfolio return at t = sum(weights_{t-1} * log-return_t) - txn cost on changes
    weights_lag = positions.shift(1).fillna(0)
    gross_ret = (weights_lag * R).sum(axis=1)
    turnover = (positions - positions.shift(1).fillna(0)).abs().sum(axis=1)
    txn = turnover * (transaction_cost_bps / 1e4)
    port_ret = gross_ret - txn

    equity = capital * np.exp(port_ret.cumsum())

    # Metrics
    ann_factor = np.sqrt(252)
    sharpe = (port_ret.mean() / port_ret.std() * ann_factor) if port_ret.std() > 0 else 0.0
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax
    max_dd = dd.min()
    n_trades = int(turnover.gt(0).sum())
    hit_rate = float((port_ret > 0).sum() / (port_ret != 0).sum()) if (port_ret != 0).any() else np.nan

    return BacktestResult(returns=port_ret, positions=positions, equity=equity,
                          n_trades=n_trades, sharpe=sharpe, max_drawdown=max_dd,
                          hit_rate=hit_rate)


# =====================================================================
# 12. Orchestration helper
# =====================================================================

def build_all_z_scores(model_class,
                       prices: pd.DataFrame,
                       **kwargs) -> pd.DataFrame:
    """Fit a model and expose a full z-score panel for backtesting."""
    model = model_class(**kwargs).fit(prices)

    if hasattr(model, "z_scores_") and model.z_scores_ is not None:
        return model.z_scores_.copy()

    # For the cointegration model we must compute per-date signals
    dates = prices.index
    out = pd.DataFrame(index=dates, columns=prices.columns, dtype=float)
    for date in dates:
        s = model.signal(date)
        out.loc[date, s.index] = s.values
    return out
