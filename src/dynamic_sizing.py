"""
dynamic_sizing.py
=================

Drop-in dynamic position sizing for paper_trader.py.

Replaces the implicit "each_weight = GROSS_LEVERAGE / MAX_POSITIONS" logic
with adaptive sizing by signal strength + inverse volatility + portfolio
vol targeting.

Chosen from the sizing_comparison.py backtest (OOS 2017–2026):
    Sharpe +0.44 vs baseline +0.43, MaxDD -16% vs -34%, Calmar +0.23 vs +0.17.
Same risk-adjusted return as equal-weight, but with roughly half the drawdown.

INTEGRATION
-----------
In paper_trader.py, after you have `z_ens` and `prices` and the lists
`to_open` (new entries) and existing `state.positions` (carried-over),
call `compute_target_weights(...)` to get a signed weight per symbol,
then translate that to shares.

Minimal patch to paper_trader.py — see the --- PATCH --- block at the
bottom of this file.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Config (override by import site if desired)
# ---------------------------------------------------------------------
GROSS_LEVERAGE       = 1.0
MAX_POSITION_PCT     = 0.20
VOL_LOOKBACK_DAYS    = 60
TARGET_VOL_ANN       = 0.10      # 10% annualized — match your risk appetite
ANN_FACTOR           = 252
MIN_VOL_FLOOR        = 0.03      # don't let a low-vol name (SHY) soak up all capital
LEVERAGE_SCALE_CAP   = (0.25, 2.0)   # bound on vol-target rescale factor


# ---------------------------------------------------------------------
# Rolling vol — the sizing engine needs this per ticker, annualized
# ---------------------------------------------------------------------
def estimate_vols(prices: pd.DataFrame,
                  window: int = VOL_LOOKBACK_DAYS) -> pd.Series:
    """Latest annualized log-return vol per ticker (floored at MIN_VOL_FLOOR)."""
    r = np.log(prices).diff().tail(window + 5)
    sigma = r.std() * np.sqrt(ANN_FACTOR)
    return sigma.clip(lower=MIN_VOL_FLOOR)


# ---------------------------------------------------------------------
# Core sizing: w_i ∝ |z_i| / σ_i, then scale gross to hit vol target
# ---------------------------------------------------------------------
def compute_target_weights(
    tickers_sides: List[Tuple[str, int]],   # [(symbol, +1/-1), ...] — the full book after decisions
    z_scores: pd.Series,                     # latest ensemble z, indexed by ticker
    vols: pd.Series,                         # annualized vol, indexed by ticker
    recent_portfolio_returns: Optional[pd.Series] = None,
    gross_leverage: float = GROSS_LEVERAGE,
    max_position_pct: float = MAX_POSITION_PCT,
    target_vol: float = TARGET_VOL_ANN,
) -> Dict[str, float]:
    """
    Joint weight computation for all positions (new + carried).

    Steps
    -----
    1. Raw score: r_i = |z_i| / σ_i
    2. Normalize so Σ|w| = gross_leverage
    3. Estimate current portfolio vol (from recent returns, or Σ|w_i|·σ_i fallback)
    4. Rescale by target_vol / estimated_vol, bounded to [0.25×, 2×]
    5. Enforce per-position cap |w_i| ≤ max_position_pct (post-scale)

    Returns
    -------
    Dict {symbol: signed weight}, where weight is a fraction of portfolio value.
    """
    if not tickers_sides:
        return {}

    # Step 1: raw |z| / σ
    raw = {}
    for sym, side in tickers_sides:
        z = z_scores.get(sym, np.nan)
        v = vols.get(sym, np.nan)
        if pd.isna(z) or pd.isna(v) or v <= 0:
            continue
        raw[sym] = side * (abs(z) / v)

    if not raw:
        return {}

    # Step 2: normalize to gross_leverage
    s = sum(abs(v) for v in raw.values())
    if s == 0:
        return {sym: 0.0 for sym in raw}
    w = {sym: x * gross_leverage / s for sym, x in raw.items()}

    # Step 3-4: vol targeting
    if recent_portfolio_returns is not None and len(recent_portfolio_returns.dropna()) >= 20:
        cur_vol = recent_portfolio_returns.dropna().std() * np.sqrt(ANN_FACTOR)
    else:
        # Conservative proxy assuming zero correlation across positions
        cur_vol = float(np.sqrt(sum((w[sym] * vols.get(sym, 0.2)) ** 2 for sym in w)))

    if cur_vol > 0:
        scale = target_vol / cur_vol
        scale = float(np.clip(scale, *LEVERAGE_SCALE_CAP))
        w = {sym: x * scale for sym, x in w.items()}

    # Step 5: per-position cap
    w = {sym: np.sign(x) * min(abs(x), max_position_pct) for sym, x in w.items()}
    return w


# ---------------------------------------------------------------------
# Shares conversion (replaces compute_order_qty for the new-entry side)
# ---------------------------------------------------------------------
def weight_to_shares(weight: float,
                     portfolio_value: float,
                     last_price: float,
                     min_notional: float = 100.0) -> int:
    """Signed share count. weight is signed (+long / -short)."""
    if last_price <= 0 or weight == 0:
        return 0
    notional = portfolio_value * weight
    if abs(notional) < min_notional:
        return 0
    shares = int(notional / last_price)   # truncation preserves sign
    return shares


# ---------------------------------------------------------------------
# --- PATCH for paper_trader.py ---
# ---------------------------------------------------------------------
# Replace the "--- Abriendo ---" block with the following:
#
#     from dynamic_sizing import compute_target_weights, estimate_vols, weight_to_shares
#
#     # Estimate recent portfolio realized vol from run_history (optional but better)
#     recent_rets = None  # TODO: plug in realized returns from trade log if you track them
#
#     # Everything currently open + everything we want to open
#     current_book: list[tuple[str, int]] = [
#         (sym, p["side"]) for sym, p in state.positions.items()
#     ] + list(to_open)
#
#     # Per-ticker vol from the same `prices` panel the signal was computed on
#     vols = estimate_vols(prices, window=60)
#
#     weights = compute_target_weights(
#         tickers_sides=current_book,
#         z_scores=z_ens,
#         vols=vols,
#         recent_portfolio_returns=recent_rets,
#         target_vol=0.10,          # 10% annualized — tune to your risk appetite
#     )
#
#     logging.info("\n--- Abriendo (dynamic sizing) ---")
#     for sym, side in to_open:
#         w = weights.get(sym, 0.0)
#         last_price = float(prices[sym].iloc[-1])
#         qty = weight_to_shares(w, portfolio_value, last_price)
#         if qty == 0:
#             logging.warning(f"  SKIP {sym}: weight={w:+.3f} -> qty=0")
#             continue
#         order_side = OrderSide.BUY if qty > 0 else OrderSide.SELL
#         logging.info(f"  {sym:6s} w={w:+.3f} -> {qty:+d} shares @ ~${last_price:.2f}")
#         submit_order(trading, sym, abs(qty), order_side, dry_run)
#
# NOTE: For existing open positions that were sized under the old equal-weight
# scheme, you have two options:
#   (a) Let them age out naturally (MAX_HOLDING_DAYS=20) and only apply dynamic
#       sizing to new entries. Simplest, minimum disruption.
#   (b) Resize them toward the new target each day (issue delta orders).
#       More "correct" but doubles your turnover during the transition.
# I recommend (a) unless your state has >4 open positions — then (b) is worth it.
