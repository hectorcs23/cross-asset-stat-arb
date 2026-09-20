"""
paper_trader.py  (v2)
=====================
Runner diario de paper trading — v2 con correcciones post-mortem:

  * TIF cambiado de OPG → DAY  (OPG tuvo 75% de expiración en ETFs medianos)
  * State reconciliado CONTRA ALPACA al inicio de cada corrida
    (en vez de asumir que lo que se submite se llena)
  * Logging explícito de fills cuando están disponibles
  * Espera entre submits para evitar rate-limit spikes

USO:
  python paper_trader.py --dry-run
  python paper_trader.py --live
  python paper_trader.py --panic --live
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, AssetStatus, AssetClass
    from alpaca.common.exceptions import APIError
except ImportError:
    sys.exit("Falta instalar alpaca-py: pip install alpaca-py")

from cross_asset_statarb import (
    flat_universe, fetch_prices,
    PCAResidualModel, FactorResidualModel,
    build_all_z_scores, ensemble_signal,
)

# =====================================================================
# Configuración
# =====================================================================

NY_TZ = ZoneInfo("America/New_York")

STATE_FILE = Path("state.json")
LOG_FILE = Path("paper_trader.log")

ENTRY_Z = 0.3
EXIT_Z = 0.05
MAX_POSITIONS = 8
MAX_HOLDING_DAYS = 20
GROSS_LEVERAGE = 1.0

# Neutralidad: |suma de lados| permitida en el libro.
# 8 posiciones con MAX_NET_POSITIONS=2 => como mucho 5 largos / 3 cortos.
# Antes esto no existía y el libro derivó a 8 largos / 0 cortos.
MAX_NET_POSITIONS = 2

# Tras un cierre por timeout, no reabrir el mismo símbolo por N días.
# Sin esto, un timeout con señal aún fuerte cerraba y reabría en la misma
# corrida => rechazo 40310000 (wash trade) y round-trip gratis.
TIMEOUT_COOLDOWN_DAYS = 3

MAX_POSITION_PCT = 0.20
MAX_DAILY_ORDERS = 30
SIGNAL_SANITY_LIMIT = 10.0
MIN_TRADE_NOTIONAL = 100.0

SUBMIT_DELAY_SEC = 0.5


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-7s  %(message)s"
    logging.basicConfig(
        level=level, format=fmt,
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# =====================================================================
# Estado persistente
# =====================================================================

@dataclass
class OpenPosition:
    symbol: str
    side: int
    entry_date: str
    entry_price: float
    target_weight: float

    def days_held(self, today: datetime) -> int:
        ed = datetime.fromisoformat(self.entry_date)
        return (today.date() - ed.date()).days


@dataclass
class TraderState:
    positions: Dict[str, dict] = field(default_factory=dict)
    last_run: str = ""
    run_history: List[dict] = field(default_factory=list)
    # symbol -> fecha ISO hasta la cual no se puede reabrir (post-timeout)
    cooldown: Dict[str, str] = field(default_factory=dict)

    def purge_cooldown(self, today: datetime) -> None:
        expired = [s for s, until in self.cooldown.items()
                   if datetime.fromisoformat(until).date() <= today.date()]
        for s in expired:
            self.cooldown.pop(s, None)

    def in_cooldown(self, sym: str, today: datetime) -> bool:
        until = self.cooldown.get(sym)
        if until is None:
            return False
        return datetime.fromisoformat(until).date() > today.date()

    @classmethod
    def load(cls, path: Path) -> "TraderState":
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, default=str)


# =====================================================================
# Reconciliación state <-> Alpaca al inicio
# =====================================================================

def reconcile_state_with_alpaca(state: TraderState,
                                 trading: TradingClient,
                                 today: datetime) -> TraderState:
    """
    La verdad es Alpaca, no state.json.
    """
    alpaca_positions = {p.symbol: p for p in trading.get_all_positions()}
    state_symbols = set(state.positions.keys())
    alpaca_symbols = set(alpaca_positions.keys())

    phantom = state_symbols - alpaca_symbols
    orphan = alpaca_symbols - state_symbols

    if phantom:
        logging.warning(f"Removiendo {len(phantom)} phantoms de state: "
                        f"{sorted(phantom)}")
        for sym in phantom:
            state.positions.pop(sym, None)

    if orphan:
        logging.warning(f"Adoptando {len(orphan)} huérfanos: {sorted(orphan)}")
        for sym in orphan:
            p = alpaca_positions[sym]
            qty = float(p.qty)
            state.positions[sym] = asdict(OpenPosition(
                symbol=sym,
                side=+1 if qty > 0 else -1,
                entry_date=today.isoformat(),
                entry_price=float(p.avg_entry_price),
                target_weight=1.0 / MAX_POSITIONS,
            ))

    # Corregir direcciones si discrepan
    for sym in state_symbols & alpaca_symbols:
        state_side = state.positions[sym]["side"]
        alpaca_qty = float(alpaca_positions[sym].qty)
        alpaca_side = +1 if alpaca_qty > 0 else -1
        if state_side != alpaca_side:
            logging.error(f"  {sym}: state side={state_side:+d} vs "
                          f"Alpaca qty={alpaca_qty:+.0f} — corrigiendo")
            state.positions[sym]["side"] = alpaca_side

    logging.info(f"Reconciliación: {len(state.positions)} posiciones, "
                 f"coincidiendo con Alpaca")
    return state


def audit_exposure(trading: TradingClient, portfolio_value: float) -> None:
    """
    Reconciliar por símbolo y signo no detecta ni un libro direccional ni una
    posición sobredimensionada. Esto sí: el libro corrió 3 meses 100% largo
    sin una sola línea de log que lo dijera.
    """
    positions = trading.get_all_positions()
    if not positions:
        logging.info("Exposición: sin posiciones")
        return

    longs = [p for p in positions if float(p.qty) > 0]
    shorts = [p for p in positions if float(p.qty) < 0]
    long_mv = sum(abs(float(p.market_value)) for p in longs)
    short_mv = sum(abs(float(p.market_value)) for p in shorts)
    gross = long_mv + short_mv
    net = long_mv - short_mv

    logging.info(
        f"Exposición: {len(longs)}L/{len(shorts)}S   "
        f"bruta ${gross:,.0f} ({gross / portfolio_value:.0%})   "
        f"neta ${net:+,.0f} ({net / portfolio_value:+.0%})"
    )

    if gross > 0 and abs(net) / gross > 0.5:
        logging.warning(
            f"LIBRO DIRECCIONAL: neto/bruto = {abs(net) / gross:.0%}. "
            f"La estrategia asume neutralidad — revisar asignación de slots."
        )

    # Cap por posición agregada. compute_order_qty limita cada ORDEN, no la
    # posición acumulada, así que dos corridas el mismo día pueden duplicar
    # tamaño sin que ningún breaker se entere.
    for p in positions:
        pct = abs(float(p.market_value)) / portfolio_value
        if pct > MAX_POSITION_PCT:
            logging.error(
                f"  {p.symbol}: {pct:.1%} del portafolio > tope "
                f"{MAX_POSITION_PCT:.0%} (qty={float(p.qty):+.0f}). "
                f"Posible doble ejecución — revisar manualmente."
            )


# =====================================================================
# Cómputo de señales
# =====================================================================

def compute_latest_z_ensemble(prices: pd.DataFrame) -> pd.Series:
    logging.info("Ajustando PCA / Avellaneda-Lee...")
    z_pca = build_all_z_scores(
        PCAResidualModel, prices,
        lookback=252, refit_every=21, half_life_bounds=(2, 60),
    )
    logging.info("Ajustando residuos factoriales...")
    z_factor = build_all_z_scores(
        FactorResidualModel, prices,
        lookback=252, half_life_bounds=(2, 60),
    )

    latest = prices.index[-1]
    sigs = {}
    if latest in z_pca.index:
        sigs["pca"] = z_pca.loc[latest].dropna()
    if latest in z_factor.index:
        sigs["factor"] = z_factor.loc[latest].dropna()

    if len(sigs) < 2:
        logging.error("No hay señales suficientes")
        return pd.Series(dtype=float)

    return ensemble_signal(sigs, require_unanimous_sign=True)


def sanity_check_signals(z: pd.Series) -> bool:
    if z.empty:
        logging.warning("Ensemble vacío, skip.")
        return False
    if z.abs().max() > SIGNAL_SANITY_LIMIT:
        logging.error(f"|z| máximo = {z.abs().max():.2f}. MODEL BROKEN.")
        return False
    logging.info(f"Señal OK: {len(z)} activos, "
                 f"|z| rango [{z.abs().min():.2f}, {z.abs().max():.2f}]")
    return True


# =====================================================================
# Lógica de entrada/salida
# =====================================================================

def decide_actions(z_ens: pd.Series,
                   state: TraderState,
                   today: datetime,
                   shortable: Optional[set] = None
                   ) -> tuple[List[str], List[tuple[str, int]]]:
    """
    Cierres por |z| < EXIT_Z o timeout, aperturas por ranking de |z|.

    Cambios vs v2 (los tres bugs que dejaron el libro 8 largos / 0 cortos):

      1. Antes los largos consumían los 8 slots en un bucle propio ANTES de
         mirar los cortos. Con >=8 largos sobre umbral, los cortos nunca
         entraban. Ahora se rankea por |z| y se emparejan lados.
      2. El filtro `shortable` se aplicaba DESPUÉS de asignar slots, así que
         un corto rechazado quemaba el slot. Ahora entra aquí.
      3. Un timeout con señal aún fuerte reabría el mismo símbolo en la
         misma corrida. Ahora hay cooldown.
    """
    state.purge_cooldown(today)

    to_close: List[str] = []
    timed_out: List[str] = []
    for sym, pos_dict in state.positions.items():
        pos = OpenPosition(**pos_dict)
        z_now = z_ens.get(sym, np.nan)
        days = pos.days_held(today)

        if pd.notna(z_now) and abs(z_now) < EXIT_Z:
            logging.info(f"  CLOSE {sym}: |z|={abs(z_now):.2f} < {EXIT_Z}")
            to_close.append(sym)
        elif days >= MAX_HOLDING_DAYS:
            logging.info(f"  CLOSE {sym}: {days} días >= {MAX_HOLDING_DAYS} (timeout)")
            to_close.append(sym)
            timed_out.append(sym)

    # El timeout existe para salir de spreads que no convergieron. Si el
    # símbolo pudiera reabrirse mañana la salida es cosmética, así que se
    # bloquea explícitamente.
    for sym in timed_out:
        until = today + timedelta(days=TIMEOUT_COOLDOWN_DAYS)
        state.cooldown[sym] = until.isoformat()
        logging.info(f"    cooldown {sym} hasta {until.date()}")

    held = set(state.positions.keys()) - set(to_close)
    slots = MAX_POSITIONS - len(held)
    net = sum(state.positions[s]["side"] for s in held)

    if slots <= 0:
        logging.info(f"  Sin slots libres (libro={len(held)}, neto={net:+d})")
        return to_close, []

    # Candidatos: |z| sobre umbral, no en cartera, no en cooldown,
    # y shortable si el lado es corto.
    cand: List[tuple[str, int, float]] = []
    skipped_short = []
    for sym, z in z_ens.items():
        if abs(z) <= ENTRY_Z or sym in held or sym in to_close:
            continue
        if state.in_cooldown(sym, today):
            continue
        side = +1 if z < 0 else -1
        if side == -1 and shortable is not None and sym not in shortable:
            skipped_short.append(sym)
            continue
        cand.append((sym, side, abs(z)))

    if skipped_short:
        logging.warning(f"  No shortable, descartados: {sorted(skipped_short)}")

    longs = sorted([c for c in cand if c[1] == +1], key=lambda c: -c[2])
    shorts = sorted([c for c in cand if c[1] == -1], key=lambda c: -c[2])
    logging.info(f"  Candidatos: {len(longs)} largos, {len(shorts)} cortos, "
                 f"{slots} slots, neto actual {net:+d}")

    to_open: List[tuple[str, int]] = []

    def take(entry) -> None:
        nonlocal slots, net
        sym, side, _ = entry
        to_open.append((sym, side))
        slots -= 1
        net += side

    # Pase 1 — emparejar largo/corto mientras haya de ambos lados.
    # Esto es lo que mantiene el libro cerca de neutral.
    while slots >= 2 and longs and shorts:
        take(longs.pop(0))
        take(shorts.pop(0))

    # Pase 2 — rellenar por |z| sin romper el tope de exposición neta.
    rest = sorted(longs + shorts, key=lambda c: -c[2])
    for entry in rest:
        if slots <= 0:
            break
        if abs(net + entry[1]) > MAX_NET_POSITIONS:
            logging.info(f"    SKIP {entry[0]}: rompería neto "
                         f"({net:+d} -> {net + entry[1]:+d}, "
                         f"tope ±{MAX_NET_POSITIONS})")
            continue
        take(entry)

    n_long = sum(1 for _, s in to_open if s == +1)
    n_short = len(to_open) - n_long
    logging.info(f"  Aperturas: {n_long} largos, {n_short} cortos, "
                 f"neto proyectado {net:+d}")

    return to_close, to_open


# =====================================================================
# Ejecución en Alpaca
# =====================================================================

def get_tradable_shortable(trading: TradingClient) -> tuple[set[str], set[str]]:
    req = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
    assets = trading.get_all_assets(req)
    tradable = {a.symbol for a in assets if a.tradable}
    shortable = {a.symbol for a in assets if a.tradable and a.shortable}
    return tradable, shortable


def submit_order(trading: TradingClient,
                 symbol: str, qty: int, side: OrderSide,
                 dry_run: bool = True) -> Optional[dict]:
    """CAMBIO CLAVE: TIF.DAY en lugar de TIF.OPG → fills más confiables."""
    if qty == 0:
        return None

    req = MarketOrderRequest(
        symbol=symbol,
        qty=abs(qty),
        side=side,
        time_in_force=TimeInForce.DAY,   # cambiado de OPG a DAY
    )

    tag = "DRY-RUN " if dry_run else ""
    logging.info(f"    {tag}ORDER  {side.value:4s}  {symbol:6s}  qty={abs(qty)}")

    if dry_run:
        return {"symbol": symbol, "qty": abs(qty), "side": side.value, "dry_run": True}

    try:
        order = trading.submit_order(order_data=req)
        time.sleep(SUBMIT_DELAY_SEC)
        return {"symbol": symbol, "qty": abs(qty), "side": side.value,
                "order_id": str(order.id), "status": order.status.value}
    except APIError as e:
        logging.error(f"    FALLO orden {symbol}: {e}")
        return {"symbol": symbol, "qty": abs(qty), "side": side.value, "error": str(e)}


def compute_order_qty(side: int, target_weight: float,
                      portfolio_value: float, last_price: float) -> int:
    if last_price <= 0 or target_weight <= 0:
        return 0
    notional = portfolio_value * target_weight
    notional = min(notional, portfolio_value * MAX_POSITION_PCT)
    if notional < MIN_TRADE_NOTIONAL:
        return 0
    shares = int(notional / last_price)
    return shares * side


# =====================================================================
# Main
# =====================================================================

def run(dry_run: bool = True, panic: bool = False):
    key = os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not sec:
        sys.exit("Faltan env vars APCA_API_KEY_ID / APCA_API_SECRET_KEY")

    trading = TradingClient(key, sec, paper=True)
    account = trading.get_account()
    portfolio_value = float(account.portfolio_value)
    cash = float(account.cash)

    logging.info("=" * 60)
    logging.info(f"Paper trader v2 — {'DRY-RUN' if dry_run else 'LIVE'}")
    logging.info(f"Portfolio value: ${portfolio_value:,.2f}   Cash: ${cash:,.2f}")

    if panic:
        logging.warning("!!! PANIC MODE !!!")
        if not dry_run:
            trading.cancel_orders()
            trading.close_all_positions(cancel_orders=True)
        return

    clock = trading.get_clock()
    logging.info(f"Market is_open={clock.is_open}   next_open={clock.next_open}")

    today = datetime.now(NY_TZ)

    state = TraderState.load(STATE_FILE)
    logging.info(f"\nEstado antes de reconciliar: {len(state.positions)} posiciones")
    state = reconcile_state_with_alpaca(state, trading, today)
    audit_exposure(trading, portfolio_value)

    logging.info("\nDescargando precios históricos...")
    prices = fetch_prices(flat_universe(), start="2015-01-01")
    age_days = (today.date() - prices.index[-1].date()).days
    if age_days > 5:
        logging.error(f"Datos con {age_days} días de antigüedad — abort.")
        return

    z_ens = compute_latest_z_ensemble(prices)
    if not sanity_check_signals(z_ens):
        return

    logging.info(f"\nTop 10 señales:")
    top = z_ens.reindex(z_ens.abs().sort_values(ascending=False).index).head(10)
    for sym, z in top.items():
        direction = "LONG " if z < 0 else "SHORT"
        logging.info(f"  {direction} {sym:6s}  z={z:+.2f}")

    tradable, shortable = get_tradable_shortable(trading)
    missing = set(z_ens.index) - tradable
    if missing:
        logging.warning(f"No tradables en Alpaca: {missing}")
        z_ens = z_ens.drop(index=list(missing))

    logging.info("\nDecisiones:")
    # shortable entra aquí: filtrar después desperdiciaba slots.
    to_close, to_open = decide_actions(z_ens, state, today, shortable=shortable)

    n_orders = len(to_close) + len(to_open)
    logging.info(f"\nTotal: {len(to_close)} cierres + {len(to_open)} aperturas = "
                 f"{n_orders} órdenes")
    if n_orders > MAX_DAILY_ORDERS:
        logging.error(f"Órdenes ({n_orders}) > breaker {MAX_DAILY_ORDERS}. Abort.")
        return
    if n_orders == 0:
        logging.info("Nada que hacer.")
        state.last_run = today.isoformat()
        state.save(STATE_FILE)
        return

    alpaca_positions = {p.symbol: float(p.qty) for p in trading.get_all_positions()}

    logging.info("\n--- Cerrando ---")
    for sym in to_close:
        current_qty = alpaca_positions.get(sym, 0)
        if current_qty == 0:
            state.positions.pop(sym, None)
            continue
        side = OrderSide.SELL if current_qty > 0 else OrderSide.BUY
        submit_order(trading, sym, int(abs(current_qty)), side, dry_run)
        # NOTA: NO removemos de state aquí. La reconciliación de
        # la próxima corrida se encargará si el cierre se ejecutó.

    logging.info("\n--- Abriendo ---")
    each_weight = GROSS_LEVERAGE / max(MAX_POSITIONS, 1)
    for sym, side in to_open:
        last_price = prices[sym].iloc[-1]
        qty = compute_order_qty(side, each_weight, portfolio_value, last_price)
        if qty == 0:
            logging.warning(f"  SKIP {sym}: tamaño $ insuficiente")
            continue
        order_side = OrderSide.BUY if side == +1 else OrderSide.SELL
        submit_order(trading, sym, abs(qty), order_side, dry_run)
        # NOTA: NO agregamos a state aquí. La reconciliación de
        # la próxima corrida se encargará si la apertura se ejecutó.

    state.last_run = today.isoformat()
    state.run_history.append({
        "date": today.strftime("%Y-%m-%d"),
        "portfolio_value": portfolio_value,
        "closes": len(to_close),
        "opens": len(to_open),
        "dry_run": dry_run,
    })
    state.run_history = state.run_history[-90:]
    state.save(STATE_FILE)

    logging.info("\n" + "=" * 60)
    logging.info("Completado. State se reconciliará en la próxima corrida.")
    logging.info("=" * 60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true")
    p.add_argument("--panic", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    setup_logging(verbose=args.verbose)
    run(dry_run=not args.live, panic=args.panic)


if __name__ == "__main__":
    main()
