"""
test_decide_actions.py
======================
Regresión para los tres bugs de asignación de slots (2026-07-25).

Contexto: el bot corrió del 2026-04-19 al 2026-07-25 con 8 posiciones largas
y CERO cortos abiertos en todo el periodo. No fue mala suerte de señal —
`decide_actions` asignaba los 8 slots a largos en un bucle propio antes de
mirar los cortos.

Los tests usan los z-scores reales del log del 2026-07-25 21:02.

Uso:
    python test_decide_actions.py

No toca Alpaca ni la red: stubea las dependencias antes de importar.
"""
from __future__ import annotations

import json
import logging
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


# ---------------------------------------------------------------------
# Stubs — evitan que importar paper_trader exija alpaca-py / yfinance
# ---------------------------------------------------------------------

def _install_stubs() -> None:
    def mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m

    class E:
        def __init__(self, v):
            self.value = v

    mod("alpaca")
    mod("alpaca.trading")
    mod("alpaca.common")
    mod("alpaca.trading.client", TradingClient=object)
    mod("alpaca.trading.requests", MarketOrderRequest=object, GetAssetsRequest=object)
    mod("alpaca.trading.enums",
        OrderSide=types.SimpleNamespace(SELL=E("sell"), BUY=E("buy")),
        TimeInForce=types.SimpleNamespace(DAY=E("day"), OPG=E("opg")),
        AssetStatus=types.SimpleNamespace(ACTIVE="active"),
        AssetClass=types.SimpleNamespace(US_EQUITY="us_equity"))
    mod("alpaca.common.exceptions", APIError=Exception)
    mod("cross_asset_statarb",
        flat_universe=lambda: [],
        fetch_prices=lambda *a, **k: None,
        PCAResidualModel=object,
        FactorResidualModel=object,
        build_all_z_scores=lambda *a, **k: None,
        ensemble_signal=lambda *a, **k: None)


_install_stubs()

import pandas as pd  # noqa: E402
import paper_trader as PT  # noqa: E402

logging.disable(logging.CRITICAL)

NY = ZoneInfo("America/New_York")
TODAY = datetime(2026, 7, 25, 21, 0, tzinfo=NY)

# Top 10 del log real del 2026-07-25, más largos débiles del resto del
# universo — que es exactamente lo que llenaba los slots en producción.
REAL_Z = {
    "WEAT": +2.96, "LQD": -2.09, "CPER": +1.81, "XLY": -1.09, "GLD": -0.94,
    "SHY": -0.76, "UUP": -0.66, "FXY": +0.64, "XLK": +0.59, "SLV": -0.59,
    "IEF": -0.55, "TIP": -0.48, "XLE": -0.44, "TLT": -0.41, "HYG": -0.36,
    "XLP": -0.33, "USO": +0.35, "DBA": +0.31,
}
Z = pd.Series(REAL_Z)

# WEAT/CPER/CORN/UNG son ETPs de commodities: típicamente no shortables.
SHORTABLE = {s for s in REAL_Z if s not in ("WEAT", "CPER", "CORN", "UNG")}


def make_state(*syms_sides, days_ago: int = 1) -> "PT.TraderState":
    st = PT.TraderState()
    for sym, side in syms_sides:
        st.positions[sym] = dict(
            symbol=sym, side=side,
            entry_date=(TODAY - timedelta(days=days_ago)).isoformat(),
            entry_price=100.0, target_weight=1.0 / PT.MAX_POSITIONS,
        )
    return st


def describe(to_open) -> str:
    n_long = sum(1 for _, s in to_open if s == +1)
    n_short = len(to_open) - n_long
    return f"{n_long}L/{n_short}S  neto={n_long - n_short:+d}"


# ---------------------------------------------------------------------
# Comparación opcional contra la versión previa
# ---------------------------------------------------------------------

def load_old():
    """Carga paper_trader.py.bak-* si existe, para contrastar."""
    baks = sorted(Path(__file__).parent.glob("paper_trader.py.bak-*"))
    if not baks:
        return None
    # El backup no termina en .py, así que hay que dar el loader a mano.
    import importlib.util
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader("old_trader", str(baks[-1]))
    spec = importlib.util.spec_from_loader("old_trader", loader)
    old = importlib.util.module_from_spec(spec)
    # @dataclass resuelve anotaciones vía sys.modules, así que hay que
    # registrar el módulo ANTES de ejecutarlo.
    sys.modules["old_trader"] = old
    loader.exec_module(old)
    return old


OLD = load_old()


def old_state(*syms_sides, days_ago: int = 1):
    src = make_state(*syms_sides, days_ago=days_ago)
    st = OLD.TraderState()
    st.positions = json.loads(json.dumps(src.positions))
    return st


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_opens_both_sides():
    """El bug principal: los largos se quedaban con los 8 slots."""
    _, to_open = PT.decide_actions(Z, make_state(), TODAY, shortable=SHORTABLE)
    n_short = sum(1 for _, s in to_open if s == -1)
    assert n_short > 0, "no abrió ningún corto — el bug sigue vivo"
    assert abs(sum(s for _, s in to_open)) <= PT.MAX_NET_POSITIONS
    if OLD is not None:
        _, old_open = OLD.decide_actions(Z, old_state(), TODAY)
        assert sum(1 for _, s in old_open if s == -1) == 0
        print(f"    versión previa: {describe(old_open)}")
    print(f"    versión actual:  {describe(to_open)}")


def test_unshortable_does_not_burn_slot():
    """WEAT y CPER tienen los |z| más altos y no son shortables."""
    _, to_open = PT.decide_actions(Z, make_state(), TODAY, shortable=SHORTABLE)
    opened = {s for s, _ in to_open}
    assert "WEAT" not in opened and "CPER" not in opened
    assert len(to_open) == PT.MAX_POSITIONS, \
        f"solo {len(to_open)}/{PT.MAX_POSITIONS} slots usados"
    print(f"    WEAT/CPER descartados, {len(to_open)}/{PT.MAX_POSITIONS} slots llenos")


def test_timeout_does_not_churn():
    """Cerrar por timeout y reabrir el mismo símbolo = rechazo wash trade."""
    st = make_state(("LQD", +1), days_ago=PT.MAX_HOLDING_DAYS + 5)
    to_close, to_open = PT.decide_actions(Z, st, TODAY, shortable=SHORTABLE)
    assert "LQD" in to_close
    assert "LQD" not in {s for s, _ in to_open}, "cerró y reabrió LQD"
    assert "LQD" in st.cooldown
    if OLD is not None:
        oc, oo = OLD.decide_actions(
            Z, old_state(("LQD", +1), days_ago=PT.MAX_HOLDING_DAYS + 5), TODAY)
        assert "LQD" in oc and "LQD" in {s for s, _ in oo}
        print("    versión previa: LQD en cierres Y aperturas (churn)")
    print(f"    versión actual:  cooldown hasta {st.cooldown['LQD'][:10]}")


def test_cooldown_expires():
    st = make_state(("LQD", +1), days_ago=PT.MAX_HOLDING_DAYS + 5)
    PT.decide_actions(Z, st, TODAY, shortable=SHORTABLE)
    for d in range(1, PT.TIMEOUT_COOLDOWN_DAYS + 2):
        probe = make_state()
        probe.cooldown = dict(st.cooldown)
        _, to_open = PT.decide_actions(Z, probe, TODAY + timedelta(days=d),
                                       shortable=SHORTABLE)
        reopened = "LQD" in {s for s, _ in to_open}
        assert reopened == (d >= PT.TIMEOUT_COOLDOWN_DAYS), f"falló en +{d}d"
    print(f"    bloqueado {PT.TIMEOUT_COOLDOWN_DAYS - 1}d, reabre en "
          f"+{PT.TIMEOUT_COOLDOWN_DAYS}d")


def test_net_cap_when_one_sided():
    """
    Si solo hay largos, el tope de neto deja el libro infra-invertido en vez
    de direccional. Es una decisión, no un accidente: subir MAX_NET_POSITIONS
    permite más despliegue a cambio de más beta.
    """
    only_long = pd.Series({k: -abs(v) for k, v in REAL_Z.items()})
    _, to_open = PT.decide_actions(only_long, make_state(), TODAY,
                                   shortable=SHORTABLE)
    assert abs(sum(s for _, s in to_open)) <= PT.MAX_NET_POSITIONS
    print(f"    {len(to_open)} posiciones en vez de {PT.MAX_POSITIONS} largos")


def test_full_book_no_orders():
    """Reproduce el libro real del 2026-07-25: 8 largos, sin slots."""
    current = [("XLE", 1), ("SHY", 1), ("GLD", 1), ("LQD", 1),
               ("UUP", 1), ("XLY", 1), ("IEF", 1), ("SLV", 1)]
    _, to_open = PT.decide_actions(Z, make_state(*current), TODAY,
                                   shortable=SHORTABLE)
    assert len(to_open) == 0
    print("    0 órdenes, igual que el log real de esa noche")


TESTS = [
    ("abre ambos lados", test_opens_both_sides),
    ("no-shortable no quema slot", test_unshortable_does_not_burn_slot),
    ("timeout sin churn", test_timeout_does_not_churn),
    ("cooldown expira", test_cooldown_expires),
    ("tope de neto con señal unilateral", test_net_cap_when_one_sided),
    ("libro lleno, cero órdenes", test_full_book_no_orders),
]


def main() -> int:
    if OLD is None:
        print("(sin paper_trader.py.bak-* — se omite la comparación)\n")
    failed = 0
    for name, fn in TESTS:
        try:
            print(f"  {name}")
            fn()
        except AssertionError as e:
            print(f"    FALLO: {e}")
            failed += 1
        print()
    print("TODOS LOS TESTS PASARON" if not failed else f"{failed} FALLARON")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
