#!/usr/bin/env python3
"""
sync_equity_snapshots.py
========================

Rellena results/paper-trading/equity_snapshots.csv desde el historial de la
cuenta en Alpaca.

Por qué existe: paper_trader.py escribe el punto del día, pero conservarlo
dependía de que la sesión diaria lograra hacer `git push`. Cuando el push
falla (403 del proxy si el repo no está en las fuentes de esa sesión), el
contenedor se recicla y el dato se pierde para siempre.

Alpaca ya guarda el equity diario, así que el registro es reconstruible y el
push deja de ser el único camino: si falla, esto lo pone al día después.

    python sync_equity_snapshots.py              # rellena desde la última fila
    python sync_equity_snapshots.py --all        # densifica todo lo disponible
    python sync_equity_snapshots.py --dry-run    # solo muestra qué agregaría

Nunca sobrescribe una fecha que ya está en el CSV: solo agrega las que faltan.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import requests

ET = dt.timezone(dt.timedelta(hours=-4))
SNAPSHOT = (Path(__file__).resolve().parents[1]
            / "results" / "paper-trading" / "equity_snapshots.csv")
BASE = "https://paper-api.alpaca.markets"


def broker_history(period: str = "3M") -> dict[str, float]:
    key, sec = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if not key or not sec:
        sys.exit("Faltan env vars APCA_API_KEY_ID / APCA_API_SECRET_KEY")

    r = requests.get(f"{BASE}/v2/account/portfolio/history",
                     params={"period": period, "timeframe": "1D"},
                     headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec},
                     timeout=30)
    r.raise_for_status()
    h = r.json()

    out: dict[str, float] = {}
    for ts, eq in zip(h.get("timestamp", []), h.get("equity", [])):
        if not eq:
            continue
        day = dt.datetime.fromtimestamp(ts, dt.timezone.utc).astimezone(ET).date()
        out[day.isoformat()] = float(eq)
    return out


def read_csv() -> dict[str, str]:
    rows: dict[str, str] = {}
    if not SNAPSHOT.exists():
        return rows
    for line in SNAPSHOT.read_text().splitlines():
        if not line.strip() or line.startswith("date,"):
            continue
        d, _, v = line.partition(",")
        rows[d.strip()] = v.strip()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="rellenar también huecos anteriores a la última fila")
    ap.add_argument("--period", default="3M",
                    help="ventana que se pide a Alpaca (1M, 3M, 1A...)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    broker = broker_history(args.period)
    rows = read_csv()
    if not broker:
        print("Alpaca no devolvió historial.")
        return 1

    print(f"Alpaca: {len(broker)} días ({min(broker)} → {max(broker)})")
    print(f"CSV:    {len(rows)} filas" + (f", última {max(rows)}" if rows else ""))

    # Hoy puede estar a medio día: su barra todavía no es de cierre.
    hoy = dt.datetime.now(ET).date().isoformat()
    corte = max(rows) if rows and not args.all else ""
    faltan = {d: v for d, v in broker.items()
              if d not in rows and d > corte and d != hoy}

    if not faltan:
        print("\nNada que agregar.")
        return 0

    print(f"\n{len(faltan)} fecha(s) por agregar:")
    for d in sorted(faltan):
        print(f"  + {d}  ${faltan[d]:,.2f}")

    if args.dry_run:
        print("\n--dry-run: no se escribió nada.")
        return 0

    for d, v in faltan.items():
        rows[d] = f"{v:.2f}"
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text("date,equity\n"
                        + "".join(f"{d},{rows[d]}\n" for d in sorted(rows)))
    print(f"\nEscrito: {SNAPSHOT.name} con {len(rows)} filas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
