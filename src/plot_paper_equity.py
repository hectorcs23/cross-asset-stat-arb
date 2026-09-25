"""
plot_paper_equity.py
====================

Equity curve of the live Alpaca paper account, from the daily snapshots
written by paper_trader.py.

Usage:
    python src/plot_paper_equity.py    # writes results/paper-trading/equity_curve.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper-trading"

df = pd.read_csv(OUT / "equity_snapshots.csv", parse_dates=["date"])
df["equity"] = df["equity"] / 1000   # thousands of USD

i_max = df["equity"].idxmax()
i_min = df["equity"].idxmin()

plt.figure()
plt.plot(df["date"], df["equity"], label="Account equity")
plt.plot(df["date"][i_max], df["equity"][i_max], "ro", label="Max")
plt.plot(df["date"][i_min], df["equity"][i_min], "gs", label="Min")
plt.xlabel("Date")
plt.ylabel("Equity (thousand USD)")
plt.title("Live Alpaca paper trading, Apr 19 - Sep 19 2026")
plt.legend()
plt.savefig(OUT / "equity_curve.png")
plt.show()
