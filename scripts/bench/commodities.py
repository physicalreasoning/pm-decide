#!/usr/bin/env python3
"""Pre-registered test (FINDINGS §49): does the oil effect of §47 hold on Kalshi's other daily
commodity ladders, which had not been looked at when this was written?

The outside model is the §47 one, unchanged and without fitted parameters: a lognormal at the last
hourly close of the front-month future before the read, volatility from the same clock window on
the previous 60 days. So every settled event is out of sample and no warm-up is needed.

Primary: 24 h before close, pooled over the five series, outside model minus the market at its
most favourable in-quote price (log loss). PASS if the 95% date-block CI lower bound is above 0.

    uv run scripts/bench/fetch.py KXGOLDD KXSILVERD KXCOPPERD KXBRENTD KXNATGASD
    uv run scripts/bench/commodities.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
import evaluate as ev  # noqa: E402
import pandas as pd  # noqa: E402

from isotherm import metrics  # noqa: E402

SERIES = {
    "KXGOLDD": "GC=F",
    "KXSILVERD": "SI=F",
    "KXCOPPERD": "HG=F",
    "KXBRENTD": "BZ=F",
    "KXNATGASD": "NG=F",
}
OUT = pathlib.Path("results/bench/commodities.json")


def gain(rows):
    d = np.array([r["clip"] - r["outside"] for r in rows])
    days = np.array([r["day"] for r in rows])
    return {"diff": float(d.mean()), "ci": metrics.date_bootstrap_mean(days, d, 2000), "n": int(len(d))}


def main():
    rows = []
    for series, sym in SERIES.items():
        df = pd.read_parquet(ev.RAW / "{}.parquet".format(series))
        px = ev.outside_prices(sym)
        for pr in ev.problems(series, df):
            pr["out"] = ev.outside_dist(pr, px)
            if pr["out"] is None:
                continue
            rows.append(
                {
                    "series": series,
                    "lead": pr["lead"],
                    "day": pr["day"],
                    "kind": pr["kind"],
                    "outside": ev.logscore(pr["out"], pr),
                    "clip": ev.logscore(ev.in_quote(pr), pr),
                    "bid": ev.logscore(ev.market(pr, "bid"), pr),
                    "mid": ev.logscore(ev.market(pr), pr),
                }
            )
    res = {"by_series": {}, "pooled": {}}
    for lead in ("24h", "6h", "1h"):
        r = [x for x in rows if x["lead"] == lead]
        if len(r) > 1:
            res["pooled"][lead] = gain(r)
        for series in SERIES:
            rs = [x for x in r if x["series"] == series]
            if len(rs) > 1:
                res["by_series"].setdefault(series, {})[lead] = gain(rs) | {
                    "mid_logloss": float(np.mean([x["mid"] for x in rs]))
                }
    p = res["pooled"].get("24h")
    res["verdict"] = "PASS" if p and p["ci"][0] > 0 else "FAIL"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
