"""Download Binance's public USDT-M futures metrics (5-minute open interest and long/short ratios) for our coins.

    .venv\\Scripts\\python experiments\\download_futures_metrics.py

Source: data.binance.vision/data/futures/um/daily/metrics/ (free, from 2020-09). The live REST endpoints keep
only 30 days, so these files are the only free history. Saved to data/futures_metrics_<SYMBOL>.csv.
"""
import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cryptoai import config  # noqa: E402

URL = "https://data.binance.vision/data/futures/um/daily/metrics/{s}/{s}-metrics-{d}.zip"


def day(sym, d, session):
    for _ in range(3):
        try:
            r = session.get(URL.format(s=sym, d=d), timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            return pd.read_csv(z.open(z.namelist()[0]))
        except (requests.RequestException, zipfile.BadZipFile):
            continue
    return None


def main():
    days = pd.date_range("2020-09-01", pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta("2D"), freq="D")
    for s in config.SYMBOLS:
        sym = s.replace("/", "")
        path = ROOT / "data" / f"futures_metrics_{sym}.csv"
        have = pd.read_csv(path, parse_dates=["create_time"]) if path.exists() else None
        todo = [d for d in days if have is None or d > have["create_time"].max().tz_localize(None).normalize()]
        session = requests.Session()
        with ThreadPoolExecutor(8) as pool:
            parts = [p for p in pool.map(lambda d: day(sym, d.strftime("%Y-%m-%d"), session), todo) if p is not None]
        new = pd.concat(parts) if parts else pd.DataFrame()
        df = pd.concat([have, new]) if have is not None else new
        df["create_time"] = pd.to_datetime(df["create_time"])
        df = df.drop_duplicates("create_time").sort_values("create_time")
        df.to_csv(path, index=False)
        print(f"{sym}: {len(df):,} rows, {df['create_time'].min()} -> {df['create_time'].max()}", flush=True)


if __name__ == "__main__":
    main()
