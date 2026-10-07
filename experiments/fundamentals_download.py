"""Download prices and point-in-time fundamentals for every S&P 500 member since 2015.

    .venv\\Scripts\\python experiments\\fundamentals_download.py

- Membership: fja05680/sp500 ticker start/end dates (data/fundamentals/sp500_ticker_start_end.csv).
- Prices: 10 years of daily adjusted candles from Yahoo (stocks.candles, cached in data/). Delisted tickers are often
  missing on Yahoo; they are listed in data/fundamentals/missing_prices.txt (a remaining survivorship gap).
- Fundamentals: SEC EDGAR companyfacts for each ticker's CIK (current tickers only: EDGAR's ticker map drops
  delisted names). Only the concepts below are kept, every reported value with its period, form and FILED date,
  so features can use the first-filed value as known at the time. Saved to data/fundamentals/facts_<TICKER>.json.
Resumable: existing files are skipped.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cryptoai import stocks  # noqa: E402

OUT = ROOT / "data" / "fundamentals"
SEC = {"User-Agent": "cryptoai-research-dashboard admin@example.com"}
CONCEPTS = {
    "dei": ["EntityCommonStockSharesOutstanding"],
    "us-gaap": ["WeightedAverageNumberOfDilutedSharesOutstanding", "GrossProfit", "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax", "CostOfRevenue", "CostOfGoodsAndServicesSold",
                "Assets", "NetIncomeLoss", "StockholdersEquity", "PaymentsForRepurchaseOfCommonStock",
                "ProceedsFromIssuanceOfCommonStock"],
}


def members():
    t = pd.read_csv(OUT / "sp500_ticker_start_end.csv", parse_dates=["start_date", "end_date"])
    return sorted(t[(t["end_date"].isna()) | (t["end_date"] >= "2015-01-01")]["ticker"].unique())


def main():
    tickers = members()
    cik = {v["ticker"]: v["cik_str"] for v in requests.get("https://www.sec.gov/files/company_tickers.json",
                                                          headers=SEC, timeout=60).json().values()}
    missing_px, missing_sec = [], []
    for i, t in enumerate(tickers):
        ytk = t.replace(".", "-")  # Yahoo uses BRK-B
        try:
            stocks.candles(ytk, max_age_hours=24 * 30)
        except Exception:
            missing_px.append(t)
        path = OUT / f"facts_{t}.json"
        if not path.exists():
            c = cik.get(t) or cik.get(t.replace(".", "-"))
            if c is None:
                missing_sec.append(t)
            else:
                for attempt in range(3):
                    try:
                        r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(c):010d}.json",
                                         headers=SEC, timeout=60)
                        if r.status_code == 404:
                            missing_sec.append(t)
                            break
                        r.raise_for_status()
                        facts = r.json().get("facts", {})
                        keep = {f"{ns}:{name}": facts.get(ns, {}).get(name, {}).get("units", {})
                                for ns, names in CONCEPTS.items() for name in names}
                        path.write_text(json.dumps(keep))
                        break
                    except (requests.RequestException, ValueError):
                        time.sleep(2 * (attempt + 1))
                time.sleep(0.15)  # SEC asks for at most 10 requests a second
        if i % 50 == 49:
            print(f"{i + 1}/{len(tickers)} done; no prices {len(missing_px)}, no SEC facts {len(missing_sec)}", flush=True)
    (OUT / "missing_prices.txt").write_text("\n".join(missing_px))
    (OUT / "missing_sec.txt").write_text("\n".join(missing_sec))
    print(f"finished: {len(tickers)} tickers; no prices {len(missing_px)}, no SEC facts {len(missing_sec)}")


if __name__ == "__main__":
    main()
