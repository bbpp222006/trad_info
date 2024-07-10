from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
import uvicorn
import numpy as np


app = FastAPI()

class StrategyResult(BaseModel):
    strategy_decision: dict
    intermediate_results: Dict[str, Any]
    end_time: str

def calculate_momentum(df, periods=[1, 3, 6, 12]):
    momentum = {}
    for period in periods:
        momentum[period] = df.pct_change(period).iloc[-1]
    return momentum

@app.on_event("startup")
async def startup():
    FastAPICache.init(InMemoryBackend())


@app.get("/canary_strategy", response_model=StrategyResult)
@cache(expire=3600)  # Cache for 1 hour
def canary_strategy():
    end_time = datetime.now()
    start_time = end_time - timedelta(days=365 * 2)  # 2 years of data

    tickers = ['TIP', 'SPY', 'IWM', 'EFA', 'EEM', 'VNQ', 'PDBC', 'IEF', 'TLT', 'BIL']
    data = yf.download(tickers, start=start_time, end=end_time, interval='1mo')#,proxy="http://192.168.123.200:20171"

    results = {}
    for ticker in tickers:
        df = data['Adj Close'][ticker].dropna()
        momentum = calculate_momentum(df)
        results[ticker] = {
            'momentum': momentum,
            'weighted_average': np.mean(list(momentum.values()))
        }

    tip_momentum = results['TIP']['weighted_average']

    decision = {}
    if tip_momentum > 0:
        offensive_assets = ['SPY', 'IWM', 'EFA', 'EEM', 'VNQ', 'PDBC', 'IEF', 'TLT']
        positive_momentums = {asset: results[asset]['weighted_average'] for asset in offensive_assets if results[asset]['weighted_average'] > 0}
        sorted_assets = sorted(positive_momentums, key=positive_momentums.get, reverse=True)[:4]

        if len(sorted_assets) < 4:
            sorted_assets.extend(['IEF' if results['IEF']['weighted_average'] > results['BIL']['weighted_average'] else 'BIL'] * (4 - len(sorted_assets)))

        for asset in sorted_assets:
            decision[asset] = 0.25
    else:
        if results['IEF']['weighted_average'] > results['BIL']['weighted_average']:
            decision['IEF'] = 1.0
        else:
            decision['BIL'] = 1.0

    return {
        "strategy_decision": decision,
        "intermediate_results": results,
        "end_time": end_time.isoformat()
    }
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
