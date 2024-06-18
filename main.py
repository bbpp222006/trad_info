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

app = FastAPI()

class StrategyResult(BaseModel):
    strategy_decision: str
    intermediate_results: Dict[str, Any]
    end_time: str

def calculate_momentum(df, periods=[1, 3, 6, 12]):
    momentum = {}
    for period in periods:
        momentum[period] = df.pct_change(period).iloc[-1]
    return momentum

def weighted_average(momentum):
    weights = [1/len(momentum)] * len(momentum)
    return sum(m * w for m, w in zip(momentum.values(), weights))

@app.on_event("startup")
async def startup():
    FastAPICache.init(InMemoryBackend())


@app.get("/canary_strategy", response_model=StrategyResult)
@cache(expire=3600)  # Cache for 1 hour
def canary_strategy():
    end_time = datetime.now()
    start_time = end_time - timedelta(days=365 * 2)  # 2 years of data

    tickers = ['TIP', 'SPY', 'IEF', 'BIL']
    data = yf.download(tickers, start=start_time, end=end_time)

    results = {}
    for ticker in tickers:
        df = data['Adj Close'][ticker].dropna()
        results[ticker] = {
            'momentum': calculate_momentum(df),
            'weighted_average': weighted_average(calculate_momentum(df))
        }

    tip_momentum = results['TIP']['weighted_average']
    spy_momentum = results['SPY']['weighted_average']

    if tip_momentum > 0 and spy_momentum > 0:
        decision = 'Hold SPY'
    else:
        ief_momentum = results['IEF']['weighted_average']
        bil_momentum = results['BIL']['weighted_average']
        if ief_momentum > bil_momentum:
            decision = 'Hold IEF'
        else:
            decision = 'Hold BIL'

    return {
        "strategy_decision": decision,
        "intermediate_results": results,
        "end_time": end_time.isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
