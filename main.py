from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import requests
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
import uvicorn
import numpy as np
from sklearn.linear_model import LinearRegression

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
    data = yf.download(tickers, start=start_time, end=end_time, interval='1mo')

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

@app.get("/B_A_rebalance")
@cache(expire=3600)  # Cache for 1 hour
def B_A_rebalance():
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30 * 2)

    target_tickers = ["CLSK", "IREN", "WULF", "BTBT","COIN","CORZ","BITF","QQQ"]
    base_tickers = ["IBIT", "MAGS"]
    tickers = target_tickers + base_tickers
    data = yf.download(tickers, start=start_time, end=end_time)['Adj Close'] #, session=session

    # 计算每日收益率
    returns = data.pct_change().dropna()

    # 创建一个空的数据框来存储回归结果
    results_df = pd.DataFrame(columns=['Target', 'Intercept'] + base_tickers + ['R_squared', 'Noise Mean', 'Noise Std',"sum_p","mags_pct"])

    # 对每个 target_tickers 进行多元回归计算
    for target in target_tickers:
        X = returns[base_tickers]
        y = returns[target]

        # 创建并拟合回归模型
        model = LinearRegression().fit(X, y)

        # 获取回归系数和截距
        weights = model.coef_
        intercept = model.intercept_
        r_squared = model.score(X, y)

        # 计算回归残差（噪声）
        predicted_y = model.predict(X)
        noise = y - predicted_y

        # 将结果添加到数据框中
        [base_1,base_2] = list(weights)
        results = [target, intercept] + [base_1,base_2] + [r_squared, noise.mean(), noise.std(),sum([base_1,base_2])*r_squared,base_1-base_2]
        results_df.loc[len(results_df)] = results

    # 将结果数据框转换为字典并返回 JSON
    results_dict = results_df.to_dict(orient='records')
    
    return {"results": results_dict}


if __name__ == "__main__":

    # 定义代理服务器
    proxies = {
        'http': 'http://192.168.123.200:20171',
        'https': 'http://192.168.123.200:20171'
    }

    # 配置 requests 使用代理
    session = requests.Session()
    session.proxies.update(proxies)
    uvicorn.run(app, host="0.0.0.0", port=8000)
