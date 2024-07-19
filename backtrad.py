import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def calculate_momentum(df, periods=[1, 3, 6, 12]):
    momentum = {}
    for period in periods:
        momentum[period] = df.pct_change(period).iloc[-1]
    return momentum

def canary_strategy(data, tickers):
    results = {}
    for ticker in tickers:
        df = data[ticker].dropna()
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

    return decision, results

def backtest_strategy(start_date, end_date, tickers):
    data = yf.download(tickers, start=start_date, end=end_date, interval='1mo')['Adj Close']
    data = data.ffill().bfill()
    strategy_history = []
    portfolio_value = 1.0
    portfolio_values = []

    for i in range(12, len(data)):
        current_data = data.iloc[:i]
        decision, results = canary_strategy(current_data, tickers)
        
        returns = sum([data.iloc[i][asset] / data.iloc[i-1][asset] - 1 for asset in decision.keys()]) / len(decision)
        portfolio_value *= (1 + returns)
        portfolio_values.append(portfolio_value)
        strategy_history.append({
            'date': data.index[i],
            'decision': decision,
            'portfolio_value': portfolio_value,
            'results': results
        })
    
    return strategy_history, portfolio_values

start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')
tickers = ['TIP', 'SPY', 'IWM', 'EFA', 'EEM', 'VNQ', 'PDBC', 'IEF', 'TLT', 'BIL']

strategy_history, portfolio_values = backtest_strategy(start_date, end_date, tickers)

# Plot portfolio value over time
dates = [entry['date'] for entry in strategy_history]
plt.figure(figsize=(10, 6))
plt.plot(dates, portfolio_values, label='Strategy Portfolio Value')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.title('Canary Strategy Backtest')
plt.legend()
plt.show()

# Display final portfolio value
print(f"Final portfolio value: {portfolio_values[-1]:.2f}")
