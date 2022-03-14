# Author: Tim Roller, CFA
# This is a simple python script that given a list of cryptocurrencies over a defined time period,
# calculates the total return, sharpe ratio, sortino ratio, max drawdown, and calmar ratio.

import pandas_datareader as web
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assumptions
N = 365 # days in a year
rf =0.01 # 1% risk free rate
start = dt.datetime(2021, 1, 1)
end = dt.datetime.now()

cryptocurrencies = ['BTC-USD', 'ETH-USD', 'ATOM-USD', 'ADA-USD']

# pull historical prices from yahoo into a dataframe
cryptos = web.DataReader(cryptocurrencies,
                        'yahoo', start, end)['Adj Close']

# convert to daily return and remove na values
df = cryptos.pct_change().dropna()

# add a column, Port, of equally weighted securities
df['Portfolio'] = df.mean(axis=1)


def total_return(return_series):
    comp_ret = (return_series + 1).cumprod()[-1]
    return comp_ret


def sharpe_ratio(return_series, N, rf):
    mean = return_series.mean() * N -rf
    sigma = return_series.std() * np.sqrt(N)
    return mean / sigma


def sortino_ratio(return_series, N,rf):
    mean = return_series.mean() * N -rf
    std_neg = return_series[return_series<0].std()*np.sqrt(N)
    return mean/std_neg


def max_drawdown(return_series):
    comp_ret = (return_series+1).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret/peak)-1
    return dd.min()


totals = df.apply(total_return,axis=0)
sharpes = df.apply(sharpe_ratio, args=(N,rf,),axis=0)
sortinos = df.apply(sortino_ratio, args=(N,rf,), axis=0 )
max_drawdowns = df.apply(max_drawdown,axis=0)
calmars = df.mean()*N/abs(max_drawdowns)

# Graph the return of $1 invested in each security as well as the portfolio
(df+1).cumprod().plot(figsize=(16,10))

# Add a summary table
summary_table = pd.DataFrame()
summary_table['Return'] = totals
summary_table['Sharpe Ratio'] = sharpes
summary_table['Sortino Ratio'] = sortinos
summary_table['Max Drawdown'] = max_drawdowns
summary_table['Calmar Ratio'] = calmars

plt.table(cellText=np.round(summary_table.values,2), colLabels=summary_table.columns,
          rowLabels=summary_table.index,rowLoc='center',cellLoc='center',loc='top',
          colWidths=[0.2]*len(summary_table.columns))
plt.tight_layout()
plt.show()