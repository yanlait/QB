#
# Python Module with Class
# for Vectorized Backtesting
# of SMA-based Strategies
#
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

from scipy.optimize import brute
from alpha_vantage.timeseries import TimeSeries

class SMAVectorBacktester(object):
    ''' Class for the vectorized backtesting of SMA-based trading strategies.

    Attributes
    ==========
    symbol: str
        RIC symbol with which to work with
    SMA1: int
        time window in days for shorter SMA
    SMA2: int
        time window in days for longer SMA
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval

    Methods
    =======
    get_data:
        retrieves and prepares the base data set
    set_parameters:
        sets one or two new SMA parameters
    run_strategy:
        runs the backtest for the SMA-based strategy
    plot_results:
        plots the performance of the strategy compared to the symbol
    update_and_run:
        updates SMA parameters and returns the (negative) absolute performance
    optimize_parameters:
        implements a brute force optimizeation for the two SMA parameters
    '''

    def __init__(self, symbol, SMA1, SMA2, start, end, amount, tc):
        self.symbol = symbol
        self.SMA1 = SMA1
        self.SMA2 = SMA2
        self.rawdata=None
        self.amount = amount
        self.tc = tc
        self.start = start
        self.end = end
        self.results = None
        self.get_data()


    def get_results(self):

        return self.results

    def get_raw(self):

        return self.rawdata

    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        #raw = yf.Ticker(self.symbol).history(start=self.start, end=self.end, actions=False)
        raw = yf.Ticker(self.symbol).history(period = "max", actions=False)
        raw = raw["Close"].to_frame().rename({"Close": "price"}, axis='columns')
        raw.index.names = ['Date']
        self.rawdata = raw.copy()
        raw['return'] = np.log(raw / raw.shift(1))
        raw['SMA1'] = raw['price'].rolling(self.SMA1).mean()
        raw['SMA2'] = raw['price'].rolling(self.SMA2).mean()
        raw = raw.loc[self.start:self.end]
        #raw.rename(columns={self.symbol: 'price'}, inplace=True)

        raw = raw.reset_index()
        self.data = raw
        return self.data


    def set_parameters(self, SMA1=None, SMA2=None):
        ''' Updates SMA parameters and resp. time series.
        '''
        if SMA1 is not None:
            self.SMA1 = SMA1
            self.data['SMA1'] = self.data['price'].rolling(
                self.SMA1).mean()
        if SMA2 is not None:
            self.SMA2 = SMA2
            self.data['SMA2'] = self.data['price'].rolling(self.SMA2).mean()

    def run_strategy(self):
        ''' Backtests the trading strategy.
        '''
        data = self.data.copy().dropna()
        data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
        data['strategy'] = data['position'].shift(1) * data['return']
        data['returns in %'] = data['strategy']*100
        data['entry'] = data.position.diff()
        data.dropna(inplace=True)
        trades = data['position'].diff().fillna(0) != 0
        data['strategy'][trades] -= self.tc
        data['creturns'] = self.amount * data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = self.amount * data['strategy'].cumsum().apply(np.exp)
        self.results = data
        # gross performance of the strategy
        aperf = data['cstrategy'].iloc[-1]
        # out-/underperformance of strategy
        operf = aperf - data['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2),

    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to the symbol.
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        title = '%s | SMA1=%d, SMA2=%d' % (self.symbol,
                                               self.SMA1, self.SMA2)
        self.results[['creturns', 'cstrategy']].plot(title=title,
                                                     figsize=(10, 6))
    def plotx_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to the symbol.
        '''
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.results["Date"], y=self.results["cstrategy"],
                    mode='lines',
                    name='Strategy'))
        fig.add_trace(go.Scatter(x=self.results["Date"], y=self.results["creturns"],
                    mode='lines',
                    name='Buy/Hold'))
        fig.show() 

    def plot_hist(self):
        ''' Plots histogram of returns.
        '''
        tempr = self.results.strategy.round(3)
        fig = px.histogram(tempr, x="strategy")
        fig.show()

    def plot_strategy(self):
        fig = go.Figure()
        title = '%s | SMA1=%d, SMA2=%d' % (self.symbol,
                                               self.SMA1, self.SMA2)
        fig.add_trace(go.Scatter(x=self.results["Date"], y=self.results["price"],
                            mode='lines',
                            name='Chart'))
        fig.add_trace(go.Scatter(x=self.results["Date"], y=self.results['SMA1'],
                            mode='lines',
                            name='SMA1'))
        fig.add_trace(go.Scatter(x=self.results["Date"], y=self.results['SMA2'],
                            mode='lines',
                            name='SMA2'))
        fig.add_trace(go.Scatter(x=self.results["Date"][self.results.entry == 2], y=self.results['SMA1'][self.results.entry == 2],
                            name='Buy',mode='markers', marker_symbol = "triangle-up", marker=dict(size = 10, color='Green')))
        fig.add_trace(go.Scatter(x=self.results["Date"][self.results.entry == -2], y=self.results['SMA2'][self.results.entry == -2],
                            name='Sell',mode='markers', marker_symbol = "triangle-down", marker=dict(size = 10, color='Red')))
        fig.update_layout(title=title)                    
        fig.show()                                                 

    def update_and_run(self, SMA):
        ''' Updates SMA parameters and returns negative absolute performance
        (for minimazation algorithm).

        Parameters
        ==========
        SMA: tuple
            SMA parameter tuple
        '''
        self.set_parameters(int(SMA[0]), int(SMA[1]))
        return -self.run_strategy()[0]

    def optimize_parameters(self, SMA1_range, SMA2_range):
        ''' Finds global maximum given the SMA parameter ranges.

        Parameters
        ==========
        SMA1_range, SMA2_range: tuple
            tuples of the form (start, end, step size)
        '''
        opt = brute(self.update_and_run, (SMA1_range, SMA2_range), finish=None)
        return opt, -self.update_and_run(opt)


if __name__ == '__main__':
    smabt = SMAVectorBacktester('EUR=', 42, 252,
                                '2010-1-1', '2020-12-31')
    print(smabt.run_strategy())
    smabt.set_parameters(SMA1=20, SMA2=100)
    print(smabt.run_strategy())
    print(smabt.optimize_parameters((30, 56, 4), (200, 300, 4)))
