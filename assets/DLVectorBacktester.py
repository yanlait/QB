
import random
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop

class DLVectorBacktester(object):
    ''' Class for the vectorized backtesting of
    Linear Regression-based trading strategies.

    Attributes
    ==========
    symbol: str
       TR RIC (financial instrument) to work with
    start: str
        start date for data selection
    end: str
        end date for data selection
    amount: int, float
        amount to be invested at the beginning
    tc: float
        proportional transaction costs (e.g. 0.5% = 0.005) per trade

    Methods
    =======
    get_data:
        retrieves and prepares the base data set
    select_data:
        selects a sub-set of the data
    prepare_lags:
        prepares the lagged data for the regression
    fit_model:
        implements the regression step
    run_strategy:
        runs the backtest for the regression-based strategy
    plot_results:
        plots the performance of the strategy compared to the symbol
    '''

    def __init__(self, symbol, start, end, amount, tc):
        self.history = None
        self.cols = None
        self.lags = None
        self.dpmin = None
        self.dpmax = None
        self.accuracy = None
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc
        self.results = None
        self.get_data()



    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        raw = yf.Ticker(self.symbol).history(period="max", actions=False)
        raw = raw["Close"].to_frame().rename({"Close": "price"}, axis='columns')
        raw['return'] = np.log(raw / raw.shift(1))
        self.dpmin = raw.index.min()
        self.dpmax = raw.index.max()
        raw = raw.loc[self.start:self.end]
        raw = raw.reset_index().rename(columns={"index": "Date"})
        self.data = raw
        return self.data

    def prepare_data(self, lags=5):

        self.data['direction'] = np.where(self.data['return'] > 0, 1, 0)
        self.cols=[]
        for lag in range(1, lags+1):
            col = f'lag_{lag}'
            self.data[col] = self.data['return'].shift(lag)
            self.cols.append(col)
        self.data.dropna(inplace=True)
        self.data['momentum'] = self.data['return'].rolling(5).mean().shift(1)
        self.data['volatility'] = self.data['return'].rolling(20).std().shift(1)
        self.data['distance'] = (self.data['price'] - self.data['price'].rolling(50).mean()).shift(1)
        self.data.dropna(inplace=True)
        self.cols.extend(['momentum', 'volatility', 'distance'])


    def select_data(self, start, end):
        ''' Selects sub-sets of the financial data.
        '''
        data = self.data[(self.data['Date'] >= start) & (self.data['Date'] <= end)].copy()
        return data

    def get_results(self):

        return self.results

    def train_model(self,start_in,end_in,start_out,end_out):
        optimizer = Adam(learning_rate=0.0001)
        def set_seeds(seed=100):
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(100)

        training_data = self.select_data(start_in, end_in).copy()
        training_data = training_data.set_index('Date')
        mu, std = training_data.mean(), training_data.std()
        training_data_ = (training_data - mu) / std
        test_data = self.select_data(start_out, end_out).copy()
        test_data = test_data.set_index('Date')
        test_data_ = (test_data - mu) / std

        set_seeds()
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape = (len(self.cols),)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(training_data_[self.cols],
                  training_data['direction'],
                  epochs=50, verbose=False,
                  validation_split=0.2,
                  shuffle=False)

        self.history = pd.DataFrame(model.history.history)
        acc = model.evaluate(test_data_[self.cols], test_data['direction'])
        self.accuracy = acc[1]
        pred = np.where(model.predict(test_data_[self.cols]) > 0.5, 1, 0)
        test_data['prediction'] = np.where(pred > 0, 1, -1)
        test_data['strategy'] = (test_data['prediction'] * test_data['return'])
        test_data['returns in %'] = test_data['strategy']*100

        trades = test_data['prediction'].diff().fillna(0) != 0
        # subtract transaction costs from return when trade takes place
        test_data['strategy'][trades] -= self.tc
        test_data['creturns'] = (self.amount * test_data['return'].cumsum().apply(np.exp))
        test_data['cstrategy'] = (self.amount * test_data['strategy'].cumsum().apply(np.exp))
        test_data[['return', 'strategy']].sum().apply(np.exp)
        test_data = test_data.reset_index().rename(columns={"index": "Date"})
        self.results = test_data

    def run_strategy(self, start_in, end_in, start_out, end_out, lags=3):
        ''' Backtests the trading strategy.
        '''
        self.prepare_data(lags)
        self.train_model(start_in, end_in, start_out, end_out)


    def get_results(self):
        return self.results

    def get_dpm(self):
        self.get_data()
        return self.dpmin, self.dpmax

    def get_accuracy(self):
        return self.accuracy

    def get_history(self):
        return self.history

if __name__ == '__main__':
    dlbt = DLVectorBacktester('.SPX', '2010-1-1', '2018-06-29', 10000, 0.0)
    print(dlbt.run_strategy('2010-1-1', '2019-12-31',
                            '2010-1-1', '2019-12-31'))
    print(dlbt.run_strategy('2010-1-1', '2015-12-31',
                            '2016-1-1', '2019-12-31'))
    dlbt = DLVectorBacktester('GDX', '2010-1-1', '2019-12-31', 10000, 0.001)
    print(dlbt.run_strategy('2010-1-1', '2019-12-31',
                            '2010-1-1', '2019-12-31', lags=5))
    print(dlbt.run_strategy('2010-1-1', '2016-12-31',
                            '2017-1-1', '2019-12-31', lags=5))
