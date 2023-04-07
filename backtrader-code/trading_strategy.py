from __future__ import (absolute_import, division, print_function, unicode_literals)

import backtrader as bt 
import backtrader.feeds as btfeeds
import os
import sys
import pandas as pd
from datetime import datetime

class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}, {txt}")
    
    def __init__(self):
        print(self.datas[0])
        self.prediction = self.datas[0]
        self.order = None 
        self.buyprice = None 
        self.buycomm = None 
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return 
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value}, Comm: {order.executed.comm}")
                self.buyprice = order.executed.price 
                self.buycomm = order.executed.comm
            elif order.issell():
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value}, Comm: {order.executed.comm}")
            
            self.bar_executed = len(self)
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        self.order = None
    
    def notify_trade(self, trade):
        if not trade.isclosed:
            return 
        
        self.log(f'OPERATION PROFIT, GROSS {trade.pnl}, NET {trade.pnlcomm}')

    def next(self):
        #self.log(f'Close, {self.prediction[0]:.2f}')
        pass
        # if self.order:
        #     return 
        # if not self.position:
        #     if self.dataclose[0] > self.sma[0]:
        #         self.log(f'BUY CREATE {self.dataclose[0]:.2f}')
        #         self.order = self.buy()
        # else:
        #     if self.dataclose[0] < self.sma[0]:
        #         self.log(f'SELL CREATE, {self.dataclose[0]:.2f}')
        #         self.order = self.sell()

if __name__ == "__main__":
    cerebro = bt.Cerebro()

    prediction_path = 'predictions.csv'
    dataframe = pd.read_csv(prediction_path,
                                parse_dates=True,
                                index_col=1)

    data = bt.feeds.PandasData(dataname = dataframe)
    cerebro.addstrategy(TestStrategy)
    cerebro.adddata(data)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0)

    print(f"Starting Porfolio Value: {cerebro.broker.getvalue()}")

    cerebro.run() 
    #m = cerebro.plot()
    #m[0][0].savefig('test')

    print(f"Final Porfolio Value: {cerebro.broker.getvalue()}")