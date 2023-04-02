from __future__ import (absolute_import, division, print_function, unicode_literals)

import backtrader as bt 
import os
import sys
from datetime import datetime

class TestStrategy(bt.Strategy):
    params = (
        ('maperiod', 15),
    )
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}, {txt}")
    
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.sma = bt.indicators.MovingAverageSimple(self.datas[0], period=self.params.maperiod)
        self.order = None 
        self.buyprice = None 
        self.buycomm = None 
        
        # Indicators for plotting. 
        bt.indicators.ExponentialMovingAverage(self.datas[0], period=25)
        bt.indicators.WeightedMovingAverage(self.datas[0], period=25).subplot = True
        bt.indicators.StochasticSlow(self.datas[0])
        bt.indicators.MACDHisto(self.datas[0])
        rsi = bt.indicators.RSI(self.datas[0])
        bt.indicators.SmoothedMovingAverage(rsi, period=10)
        bt.indicators.ATR(self.datas[0]).plot = False
    
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
        self.log(f'Close, {self.dataclose[0]:.2f}')

        if self.order:
            return 
        if not self.position:
            if self.dataclose[0] > self.sma[0]:
                self.log(f'BUY CREATE {self.dataclose[0]:.2f}')
                self.order = self.buy()
        else:
            if self.dataclose[0] < self.sma[0]:
                self.log(f'SELL CREATE, {self.dataclose[0]:.2f}')
                self.order = self.sell()

if __name__ == "__main__":
    cerebro = bt.Cerebro()
    datapath = os.path.join('orcl-1995-2014.txt')


    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath, 
        fromdate=datetime(2000, 1, 1), 
        todate=datetime(2000, 12, 31),
        reverse=False
    )
    cerebro.addstrategy(TestStrategy)
    cerebro.adddata(data)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0)

    print(f"Starting Porfolio Value: {cerebro.broker.getvalue()}")

    cerebro.run() 
    m = cerebro.plot()
    m[0][0].savefig('test')

    print(f"Final Porfolio Value: {cerebro.broker.getvalue()}")