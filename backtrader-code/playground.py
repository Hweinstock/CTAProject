from __future__ import (absolute_import, division, print_function, unicode_literals)

import backtrader as bt 
import os
import sys
from datetime import datetime

class TestStrategy(bt.strategy):

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}, {txt}")
    
    def __init__(self):
        self.dateclose = self.datas[0].close
    
    def next(self):
        self.log(f'Close, %{self.dataclose[0]:.2f}')

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
    cerebro.broker.setcash(100000.0)

    print(f"Starting Porfolio Value: {cerebro.broker.getvalue()}")

    cerebro.run() 

    print(f"Final Porfolio Value: {cerebro.broker.getvalue()}")