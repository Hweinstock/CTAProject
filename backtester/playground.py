import os
import pandas as pd
from uuid import uuid4
from backtesting import Strategy, Backtest

class SimpleStrategy(Strategy):
    
    def init(self):
        pass
    
    def log_order(self, order):
        if order.size > 0:
            order_type = 'BUY' 
        else:
            order_type = 'SELL'
        print(f"{self.data.df.index[-1]} Executing {order_type} at price {self.data.Close[-1]}")

    def current_prediction(self):
        return self.data.Pred_label[-1]
    
    def next(self):
        print('before')
        print(self.position)
        self.position.close()
        print(self.position)
        print('after')
        
        if self.current_prediction() == 1:
            t = self.sell(size=1)
            self.log_order(t)

        if self.current_prediction() == 0:
            t = self.buy(size=1)
            self.log_order(t)
            

if __name__ == '__main__':
    # prediction_dir = '/data/prediction_data/'
    # stock_files = [file for file in os.listdir(prediction_dir) if file.endswith('.csv')]
    cur_stock = 'AAPL'
    prediction_source = f'../data/prediction_data/{cur_stock}_predictions.csv'

    df = pd.read_csv(prediction_source, parse_dates=['date'], index_col=['date'])
    df.columns = [x.capitalize() for x in df.columns]

    bt = Backtest(df, SimpleStrategy, cash=10000, trade_on_close=True)
    stats = bt.run()
    print(stats)
    fig = bt.plot()
    
    