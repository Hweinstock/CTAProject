import os
import pandas as pd
from uuid import uuid4
from backtesting import Strategy, Backtest

class SimpleStrategy(Strategy):
    
    def init(self):
        self.open_trades = {}
    
    def log_order(self, order, tag=''):
        if order.size > 0:
            order_type = 'BUY' 
        else:
            order_type = 'SELL'
        print(f"{self.data.df.index[-1]} Executing {order_type} at price {self.data.Close[-1]} with tag {tag}")

    def current_prediction(self):
        return self.data.Pred_label[-1]
    

    def next(self):

        for key in self.open_trades:
            order_type = self.open_trades[key]
            if order_type == 'sell':
                self.position.close()
                t = self.buy(size=1)
            elif order_type == 'buy':
                self.position.close()
                t = self.sell(size=1)
            self.log_order(t, 'Exit')
            

        self.open_trades = {}
        
        if self.current_prediction() == 1:
            self.position.close()
            t = self.sell(size=1)

            order_id = uuid4()
            self.open_trades[order_id] = 'sell'
            self.log_order(t, 'Entry')

        elif self.current_prediction() == 0:
            self.position.close()
            t = self.buy(size=1)

            order_id = uuid4()
            self.open_trades[order_id] = 'buy'
            self.log_order(t, 'Entry')
            

if __name__ == '__main__':
    # prediction_dir = '/data/prediction_data/'
    # stock_files = [file for file in os.listdir(prediction_dir) if file.endswith('.csv')]
    cur_stock = 'AMZN'
    prediction_source = f'../data/prediction_data/{cur_stock}_predictions.csv'

    df = pd.read_csv(prediction_source, parse_dates=['date'], index_col=['date'])
    df.columns = [x.capitalize() for x in df.columns]

    bt = Backtest(df, SimpleStrategy, cash=10000, trade_on_close=True)
    stats = bt.run()
    fig = bt.plot()
    
    