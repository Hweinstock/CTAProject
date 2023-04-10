import os
import pandas as pd
from uuid import uuid4
from backtesting import Strategy, Backtest

LOGGING = False 

def order_size_from_confidence(conf: float) -> int:
    confidence_map = {
    (0.0, 0.25): 1, 
    (0.25, 0.5): 2, 
    (0.5, 0.75): 4, 
    (0.75, 1.00): 8,
    }
    
    for key, value in confidence_map.items():
        lower_bound = key[0]
        upper_bound = key[1]

        if lower_bound < conf <= upper_bound:
            return value 
    print(f"Error: Unable to find map for confidence value: {conf}")
    return 0

class SimpleStrategy(Strategy):
    
    def init(self):
        pass
    
    def log_order(self, order):
        if LOGGING:
            if order.size > 0:
                order_type = 'BUY' 
            else:
                order_type = 'SELL'
            print(f"{self.current_date()} Executing {order_type} at price {self.data.Close[-1]}")

    def current_prediction(self):
        return self.data.Pred_label[-1]

    def current_date(self):
        return self.data.df.index[-1]

    def current_confidence(self):
        return self.data.df.Confidence[-1]
    
    def next(self):
        self.position.close()
    
        if self.current_prediction() == 1:
            conf = self.current_confidence()
            sell_size = order_size_from_confidence(conf)
            if sell_size == 0:
                print(self.current_date(), self.current_confidence())
                sell_size = 1
            t = self.sell(size=sell_size)
            self.log_order(t)

        elif self.current_prediction() == 0:
            conf = self.current_confidence()
            buy_size = order_size_from_confidence(conf)
            if buy_size == 0:
                print(self.current_date(), self.current_confidence())
                buy_size = 1
            t = self.buy(size=buy_size)
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
    
    