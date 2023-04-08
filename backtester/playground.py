import os
import pandas as pd
from backtesting import Strategy, Backtest

class SimpleStrategy(Strategy):
    
    def init(self):
        self.open_trades = {}

    def current_prediction(self):
        return self.data.Pred_label[-1]

    def next(self):
        
        if self.current_prediction() == 1:
            self.position.close()
            self.sell()
        elif self.current_prediction() == 0:
            self.position.close()
            self.buy()

if __name__ == '__main__':
    # prediction_dir = '/data/prediction_data/'
    # stock_files = [file for file in os.listdir(prediction_dir) if file.endswith('.csv')]
    cur_stock = 'AAPL'
    prediction_source = f'../data/prediction_data/{cur_stock}_predictions.csv'

    df = pd.read_csv(prediction_source, parse_dates=['date'], index_col=['date'])
    df.columns = [x.capitalize() for x in df.columns]

    bt = Backtest(df, SimpleStrategy, cash=10000)
    stats = bt.run()
    print(stats)
    