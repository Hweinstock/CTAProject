import numpy as np 
import pandas as pd

if __name__ == '__main__':
    data_source = 'strategy_stats.csv'
    df = pd.read_csv(data_source)
    fig = df.plot.scatter(y='Return [%]', x='Buy & Hold Return [%]', s=2, title='StockBot Return Against Baseline').get_figure()
    fig.savefig('test')