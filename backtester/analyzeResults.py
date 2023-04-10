import numpy as np 
import pandas as pd

from typing import List

def plot_against_baseline(df: pd.DataFrame, plot_name: str):
    fig = df.plot.scatter(y='Return [%]', x='Buy & Hold Return [%]', s=2, title='StockBot Return Against Baseline').get_figure()
    fig.savefig(plot_name)

def report_columns(df: pd.DataFrame, columns: List[str]):
    print("Average Values:")
    for col in columns:
        print(f"\t{col}:{df[col].mean()}")
    

if __name__ == '__main__':
    data_source = 'strategy_stats.csv'
    df = pd.read_csv(data_source)
    plot_against_baseline(df)