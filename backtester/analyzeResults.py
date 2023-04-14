import numpy as np 
import pandas as pd
import seaborn as sns

from typing import List

def plot_against_baseline(df: pd.DataFrame, plot_name: str):
    fig = df.plot.scatter(y='Return [%]', x='Buy & Hold Return [%]', s=2, title='StockBot Return Against Baseline').get_figure()
    fig.savefig(plot_name)

def plot_results(df: pd.DataFrame, plot_name: str):
    sns.set_theme(style="whitegrid")
    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    g = sns.relplot(
        data=df,
        x="Buy & Hold Return [%]",y="Return [%]",
        hue="# Trades",
        palette=cmap, sizes=(10, 200),
    )
    g.set(title='Model Performance Against Baseline')
    # g.set(xscale="log", yscale="log")
    g.ax.xaxis.grid(True, "minor", linewidth=.25)
    g.ax.yaxis.grid(True, "minor", linewidth=.25)
    g.despine(left=True, bottom=True)
    g.savefig(plot_name)

def report_columns(df: pd.DataFrame, columns: List[str]):
    """Print Average values for each col in df. 

    Args:
        df (pd.DataFrame): target df
        columns (List[str]): list of column names to be printed (avg)
    """
    print("Average Values:")
    for col in columns:
        print(f"\t{col}:{df[col].mean()}")
    

if __name__ == '__main__':
    data_source = 'strategy_stats.csv'
    df = pd.read_csv(data_source)
    plot_results(df, 'test_seaborn')