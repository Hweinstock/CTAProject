import numpy as np 
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

from typing import List, Any

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
    g.ax.xaxis.grid(True, "minor", linewidth=.25)
    g.ax.yaxis.grid(True, "minor", linewidth=.25)
    g.despine(left=True, bottom=True)
    g.savefig(os.path.join('plots', plot_name))
    plt.clf()

def get_bins(min_val: int, max_val: int, num_bins: int):
    interval = max_val - min_val 
    #assert interval % num_bins == 0, "Must choose step that equally divides interval"
    step = interval / num_bins
    return [(step * i) + min_val for i in range(num_bins)] 

def every_other_element(lst: List[Any]) -> List[Any]:
    return lst[::2]

def plot_distribution(df: pd.DataFrame, plot_name: str):
    min_val = -5
    max_val = 5
    num_bins = 40
    bins = get_bins(min_val, max_val, num_bins)
    sns.set_theme(style="darkgrid")
    fig = sns.histplot(data=df, x='Return [%]', 
                       hue='profitable',
                       bins = bins, 
                       linewidth=.5, 
                       palette=['red', 'green']
                    )
    plt.legend([],[], frameon=False)
    #plt.xticks(every_other_element(bins))
    #plt.xticks(every_other_element(bins))
    fig.set_title('Earnings Across Stocks')
    fig.get_figure().savefig(os.path.join('plots', plot_name), bbox_inches="tight")
    fig.get_figure().clf()

def plot_profitable(df: pd.DataFrame, plot_name: str):
    fig = sns.histplot(data=df, x='profitable', 
                       hue='profitable', 
                       linewidth=.5, 
                       palette=['red', 'green']
                    )
    plt.legend([],[], frameon=False)
    #plt.xticks(every_other_element(bins))
    #plt.xticks(every_other_element(bins))
    fig.set_title('Profitable or Not by Stock')
    fig.get_figure().savefig(os.path.join('plots', plot_name), bbox_inches="tight")
    fig.get_figure().clf()

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
    data_source = 'strategy_stats/newresults.csv'
    df = pd.read_csv(data_source)
    df = df[df['Buy & Hold Return [%]'] < 100]
    df['profitable'] = df['Return [%]'].map(lambda x: 'yes' if x >= 0 else 'no')
    #df = df.astype({"profitable": str})
    print(f"median: {df['Return [%]'].median()}")
    print(f"mean: {df['Return [%]'].mean()}")
    plot_results(df, 'presentationPlot')
    plot_distribution(df, 'returns_hist')
    plot_profitable(df, 'profitable_count')