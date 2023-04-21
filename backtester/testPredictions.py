import os
import pandas as pd
from backtesting import Backtest
from tqdm import tqdm
import numpy as np 

from simpleStrategy import SimpleStrategy
from analyzeResults import plot_results, report_columns
from args import get_test_predictions_arguments


np.seterr(divide='ignore')

def parse_stats(stats: pd.Series):
    return stats[['Start', 'Stock', 'End', 'Duration', 'Exposure Time [%]', 'Equity Final [$]',
       'Equity Peak [$]', 'Return [%]', 'Buy & Hold Return [%]',
       'Return (Ann.) [%]', 'Volatility (Ann.) [%]', 'Sharpe Ratio',
       'Sortino Ratio', 'Calmar Ratio', 'Max. Drawdown [%]',
       'Avg. Drawdown [%]', 'Max. Drawdown Duration', 'Avg. Drawdown Duration',
       '# Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
       'Avg. Trade [%]', 'Max. Trade Duration', 'Avg. Trade Duration',
       'Profit Factor', 'Expectancy [%]']]

if __name__ == '__main__':
    args = get_test_predictions_arguments()

    prediction_files = [f for f in os.listdir(args.prediction_path) if f.endswith('.csv')]
    cum_stats = []

    for index, pred_source_file in tqdm(enumerate(prediction_files), total=len(prediction_files)):
        source_file_path = os.path.join(args.prediction_path, pred_source_file)
        stock = pred_source_file.split('_')[0]
        df = pd.read_csv(source_file_path, parse_dates=['date'], index_col=['date'])
        df.columns = [x.capitalize() for x in df.columns]

        bt = Backtest(df, SimpleStrategy, cash=10000, trade_on_close=True)
        stats = bt.run()
        stats.at['Stock'] = stock
        cum_stats.append(parse_stats(stats))

    stats_df = pd.concat(cum_stats, axis=1).transpose()
    plot_results(stats_df, os.path.join('./plots/', args.output))
    report_columns(stats_df, ['Return [%]', 'Volatility (Ann.) [%]', '# Trades', 'Win Rate [%]'])
    stats_df.to_csv(os.path.join('./strategy_stats', args.output + ".csv"))
