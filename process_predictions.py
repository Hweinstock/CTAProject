import pandas as pd 
import numpy as np 

from typing import List 
from tqdm import tqdm
from datetime import timedelta
from dateutil import parser
import os
from stock_data import get_stock_data
from config.load_env import DATE_FORMAT


BUFFER_SIZE = 5
def split_dataframe(df: pd.DataFrame) -> List[pd.DataFrame]:
    """Split dataframe on unique stocks

    Args:
        df (pd.DataFrame): combined df

    Returns:
        List[pd.DataFrame]: list of dataframes, one for each stock. 
    """
    unique_stocks = df['stock'].unique()
    dfs = [df[df['stock'] == cur_stock] for cur_stock in unique_stocks]
    return dfs

def add_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add Close Column to predictions df. 

    Args:
        df (pd.DataFrame): starting dataset in form [date, stock, pred_label, confidence, close]

    Returns:
        pd.DataFrame: original dataframe with close column. 
    """
    earliest_date = parser.parse(df.iloc[0]['date'])
    latest_date = parser.parse(df.iloc[-1]['date'])

    buffer = df.head(BUFFER_SIZE).copy()
    buffer['date'] = [latest_date + timedelta(days=i) for i in range(1, len(buffer)+1)]
    buffer['date'] = buffer['date'].apply(lambda d: d.strftime(DATE_FORMAT))

    buffer['pred_label'] = [-1 for i in range(1, len(buffer)+1)]
    buffer['confidence'] = [-1 for i in range(1, len(buffer)+1)]
    df = pd.concat([df, buffer], ignore_index=True)

    stock_ticker = df['stock'].iloc[0]
    stock_df = get_stock_data(stock_ticker, earliest_date, latest_date+timedelta(days=BUFFER_SIZE), raw=True)
    stock_df = stock_df[['date', 'close']]
    # Remove buffer before first day. 
    stock_df = stock_df[stock_df['date'] >= earliest_date.strftime(DATE_FORMAT)]

    combined_df = pd.merge(stock_df, df, on="date", how='left').drop_duplicates()
    # Repair NaN values
    combined_df['stock'].fillna(stock_ticker, inplace=True)
    combined_df['pred_label'].fillna(-1, inplace=True)
    combined_df['confidence'].fillna(-1, inplace=True)


    return combined_df

if __name__ == '__main__':
    predictions_path = 'backtester/predictions.csv'
    export_path = 'data/prediction_data'

    df = pd.read_csv(predictions_path, index_col=[0])
    split_dfs = split_dataframe(df)
    res_df = add_stock_data(split_dfs[0])

    if not os.path.exists(export_path):
        os.mkdir(export_path)
    
    for index, cur_df in tqdm(enumerate(split_dfs), total=len(split_dfs)):
        res_df = add_stock_data(cur_df)
        stock = res_df['stock'].iloc[0]
        res_df = res_df.drop('stock', axis=1)
        filepath = os.path.join(export_path, f"{stock}_predictions.csv")
        res_df.to_csv(filepath, index=False)




    
