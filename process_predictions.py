import pandas as pd 
from typing import List 
from stock_data import get_stock_data

def split_dataframe(df: pd.DataFrame) -> List[pd.DataFrame]:
    unique_stocks = df['stock'].unique()
    dfs = [df[df['stock'] == cur_stock] for cur_stock in unique_stocks]
    return dfs

def add_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    earliest_date = df['date'].dt.date.min() 
    latest_date = df['date'].dt.date.max()

    buffer_size = 5
    buffer = df.head(-buffer_size).copy()
    buffer['date'] = [latest_date + pd.Timedelta(days=i) for i in range(1, len(buffer)+1)]
    buffer['pred_label'] = [-1 for i in range(1, len(buffer)+1)]
    buffer['confidence'] = [-1 for i in range(1, len(buffer)+1)]
    df = pd.concat([df, buffer], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])

    stock_ticker = df['stock'].iloc[0]
    stock_df = get_stock_data(stock_ticker, earliest_date, latest_date+pd.Timedelta(days=buffer_size), raw=True)
    stock_df = stock_df[['date', 'close']]
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    combined_df = pd.merge(df, stock_df, on="date").drop_duplicates()
    return combined_df

if __name__ == '__main__':
    predictions_path = 'backtester/predictions.csv'
    export_path = '/bot_data/'
    df = pd.read_csv(predictions_path, index_col=[0])
    df['date'] = pd.to_datetime(df['date'])
    split_dfs = split_dataframe(df)
    result_df = add_stock_data(split_dfs[0])
    print(result_df)


    
