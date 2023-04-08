import pandas as pd 
from typing import List 

def split_dataframe(df: pd.DataFrame) -> List[pd.DataFrame]:
    unique_stocks = df['stock'].unique()
    dfs = [df[df['stock'] == cur_stock] for cur_stock in unique_stocks]
    return dfs

if __name__ == '__main__':
    predictions_path = 'backtester/predictions.csv'
    df = pd.read_csv(predictions_path, index=False)
    split_dfs = split_dataframe(df)
    print(split_dfs[0])
    print(split_dfs[5])

    
