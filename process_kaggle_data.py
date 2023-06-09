import pandas as pd 
import os 
from dateutil import parser
from args import get_process_kaggle_arguments
from typing import List
from config.logger import RootLogger
from stock_data import *
from tqdm import tqdm

BUFFER_SIZE = 5

def process_raw_kaggle(csv_path: str, export_path: str) -> List[str]:
    """Process kaggle file by splitting it and adding stock data. 

    Args:
        csv_path (str): path to raw kaggle .csv
        export_path (str): dir path to expore kaggle entries. 

    Returns:
        List[str]: _description_
    """
    df = pd.read_csv(csv_path, lineterminator='\n', index_col=0)
    stocks = df['stock'].unique()
    files = []
    RootLogger.log_info(f"Splitting Kaggle Data into {len(stocks)} stock files to start processing.")
    for stock in tqdm(stocks, total=len(stocks)):
        stock_df = df[df['stock'] == stock].copy()
        # Sort by date, then reformat it to a string. 
        stock_df.sort_values(by='date', inplace=True)
        stock_df['date'] = stock_df['date'].map(lambda date: str(parser.parse(date).strftime(DATE_FORMAT)))
        
        stock_df.rename(columns={'title':'text'}, inplace=True)
        # Aggregate the text 
        stock_df = fill_in_missing_dates(stock_df)
        stock_df = aggregate_delta_days(stock_df)
        # Add the stock data to the df 
        stock_df = add_stock_data(stock_df)
        

        stock_df = stock_df.astype({'stock':'str', 'text':'str'})
        stock_df.dropna(inplace=True)
        # If we don't meet cutoff
        if(len(stock_df.index) < HARD_ARTICLE_COUNT_CUTOFF):
            continue

        # Export .csv
        if not os.path.exists(export_path):
            os.mkdir(export_path)

        output_file = os.path.join(export_path, f'{stock}_raw.csv')
        stock_df.to_csv(output_file, index=False)
        files.append(output_file)

    RootLogger.log_info(f"Done Processing Kaggle Data...")
    return files

def add_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add stock data to kaggle barebones data file

    Args:
        df (pd.DataFrame): kaggle df for single stock. 

    Returns:
        pd.DataFrame: combined dataframe with stock data. 
    """
    earliest_date = parser.parse(df.iloc[0]['date'])
    latest_date = parser.parse(df.iloc[-1]['date'])

    buffer = df.head(BUFFER_SIZE).copy()
    buffer['date'] = [latest_date + timedelta(days=i) for i in range(1, len(buffer)+1)]
    buffer['date'] = buffer['date'].apply(lambda d: d.strftime(DATE_FORMAT))

    df = pd.concat([df, buffer], ignore_index=True)

    stock_ticker = df['stock'].iloc[0]
    stock_df = get_stock_data(stock_ticker, earliest_date, latest_date+timedelta(days=BUFFER_SIZE))
    # Remove buffer before first day. 
    stock_df = stock_df[stock_df['date'] >= earliest_date.strftime(DATE_FORMAT)]
    combined_df = pd.merge(stock_df, df, on="date", how='left').drop_duplicates()

    return combined_df

def resplit_data(new_split_data: datetime, path: str = './data/processed_kaggle_data/') -> List[str]:
    """Split by date (-r) option to avoid reprocessing data, but split on a different date. 

    Args:
        new_split_data (datetime): 
        path (str, optional): path find data and place re-split data. Defaults to './data/processed_kaggle_data/'.

    Returns:
        List[str]: list of paths to files resulting from the split. 
    """
    data_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    combined_df = pd.concat([pd.read_csv(f, lineterminator='\n') for f in data_files])
    combined_df.reset_index(inplace=True, drop=True)
    split_files = split_data_on_date('', new_split_data, path, data_df=combined_df)
    return split_files

if __name__ == "__main__":
    args = get_process_kaggle_arguments()
    RootLogger.initialize('./config/', args.verbosity, args.file_verbosity)

    split_date = parser.parse(args.split_date)
    raw_kaggle_dir = './data/raw_kaggle_data/'

    if args.resplit:
        RootLogger.log_info(f"Resplitting data on date {args.split_date}")
        split_file = resplit_data(split_date)
    else:
        files = process_raw_kaggle(args.data_path, './data/raw_kaggle_data/')
        data = process_data_dir('./data/raw_kaggle_data/', args.output_dir)
        merged_file = merge_stock_data(files, args.output_dir, 'kaggle-data.csv')
        # Split file will contain neutrals. 
        split_file = split_data_on_date(merged_file, split_date, args.output_dir, remove=True)