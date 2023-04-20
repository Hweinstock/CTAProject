from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from enum import IntEnum 
import os 
from typing import List, Tuple
from tqdm import tqdm
import numpy as np

from config.load_env import DATE_FORMAT, STOCK_PRICE_LAG
from config.logger import RootLogger

INC_CUTOFF = 0.005 
DEC_CUTOFF = -0.005

class Label(IntEnum):
    INCREASE = 0
    DECREASE = 1
    NEUTRAL = 2

def determine_label(val: float) -> Label:
    """
    Determine label for value using threshholds

    Args:
        val (float)

    Returns:
        Label
    """
    if val >= INC_CUTOFF:
        return Label.INCREASE
    if val <= DEC_CUTOFF:
        return Label.DECREASE

    return Label.NEUTRAL

def get_stock_data(stock_ticker: str, start_date: datetime, end_date: datetime, raw: bool = False) -> pd.DataFrame:
    """

    Args:
        stock_ticker (str): 
        start_date (datetime): 
        end_date (datetime): 

    Returns:
        pd.DataFrame: Dataframe of Relative Close and Volume for stock across range. 
    """
    # Go back 3 days, in case we start on monday and need friday num.
    adjusted_start = (start_date - timedelta(days=3+STOCK_PRICE_LAG)).strftime(DATE_FORMAT)
    adjusted_end = (end_date + timedelta(days=1)).strftime(DATE_FORMAT)
    RootLogger.log_debug(f"yFinance Query: stock: {stock_ticker}, start: {adjusted_start}, end: {end_date.strftime(DATE_FORMAT)}")


    raw_stock_data = yf.Ticker(stock_ticker).history(start=adjusted_start, 
                                                            end=adjusted_end)
    if raw:
        stock_data = raw_stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index().rename(str.lower, axis='columns')
    else:
        stock_data = raw_stock_data[['Close', 'Volume']].pct_change().reset_index().rename(str.lower, axis='columns')

    for window in range(1, STOCK_PRICE_LAG+1):
        stock_data[f'{window}_past_close'] = stock_data['close'].shift(window)
    
    stock_data['next_close'] = stock_data['close'].shift(-1)
    stock_data['next_volume'] = stock_data['volume'].shift(-1)
    stock_data['date'] = stock_data['date'].apply(lambda d: d.strftime(DATE_FORMAT))
    
    return stock_data

def merge_stock_data(file_names: List[str], output_dir: str, output_name: str, remove: bool = True) -> str:
    """Merge all processed stock .csv files together. 

    Args:
        file_names (List[str]): List of files to merge
        output_dir (str): root path of where to place combined .csv
        output_name (str): name of merged .csv file. 
        remove (bool, optional): False means we keep individual .csv, True means delete them. Defaults to True.

    Returns:
        str: filepath of combined file. 
    """
    output_file = os.path.join(output_dir, output_name)
    RootLogger.log_info(f"Merged stock data into {output_file}")
    df = pd.concat((pd.read_csv(f, lineterminator='\n') for f in file_names), ignore_index=True)
    if remove:
        for f in file_names:
            os.remove(f)
    output_file = os.path.join(output_dir, output_name)
    df.to_csv(output_file, index=False)
    return output_file

def filter_out_neutral(data_file: str, output_file: str, remove: bool = True) -> str:
    """Filter dataset to only contain positive and negative labels. 

    Args:
        data_file (str): .csv filepath
        output_file (str): output file path. 
        remove (bool, optional): True removes old .csv file, False keeps it. Defaults to True.

    Returns:
        str: filepath to new file. 
    """
    RootLogger.log_info(f"Filtering out neutral days from {output_file}")
    og_df = pd.read_csv(data_file, lineterminator='\n')
    if remove:
        os.remove(data_file)
    new_df = og_df[og_df['label'] != Label.NEUTRAL]
    new_df.to_csv(output_file, index=False)
    return output_file

def get_stock_historical_headers():
    return [f"{i}_past_close" for i in range(1, STOCK_PRICE_LAG+1)]

def process_stock_csv(path: str, output_path: str) -> str:
    """Process individual CSV file by adding label.  

    Args:
        path (str): path to csv file. 
        output_path (str): where to export resulting csv

    Returns:
        str: filepath to new csv file. 
    """
    df = pd.read_csv(path, lineterminator='\n')

    df['label'] = df['next_close'].apply(determine_label)

    for col in df.columns:
        df[col].replace('', np.nan, inplace=True)

    df.dropna(inplace=True)

    if 'title' in df.columns:
        df.rename(columns={'title':'text'}, inplace=True)

    filename = os.path.basename(path)
    outputfile = os.path.join(output_path, filename)

    # Normalize stock data to 0-1 range. 
    for col in get_stock_historical_headers():
        df[col] = (df[col] / 2.0) + 0.5

    # TODO: check if placing it here works better. 
    # df = fill_in_missing_dates(df)
    # df = aggregate_delta_days(df)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    df.to_csv(outputfile, index=False)
    return outputfile

def process_data_dir(dir_path: str, output_path: str) -> List[str]:
    """Process directory of .csv stock data files. 

    Args:
        dir_path (str): root path of directory
        output_path (str): root path of output directory

    Returns:
        List[str]: List of filepath associated with new files. 
    """
    RootLogger.log_info(f"Processing data directory {dir_path}")
    csv_to_process = [f for f in os.listdir(dir_path) if os.path.splitext(f)[-1] == '.csv']
    output_files = []
    for _, csv_name in tqdm(enumerate(csv_to_process), total=len(csv_to_process)):
        filename = process_stock_csv(os.path.join(dir_path, csv_name), output_path)
        output_files.append(filename)
    RootLogger.log_info(f"Exporting files to {output_path}")
    return output_files

def split_data_on_date(data_path: str, target_date: datetime, output_dir: str, remove: bool = False) -> Tuple[str, str]:
    """Read in a pandas df from csv and write two csv: one before a date, one after. 

    Args:
        data_path (str): original path to .csv
        target_date (datetime): date to split on. 
        output_dir (str): where to write generated .csv files.
        remove (bool): delete original file if true.  

    Returns:
        Tuple[str, str]: pair of filepath to newly written files. 
    """
    RootLogger.log_info(f"Splitting dataframe {data_path} based on {target_date.strftime(DATE_FORMAT)}")
    orig_df = pd.read_csv(data_path, lineterminator='\n')
    
    df_before = orig_df.loc[pd.to_datetime(orig_df['date']) <= target_date]
    df_after = orig_df.loc[pd.to_datetime(orig_df['date']) > target_date]

    if remove:
        os.remove(data_path)
        
    before_filepath = os.path.join(output_dir, f"<={target_date.strftime(DATE_FORMAT)}.csv")
    after_filepath = os.path.join(output_dir, f">{target_date.strftime(DATE_FORMAT)}.csv")

    df_before.to_csv(before_filepath, index=False)
    df_after.to_csv(after_filepath, index=False)

    return before_filepath, after_filepath

def fill_in_missing_dates(data_df: pd.DataFrame) -> pd.DataFrame:
    data_df = data_df.groupby(by='date').agg({'text': lambda x: ".".join(x), 
                                               'stock': lambda x: x.iloc[0]}).reset_index()
    
    data_df = data_df.set_index('date', drop=True)
    data_df.index=pd.to_datetime(data_df.index)
    data_df = data_df.asfreq('D', fill_value='')
    data_df['date'] = data_df.index 
    data_df.reset_index(inplace=True, drop=True)
    data_df['date'] = data_df['date'].dt.strftime('%Y-%m-%d')

    return data_df

def aggregate_day_k(original_df: pd.DataFrame, data_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Add text for day d-k to day d and return the resulting dataframe. 

    We must pass in the original to avoid duplicate adding. 

    Args:
        original_df (pd.DataFrame): df before any text was modified. 
        stock_articles_df (pd.DataFrame): df currently in modification
        k (int): offset to add text

    Returns:
        pd.DataFrame: updated version of stock_articles_df with offset text added. 
    """
    data_df['offset_text'] = original_df['text'].shift(k).fillna("")
    data_df['text'] = data_df['offset_text'] + " " + data_df['text']
    data_df['text'] = data_df['text'].str.strip()
    data_df = data_df.drop(columns=['offset_text'])
    # Delete concatenations of empty strings. 
    data_df.loc[data_df['text'] == ' '] = ''
    return data_df

def aggregate_delta_days(data_df: pd.DataFrame) -> pd.DataFrame:
    """Comprise text from delta days together in the text field. 

    Args:
        stock_articles_df (pd.DataFrame): source df

    Returns:
        pd.DataFrame: output df
    """


    data_df.rename(columns={'title':'text'}, inplace=True)
    original_df = data_df.copy()
    for window in range(1, STOCK_PRICE_LAG+1):
        data_df = aggregate_day_k(original_df, data_df, k=window)

    return data_df