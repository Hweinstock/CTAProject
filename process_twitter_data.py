import os
from datetime import datetime
from dateutil import parser
from typing import List
from tqdm import tqdm
import json
import pandas as pd

from config.logger import RootLogger
from config.load_env import DATE_FORMAT
from stock_data import get_stock_data, merge_stock_data, filter_out_neutral, process_data_dir, fill_in_missing_dates, aggregate_delta_days

def read_in_tweet_data(root_dir: str, output_path: str) -> List[str]:
    """

    Args:
        root_dir (str): source path for tweet files
        output_path (str): dest path for .csv files

    Returns:
        List[str]: List of exported filepaths
    """
    RootLogger.log_info(f"Processing tweet data: {root_dir}->{output_path}...")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    stocks = os.listdir(root_dir)
    files = []
    for cur_stock in tqdm(stocks):
        stock_path = os.path.join(root_dir, cur_stock)
        filename = process_dir(stock_path, output_path)
        if filename is not None:
            files.append(filename) 
    RootLogger.log_info(f"Able to produce data for {len(files)} out of {len(stocks)} stocks.")
    return files
    
def process_dir(dir_path: str, output_dir: str) -> str or None:
    """

    Args:
        dir_path (str): stock tweet directory
        output_dir (str): dest dir for .csv

    Returns:
        str or None: returns output filename if successful, otherwise none. 
    """
    files = os.listdir(dir_path)
    stock = os.path.basename(dir_path)
    rows = []
    minimum_date, maxmimum_date = None, None 
    for cur_file in files:
        filepath = os.path.join(dir_path, cur_file)
        
        # Track start and enddate of dataset such that we can get stock data on that range. 
        date = parser.parse(cur_file)
        if minimum_date is None or date < minimum_date:
            minimum_date = date
        if maxmimum_date is None or date > minimum_date:
            maxmimum_date = date 

        tweets = process_file(filepath)
        new_rows = [[date.strftime(DATE_FORMAT), t, stock] for t in tweets]
        rows += new_rows

    df = pd.DataFrame(rows, columns=['date', 'text', 'stock'])
    df = fill_in_missing_dates(df)
    df = aggregate_delta_days(df)

    stock_df = get_stock_data(stock, start_date=minimum_date, end_date=maxmimum_date)
    if stock_df.empty:
        return None
    
    combined_df = pd.merge(df, stock_df, on="date").drop_duplicates()
    output_path = os.path.join(output_dir, f"{stock}.csv")
    combined_df.to_csv(output_path, index=False)
    return output_path

def process_file(filepath: str) -> List[str]:
    """

    Args:
        filepath (str): path to tweet file. 

    Returns:
        List[str]: List of tweet text extracted from file. 
    """
    with open(filepath, "r") as f:
        lines = [" ".join(json.loads(l)['text']) for l in f.readlines()]
        
    return lines

if __name__ == "__main__":
    console_level = 2
    file_level = 3
    RootLogger.initialize('/config/', console_level, file_level)
    root_dir = 'data/raw_tweet_data/'
    output_path = 'data/processed_tweet_data/'
    final_path = os.path.join(output_path, 'tweet-data-f.csv')

    tweet_data = read_in_tweet_data(root_dir, output_path)
    data = process_data_dir(output_path, output_path)
    merged_file = merge_stock_data(data, output_path, 'tweet-data.csv')
    filtered_file = filter_out_neutral(merged_file, final_path, remove=False)