import pandas as pd 
from typing import List
from tqdm import tqdm 
import os 
from dateutil import parser

from config.logger import RootLogger
from stock_data import merge_stock_data, filter_out_neutral, process_data_dir, split_data_on_date

if __name__ == "__main__":
    console_level = 2
    file_level = 3
    RootLogger.initialize('/config/', console_level, file_level)

    data_path = "data/raw_headline_data/"

    output_path = "data/processed_headline_data/"
    final_path = os.path.join(output_path, 'headline-data-filtered.csv')
    split_date = parser.parse("2022-03-01")

    files = process_data_dir(data_path, output_path)
    merged_file = merge_stock_data(files, output_path, 'headline-data.csv')
    filter_file = filter_out_neutral(merged_file, final_path, remove=False)
    # Split file will contain neutrals. 
    split_file = split_data_on_date(merged_file, split_date, output_path)