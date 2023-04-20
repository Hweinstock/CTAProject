import pandas as pd 
import os 
from dateutil import parser
from args import get_process_headline_parameters
from typing import List
from config.logger import RootLogger
from stock_data import merge_stock_data, filter_out_neutral, process_data_dir, split_data_on_date, aggregate_delta_days, fill_in_missing_dates

if __name__ == "__main__":
    args = get_process_headline_parameters()
    RootLogger.initialize('/config/', args.verbosity, args.file_verbosity)

    output_path = args.output_dir
    final_path = os.path.join(output_path, 'headline-data-filtered.csv')
    split_date = parser.parse(args.split_date)

    files = process_data_dir(args.data_path, output_path)
    merged_file = merge_stock_data(files, output_path, 'headline-data.csv')
    filter_file = filter_out_neutral(merged_file, final_path, remove=False)
    # Split file will contain neutrals. 
    split_file = split_data_on_date(merged_file, split_date, output_path)