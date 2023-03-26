import pandas as pd 
from typing import List
from enum import IntEnum 
from tqdm import tqdm 
import os 

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

def process_stock_csv(path: str, output_path: str) -> str:

    df = pd.read_csv(path, index_col=[0])

    df['label'] = df['next_close'].apply(determine_label)
    #df.reset_index(drop=True)
    filename = os.path.basename(path)
    outputfile = os.path.join(output_path, filename)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    df.to_csv(outputfile, index=False)
    return outputfile

def process_data_dir(dir_path: str, output_path: str) -> List[str]:
    
    csv_to_process = [f for f in os.listdir(dir_path) if os.path.splitext(f)[-1] == '.csv']
    output_files = []
    for _, csv_name in tqdm(enumerate(csv_to_process), total=len(csv_to_process)):
        filename = process_stock_csv(os.path.join(dir_path, csv_name), output_path)
        output_files.append(filename)
    
    return output_files

def merge_stock_data(file_names: List[str], output_dir: str, output_name: str) -> str:
    
    df = pd.concat((pd.read_csv(f) for f in file_names), ignore_index=True)

    output_file = os.path.join(output_dir, output_name)
    df.to_csv(output_file)
    return output_file

def filter_out_neutral(df: pd.DataFrame, outputfile: str):
    new_df = df[df['label'] != Label.NEUTRAL]
    new_df.to_csv(outputfile)
    return outputfile



if __name__ == "__main__":
    data_path = "../raw_stock_data/"

    output_path = "../processed_stock_data/"
    final_path = os.path.join(output_path, 'headline-data-filtered.csv')

    # process_stock_csv(file_path, output_path)
    files = process_data_dir(data_path, output_path)
    merged_file = merge_stock_data(files, output_path, 'headline-data.csv')
    filter_file = filter_out_neutral(pd.read_csv(merged_file), final_path)