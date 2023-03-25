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



if __name__ == "__main__":
    data_path = "../raw_stock_data/"
    file_path = os.path.join(data_path, 'AAPL-data.csv')

    output_path = "../processed_stock_data/"

    # process_stock_csv(file_path, output_path)
    process_data_dir(data_path, output_path)