import os 
import pandas as pd

def read_in_chunked_data(dir_path: str, prefix: str) -> pd.DataFrame:
    """Read in all files with prefix from directory as pandas df and concat them into one df. 

    Args:
        dir_path (str): path to search for files. 
        prefix (str): prefix to match in path. 

    Returns:
        pd.DataFrame: concatted/combined df
    """
    data_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.startswith(prefix) and f.endswith('.csv')]
    combined_df = pd.concat([pd.read_csv(f, lineterminator='\n') for f in data_files])
    combined_df.reset_index(inplace=True, drop=True)
    return combined_df