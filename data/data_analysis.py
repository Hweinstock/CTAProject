import os 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from typing import List, Any

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

def get_bins(min_val: int, max_val: int, num_bins: int):
    interval = max_val - min_val 
    #assert interval % num_bins == 0, "Must choose step that equally divides interval"
    step = int(interval / num_bins)
    return [(step * i) + min_val for i in range(num_bins)] 

def every_other_element(lst: List[Any]) -> List[Any]:
    return lst[::2]

def plot_text_length_hist(data: pd.DataFrame) -> None:
    min_val = 0
    max_val = 105 
    num_bins = 20
    bins = get_bins(min_val, max_val, num_bins)
    sns.set_theme(style="darkgrid")
    fig = sns.histplot(data=data, x='text_len', 
                       bins = bins, 
                       linewidth=.5, 
                       color='plum'
                    )
    fig.set(xlabel='Word Count')
    plt.xticks(every_other_element(bins))
    fig.set_title('Aggregated Headline Lengths (delta=10)')
    fig.get_figure().savefig(os.path.join('plots', 'test_length_hist'), bbox_inches="tight")
    fig.get_figure().clf()

if __name__ == '__main__':
    data_source = 'processed_headline_data/'
    text_length_hist = 'text_length_hist'
    full_df = read_in_chunked_data(data_source, '')
    num_stocks = len(full_df['stock'].unique())
    print(f"Dataset contains {num_stocks} unique stocks")
    print(f"Dataset contains {len(full_df.index)} headlines")
    full_df['text_len'] = full_df['text'].map(lambda x: len(x.split()))
    print(f"Average length of headline: {full_df['text_len'].mean()}")
    print(f"Median length of headline: {full_df['text_len'].median()}")

    plot_text_length_hist(full_df)



