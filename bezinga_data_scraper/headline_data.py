from datetime import datetime
from bezinga_data_scraper.bezinga_api import query_articles
from dateutil import parser
import pandas as pd
import os 
from config.logger import RootLogger
from config.load_env import STOCK_PRICE_LAG
from stock_data import get_stock_data
from tqdm import tqdm

def fill_in_missing_dates(stock_articles_df: pd.DataFrame) -> pd.DataFrame:
    stock_articles_df = stock_articles_df.groupby(by='date').agg({'title': lambda x: ".".join(x), 
                                               'stock': lambda x: x.iloc[0]}).reset_index()
    
    stock_articles_df = stock_articles_df.set_index('date', drop=True)
    stock_articles_df.index=pd.to_datetime(stock_articles_df.index)
    stock_articles_df = stock_articles_df.asfreq('D', fill_value='')
    stock_articles_df['date'] = stock_articles_df.index 
    stock_articles_df.reset_index(inplace=True, drop=True)
    stock_articles_df['date'] = stock_articles_df['date'].dt.strftime('%Y-%m-%d')

    return stock_articles_df

def aggregate_day_k(original_df: pd.DataFrame, stock_articles_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Add text for day d-k to day d and return the resulting dataframe. 

    We must pass in the original to avoid duplicate adding. 

    Args:
        original_df (pd.DataFrame): df before any text was modified. 
        stock_articles_df (pd.DataFrame): df currently in modification
        k (int): offset to add text

    Returns:
        pd.DataFrame: updated version of stock_articles_df with offset text added. 
    """
    stock_articles_df['offset_text'] = original_df['text'].shift(k).fillna("")
    stock_articles_df['text'] = stock_articles_df['offset_text'] + " " + stock_articles_df['text']
    stock_articles_df['text'] = stock_articles_df['text'].str.strip()
    stock_articles_df = stock_articles_df.drop(columns=['offset_text'])
    # Delete concatenations of empty strings. 
    stock_articles_df.loc[stock_articles_df['text'] == ' '] = ''
    return stock_articles_df

def aggregate_delta_days(stock_articles_df: pd.DataFrame) -> pd.DataFrame:
    """Comprise text from delta days together in the text field. 
    same function as one defined in stock_data. 

    Args:
        stock_articles_df (pd.DataFrame): source df

    Returns:
        pd.DataFrame: output df
    """


    stock_articles_df.rename(columns={'title':'text'}, inplace=True)
    original_df = stock_articles_df.copy()
    for window in range(1, STOCK_PRICE_LAG+1):
        stock_articles_df = aggregate_day_k(original_df, stock_articles_df, k=window)
    #stock_articles_df['text'] = stock_articles_df['text'].apply(adjust_text)
    return stock_articles_df

def download_data(start_date: datetime, end_date: datetime, article_count_cutoff: int, output_dir: str = 'data/raw_headline_data/'):
    """
    Download all articles from Benzinga
    Parse unique stock tags. 
    For each stock download ticker-data and match to headline data. 

    Args:
        start_date (datetime): start
        end_date (datetime): end
    """
    articles_df, unique_stocks = query_articles(start_date, end_date)
    RootLogger.log_info(f"Found {len(articles_df.index)} articles with {len(unique_stocks)} stocks")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    RootLogger.log_info(f"Exporting indivdual stock headline data to directory {output_dir}")
    matching_stocks = 0
    for index, cur_stock in tqdm(enumerate(unique_stocks), total=len(unique_stocks)):
        RootLogger.log_debug(f"On index {index} of {len(unique_stocks)}.")
        stock_articles = articles_df[articles_df['stock'] == cur_stock]

        if len(stock_articles.index) < article_count_cutoff:
            continue 

        matching_stocks += 1
        
        stock_start_date = parser.parse(stock_articles.iloc[0]['date'])
        stock_end_date = parser.parse(stock_articles.iloc[-1]['date'])

        stock_articles = fill_in_missing_dates(stock_articles)
        agg_stock_articles = aggregate_delta_days(stock_articles)

        assert len(stock_articles.index) == len(agg_stock_articles.index), "Aggregating articles causes decrease in dimension!"

        stock_df = get_stock_data(cur_stock, stock_start_date, stock_end_date)

        if stock_df.empty:
            RootLogger.log_warning(f"yFinance failed to find data for {cur_stock}, skipping.")
            continue 
            
        combined_df = pd.merge(agg_stock_articles, stock_df, on="date").drop_duplicates()
        output_file = f'{cur_stock}-data.csv'
        RootLogger.log_debug(f"Outputting data to {output_file}")
        combined_df.to_csv(os.path.join(output_dir, output_file), index=False) 
    RootLogger.log_info(f"Exported data for {matching_stocks} stocks to {output_dir}")