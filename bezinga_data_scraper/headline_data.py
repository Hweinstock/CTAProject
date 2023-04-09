from datetime import datetime
from bezinga_data_scraper.bezinga_api import query_articles
from dateutil import parser
import pandas as pd
import os 
from config.logger import RootLogger
from config.load_env import MIN_ARTICLES
from stock_data import get_stock_data

def download_data(start_date: datetime, end_date: datetime, output_dir: str = 'data/raw_headline_data/'):
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

    for index, cur_stock in enumerate(unique_stocks):
        RootLogger.log_debug(f"On index {index} of {len(unique_stocks)}.")
        stock_articles = articles_df[articles_df['stock'] == cur_stock]

        if len(stock_articles.index) < MIN_ARTICLES:
            continue 

        stock_start_date = parser.parse(stock_articles.iloc[0]['date'])
        stock_end_date = parser.parse(stock_articles.iloc[-1]['date'])
        stock_df = get_stock_data(cur_stock, stock_start_date, stock_end_date)

        if stock_df.empty:
            RootLogger.log_warning(f"yFinance failed to find data for {cur_stock}, skipping.")
            continue 

        combined_df = pd.merge(stock_articles, stock_df, on="date").drop_duplicates()
        combined_df.to_csv(os.path.join(output_dir, f'{cur_stock}-data.csv'), index=False)