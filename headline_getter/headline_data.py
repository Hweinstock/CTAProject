from datetime import datetime, timedelta
from benzinga_test import Benzinga
from dateutil import parser
import yfinance as yf
import pandas as pd
import os 
from config.logger import RootLogger

from config.load_env import DATE_FORMAT, MIN_ARTICLES, STOCK_PRICE_LAG

class HeadlineData:

    def get_stock_data(stock_ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
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
        RootLogger.log_info(f"yFinance Query: stock: {stock_ticker}, start: {adjusted_start}, end: {end_date.strftime(DATE_FORMAT)}")


        raw_stock_data = yf.Ticker(stock_ticker).history(start=adjusted_start, 
                                                                end=adjusted_end)
        
        stock_data = raw_stock_data[['Close', 'Volume']].pct_change().reset_index().rename(str.lower, axis='columns')
        for window in range(1, STOCK_PRICE_LAG+1):
            stock_data[f'{window}_past_close'] = stock_data['close'].shift(window)
        
        stock_data['next_close'] = stock_data['close'].shift(-1)
        stock_data['next_volume'] = stock_data['volume'].shift(-1)
        stock_data['date'] = stock_data['date'].apply(lambda d: d.strftime(DATE_FORMAT))
        return stock_data

        
    def download_data(start_date: datetime, end_date: datetime, output_dir: str = '../raw_stock_data/'):
        """
        Download all articles from Benzinga
        Parse unique stock tags. 
        For each stock download ticker-data and match to headline data. 

        Args:
            start_date (datetime): start
            end_date (datetime): end
        """
        articles_df, unique_stocks = Benzinga.query_articles(start_date, end_date)
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
            stock_df = HeadlineData.get_stock_data(cur_stock, stock_start_date, stock_end_date)

            if stock_df.empty:
                RootLogger.log_warning(f"yFinance failed to find data for {cur_stock}, skipping.")
                continue 

            combined_df = pd.merge(stock_articles, stock_df, on="date").drop_duplicates()
            combined_df.to_csv(os.path.join(output_dir, f'{cur_stock}-data.csv'))