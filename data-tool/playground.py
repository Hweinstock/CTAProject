#!/Users/hweinstock/Documents/GitHub/CTAProject/venv/bin/python
import pandas as pd 
import yfinance as yf
from benzinga import news_data 

from dateutil import parser
from datetime import datetime, timedelta

from typing import List
from config.load_env import DATE_FORMAT, APIkeys

# class HeadlineData:

#     def __init__(self, csv_filepath: str):
#         self.filepath = csv_filepath
#         self.df = pd.read_csv(csv_filepath, index_col=0)
    
#     def get_stock_list(self) -> pd.Series:
#         """
#         Returns:
#             pd.Series: Stock symbols in order of how frequently they occur in the data. 
#         """
#         return self.df['stock'].value_counts()
    
#     def filter_by_stocks(self, stocks: List[str]) -> pd.DataFrame:
#         """
#         Args:
#             stocks (List[str]): filter

#         Returns:
#             pd.DataFrame: self.df filtered to rows with stocks in stock filter input. 
#         """
#         return self.df[self.df.stock.isin(stocks)]
    
#     def stock_data_df(self, stocks: List[str]): 

#         stock_df = self.filter_by_stocks(stocks)
#         print(stock_df['stock'].value_counts())
#         yfinance_data = {}
#         for cur_stock in stocks:
#             stock_date_sorted = stock_df[stock_df.stock == cur_stock].sort_values(by='date')
#             if not stock_date_sorted.empty:
#                 start_date = parser.parse(stock_date_sorted.iloc[0]['date'])
#                 end_date = parser.parse(stock_date_sorted.iloc[-1]['date'])
#                 print("yFinance Query: ", cur_stock, start_date, end_date)
#                 yfinance_data[cur_stock] = yf.Ticker(cur_stock).history(start=(start_date - timedelta(days=1)).strftime("%Y-%m-%d"), 
#                                                                         end=end_date.strftime("%Y-%m-%d"))
        
#         for stock in yfinance_data:
#             print("Resulting Data Points: ", stock, len(yfinance_data[stock].index))
        

if __name__ == '__main__':
    from headline_data import HeadlineData 
    start_date = datetime.strptime('2023-01-06', DATE_FORMAT)
    end_date = datetime.strptime('2023-03-09', DATE_FORMAT)
    df = HeadlineData.download_data(start_date, end_date)
