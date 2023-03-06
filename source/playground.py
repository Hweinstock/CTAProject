#!/Users/hweinstock/Documents/GitHub/CTAProject/venv/bin/python
import pandas as pd 

from typing import List

class HeadlineData:

    def __init__(self, csv_filepath: str):
        self.filepath = csv_filepath
        self.df = pd.read_csv(csv_filepath, index_col=0)
    
    def get_stock_list(self) -> pd.Series:
        """
        Returns:
            pd.Series: Stock symbols in order of how frequently they occur in the data. 
        """
        return self.df['stock'].value_counts()
    
    def filter_by_stocks(self, stocks: List[str]) -> pd.DataFrame:
        """
        Args:
            stocks (List[str]): filter

        Returns:
            pd.DataFrame: self.df filtered to rows with stocks in stock filter input. 
        """
        return self.df[self.df.stock.isin(stocks)]

if __name__ == '__main__':
    processed_data = HeadlineData('../data/set1/analyst_ratings_processed.csv')
    tech_stocks = processed_data.filter_by_stocks(['AAPL', 'AMZN','GOOGL', 'MSFT', 'META'])
    print(tech_stocks['stock'].value_counts())
