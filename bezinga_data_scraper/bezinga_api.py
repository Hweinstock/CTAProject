from config.load_env import APIkeys, DATE_FORMAT
from config.logger import RootLogger
from benzinga import news_data 
from benzinga.benzinga_errors import BadRequestError
from datetime import datetime
from dateutil import parser
import pandas as pd

from typing import List, Tuple

class BenzingaArticle:

    def parse_benzinga_article(article):
        date = parser.parse(article['created'])
        title = article['title']
        stocks = [s['name'] for s in article['stocks']]
        return BenzingaArticle(date, title, stocks)

    def __init__(self, date: datetime, title: str, stocks: List[str]):
        self.date = date 
        self.title = title 
        self.stocks = stocks 
    
    def to_rows(self) -> List[str]:
        rows = []
        for stock in self.stocks:
            row = [self.date.strftime(DATE_FORMAT), self.title, stock]
            rows.append(row)

        return rows

    def get_headers():
        return ["date", "title", "stock"]

def parse_benzinga_news_response(resp):
    articles = [BenzingaArticle.parse_benzinga_article(article) for article in resp]
    return articles

def convert_articles_to_df(articles: List[BenzingaArticle]) -> Tuple[pd.DataFrame, List[str]]: 
    unique_stocks = set()
    rows = []
    for a in articles:
        unique_stocks.update(a.stocks)
        rows += a.to_rows()

    headers = BenzingaArticle.get_headers() 
    df = pd.DataFrame(data=rows, columns=headers)
    return df, unique_stocks

def query_articles_helper(start_date: datetime, end_date: datetime, News) -> List[BenzingaArticle]:
    """
    Make as many requests to Benzinga API until they stop listing articles. 

    Args:
        start_date (datetime): 
        end_date (datetime): 
        News (Benzinga.News): API obj.

    Returns:
        List[BenzingaArticle]: List of Benzinga Articles between dates. 
    """
    all_articles = []
    page_count = 0

    while True:

        try:
            news_page = News.news(date_from = start_date.strftime(DATE_FORMAT),
                    date_to = end_date.strftime(DATE_FORMAT), 
                    pagesize = 100, 
                    page=page_count)
            articles = parse_benzinga_news_response(news_page)

            if len(articles) == 0:
                break 
            
            RootLogger.log_debug(f'\t Queried page {page_count}, got {len(articles)}')
            page_count += 1

            all_articles += articles
        except BadRequestError as err:
            break 
    
    return all_articles

def query_articles(start_date: datetime, end_date: datetime, limit: int or None = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Benzinga API stops listing articles after 100 pages of 100 articles (10,000), so we repeat request until fails, or 
    we reach the end of the date range. 
    Args:
        ticker (str): Stock Ticker
        start_date (datetime): first date to look for news (lower bound)
        end_date (datetime): last date to look for news (upper bound)

    Returns:
        pd.DataFrame: df of Benzinga articles sorted by date. 
    """
    News = news_data.News(APIkeys.benzingaNewsAPI, log=False)
    all_articles = []
    cur_start_date = start_date 
    cur_end_date = end_date

    while cur_start_date.strftime(DATE_FORMAT) != cur_end_date.strftime(DATE_FORMAT):
        RootLogger.log_info(f'Querying with dates: {cur_start_date.strftime(DATE_FORMAT)}-{cur_end_date.strftime(DATE_FORMAT)}')
        articles = query_articles_helper(cur_start_date, cur_end_date, News)

        if articles == []:
            break 
        sorted_articles = sorted(articles, key=lambda a: a.date)
        cur_start_date = sorted_articles[-1].date 
        all_articles += articles
    
    articles_df, unique_stocks = convert_articles_to_df(all_articles)
    return articles_df.sort_values(by='date'), unique_stocks

            
        