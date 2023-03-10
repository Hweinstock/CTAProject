from config.load_env import APIkeys 
from benzinga import news_data 
from benzinga.benzinga_errors import BadRequestError
from datetime import datetime
from dateutil import parser

from typing import List
DATE_FORMAT = "%Y-%M-%d"
# start_date = '2000-10-06'
# end_date = '2023-03-09'
# paper = news_data.News(APIkeys.benzingaNewsAPI)
# print(paper.quantified_news(date_from = start_date, 
#                  date_to= end_date, 
#                  company_tickers='AAPL'))

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

class Benzinga:

    def parse_benzinga_news_response(resp):
        articles = [BenzingaArticle.parse_benzinga_article(article) for article in resp]
        return articles

    def query_stock(ticker: str, start_date: datetime, end_date: datetime) -> List[BenzingaArticle]:
        page_count = 0 
        News = news_data.News(APIkeys.benzingaNewsAPI)
        all_articles = []
        
        while True:
            try:
                news_page = News.news(date_from = start_date.strftime(DATE_FORMAT),
                        date_to = end_date.strftime(DATE_FORMAT), 
                        pagesize = 100, 
                        page=page_count)
            
                articles = Benzinga.parse_benzinga_news_response(news_page)
                page_count += 1
                all_articles.append(articles)
            except BadRequestError as err:
                break 
    
        return all_articles

if __name__ == "__main__":
    start_date = datetime.strptime('2000-10-06', DATE_FORMAT)
    end_date = datetime.strptime('2023-03-09', DATE_FORMAT)
    Benzinga.query_stock('AAPL', start_date, end_date)

            
            
        