import os
from dotenv import load_dotenv
from pathlib import Path
from dataclasses import dataclass
import logging


DATE_FORMAT = "%Y-%m-%d"
STOCK_PRICE_LAG = 10
HARD_ARTICLE_COUNT_CUTOFF = 10
dotenv_path = Path('config/.env')
load_dotenv(dotenv_path)

@dataclass(frozen=True)
class APIkeys:
    finnhubAPI: str = os.getenv('finnhubAPI')
    benzingaNewsAPI: str = os.getenv('benzingaNewsAPI')
    NYTKey: str = os.getenv('NYTKey')
    NYTSecret: str = os.getenv('NYTSecret')
