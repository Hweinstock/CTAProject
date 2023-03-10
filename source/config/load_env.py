import os
from dotenv import load_dotenv
from pathlib import Path
from dataclasses import dataclass
import logging


DATE_FORMAT = "%Y-%m-%d"
MIN_ARTICLES = 100
STOCK_PRICE_LAG = 3
dotenv_path = Path('config/.env')
load_dotenv(dotenv_path)

@dataclass(frozen=True)
class APIkeys:
    finnhubAPI: str = os.getenv('finnhubAPI')
    benzingaNewsAPI: str = os.getenv('benzingaNewsAPI')
