import os
from dotenv import load_dotenv
from pathlib import Path
from dataclasses import dataclass

dotenv_path = Path('config/.env')
load_dotenv(dotenv_path)

@dataclass(frozen=True)
class APIkeys:
    finnhubAPI: str = os.getenv('finnhubAPI')
    benzingaNewsAPI: str = os.getenv('benzingaNewsAPI')
