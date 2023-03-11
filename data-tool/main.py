from config.logger import RootLogger
from headline_data import HeadlineData 
from config.load_env import DATE_FORMAT
from datetime import datetime

if __name__ == '__main__':
    console_level = 2
    file_level = 3
    RootLogger.initialize('.', console_level, file_level)

    start_date = datetime.strptime('2000-01-06', DATE_FORMAT)
    end_date = datetime.strptime('2023-03-09', DATE_FORMAT)
    df = HeadlineData.download_data(start_date, end_date)


    
    