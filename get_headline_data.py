from config.logger import RootLogger
from bezinga_data_scraper.headline_data import download_data
from config.load_env import DATE_FORMAT
from datetime import datetime
from args import get_headline_data_parameters

if __name__ == '__main__':
    args = get_headline_data_parameters()
    RootLogger.initialize('/config/', args.verbosity, args.file_verbosity)
    args = get_headline_data_parameters()
    start_date = datetime.strptime(args.start_date, DATE_FORMAT)
    end_date = datetime.strptime(args.end_date, DATE_FORMAT)
    download_data(start_date, end_date, article_count_cutoff=args.minimum_count, output_dir=args.output_dir)


    
    