import argparse

def add_output_dir_arguments(parser: argparse._ArgumentGroup, default: str) -> None:

    parser.add_argument("-o", "--output_dir", type=str, 
                                     help="path to directory where final csvs should be placed. ",
                                     default=default)

def add_get_headline_arguments(parser: argparse.ArgumentParser) -> None:
    headline_parameters = parser.add_argument_group('headline parameters')
    
    headline_parameters.add_argument("-s", "--start_date", type=str,
                   help="YYYY-MM-DD format of what date the scraping should start from.", 
                   default='2000-01-06')
    
    headline_parameters.add_argument("-e", "--end_date", type=str,
                   help="YYYY-MM-DD format of what date the scraping should finish scraping from.", 
                   default='2023-04-23')
    
    headline_parameters.add_argument("-c", "--minimum_count", type=int,
                   help="cutoff value for # of articles to keep stock in dataset", 
                   default=100)
    
    
    add_output_dir_arguments(headline_parameters, './data/raw_headline_data/')
    
def add_process_headline_arguments(parser: argparse.ArgumentParser) -> None:
    headline_parameters = parser.add_argument_group('process headline parameters')

    headline_parameters.add_argument("-sp", "--split_date", type=str, 
                                     help="YYYY-MM-DD Date to split train/validate from test.",
                                     default="2022-03-01")
    
    headline_parameters.add_argument("-d", "--data_path", type=str, 
                                     help="path to source the data from.", 
                                     default="data/raw_headline_data/")

    add_output_dir_arguments(headline_parameters, "data/processed_headline_data/")

def add_process_kaggle_arguments(parser: argparse.ArgumentParser) -> None:
    headline_parameters = parser.add_argument_group('process kaggle parameters')

    headline_parameters.add_argument("-sp", "--split_date", type=str, 
                                     help="YYYY-MM-DD Date to split train/validate from test.",
                                     default="2019-06-01")
    
    headline_parameters.add_argument("-d", "--data_path", type=str, 
                                     help="path to source the data from (.csv).", 
                                     default="data/raw_kaggle_data/kaggle.csv")
    
    headline_parameters.add_argument("-r", "--resplit", 
                                     help="option to re-split data by date if already loaded/processed.", 
                                     action="store_true")
    

    add_output_dir_arguments(headline_parameters, "data/processed_kaggle_data/") 

def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    logging_options = parser.add_argument_group('logging options')
    logging_options.add_argument("-v", "--verbosity", type=int, choices=[0,1,2,3], default=2,
                   help="increase output verbosity (default: %(default)s)")
    
    logging_options.add_argument("-fv", "--file_verbosity", type=int, choices=[0, 1, 2, 3], default=3,
                   help="decrease output log file verbosity (default: %(default)s)")

def add_process_predictions_arguments(parser: argparse.ArgumentParser) -> None:
    process_predictions_arguments = parser.add_argument_group('process predictions parameters')

    default_prediction_path = 'backtester/predictions.csv'
    process_predictions_arguments.add_argument("-p", "--predictions_path", type=str, 
                                               help=f"path to raw predictions csv file generated by useModel. DEFAULT: {default_prediction_path}", 
                                               default=default_prediction_path)
    
    add_output_dir_arguments(parser, 'data/prediction_data')

def get_process_predictions_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    add_process_predictions_arguments(p)

    return p.parse_args()

def get_process_headline_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    
    add_process_headline_arguments(p)
    add_logging_arguments(p)

    return p.parse_args()

def get_headline_data_arguments() -> argparse.Namespace:
    
    p = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    
    add_get_headline_arguments(p)
    add_logging_arguments(p)

    return p.parse_args()

def get_process_kaggle_arguments() -> argparse.Namespace:
    
    p = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    
    add_process_kaggle_arguments(p)
    add_logging_arguments(p)

    return p.parse_args()