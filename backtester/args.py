import argparse

def add_output_dir_arguments(parser: argparse._ArgumentGroup, default: str) -> None:

    parser.add_argument("-o", "--output_dir", type=str, 
                                     help="path to directory where final csvs should be placed. ",
                                     default=default)

def add_test_predictions_arguments(parser: argparse.ArgumentParser) -> None:
    test_predictions_arguments = parser.add_argument_group('test_predictions_arguments')

    default_prediction_dir = '../data/prediction_data/'
    test_predictions_arguments.add_argument("-p", "--predictions_path", type=str, 
                                               help=f"path to directory of processed prediction files. DEFAULT: {default_prediction_dir}", 
                                               default=default_prediction_dir)
    
    test_predictions_arguments.add_argument("-o", "--output", type=str, 
                                     help="name of file to export both .csv strategy summary and .png plot",
                                     default='newresults')
    
    add_output_dir_arguments(parser, 'data/prediction_data')

def get_test_predictions_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    add_test_predictions_arguments(p)

    return p.parse_args()