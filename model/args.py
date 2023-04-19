import argparse
import sys

def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    model_parameters = parser.add_argument_group('model parameters')
    
    model_parameters.add_argument("-l", "--data_limit", type=int,
                   help="limit to model to a certain number of data points for testing purposes.", 
                   default=None)
    
    model_parameters.add_argument("-lr", "--learning_rate", type=float,
                   help="learning rate for model.", 
                   default=1e-05)
    
    model_parameters.add_argument("-m", "--model_type", type=str, 
                                  choices=['distill', 'tiny', 'small', 'medium'],
                                  default='tiny', 
                                  help='which version of BERT to use.')
    
    model_parameters.add_argument("-trb", "--train_batch_size", type=int, 
                                  help="size of batches for training", 
                                  default=8)
    
    model_parameters.add_argument("-tsb", "--test_batch_size", type=int, 
                                  help="size of batches for training", 
                                  default=8)
    
    model_parameters.add_argument("-e", "--epochs", type=int, 
                                  help="number of epochs to interate through data set", 
                                  default=1)
    
    model_parameters.add_argument("-f", "--freeze_model", 
                                  help="freeze weights of imported model to do feature based training instead of fine-tuning", 
                                  action='store_true')
    
    model_parameters.add_argument('-s', "--data_source", 
                                  choices=['tweet', 'headline'],
                                  default='headline',
                                  type=str,
                                  help="source of data")
    
    model_parameters.add_argument('-o', "--output_dir", type=str, 
                                  help="path to where model should be saved.", 
                                  default='.')
    
    model_parameters.add_argument('-w', "--starting_weights", type=str,
                                  help="path to model weights file for model to start with. Note, must be of format: *_e where e is epoch #", 
                                  default=None)
    
    model_parameters.add_argument('-st', '--stats_filepath', type=str, 
                                  help="path to training stats tracking csv, which we will append to",
                                  default=None)
    
    model_parameters.add_argument('-sf', '--stats_filename', type=str, 
                                  help="what to name training data csv that is exported.", 
                                  default='training_data.csv')

    model_parameters.add_argument('-t', '--testing',
                                  help='testing the model (runs train and test on same dataset)',
                                  action='store_true')

def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    logging_options = parser.add_argument_group('logging options')
    logging_options.add_argument("-v", "--verbosity", type=int, choices=[0,1,2,3], default=0,
                   help="increase output verbosity (default: %(default)s)")
    
    logging_options.add_argument("-fv", "--file_verbosity", type=int, choices=[0, 1, 2, 3], default=3,
                   help="decrease output log file verbosity (default: %(default)s)")

def get_model_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(p)
    args = p.parse_args()

    if (args.starting_weights is None and args.stats_filepath is not None) or \
       (args.starting_weights is not None and args.stats_filepath is None):
        p.error("starting_weights and stat_file mututually required.")
    
    if (args.stats_filepath is not None and args.stats_filename != 'training_data.csv'):
        p.error("stats_filepath already specified, can't set new name for file with stats_filename.")

    return p.parse_args()