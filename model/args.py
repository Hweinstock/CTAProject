import argparse

def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    model_parameters = parser.add_argument_group('model parameters')
    
    model_parameters.add_argument("-l", "--data_limit", type=int,
                   help="limit to model to a certain number of data points for testing purposes.", 
                   default=None)
    
    model_parameters.add_argument("-lr", "--learning_rate", type=float,
                   help="learning rate for model.", 
                   default=1e-05)
    
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
    return p.parse_args()