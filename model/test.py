from model import ModelClass, get_model, get_train_data, ModelFineTuner
import torch
import unittest
import io 
import sys
from tqdm import tqdm
from statistics import mean
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

def disable_print():
    text_trap = io.StringIO()
    sys.stdout = text_trap

def enable_print():
    sys.stdout = sys.__stdout__

def always_print(string: str):
    enable_print()
    print(string)
    disable_print()


class TestModel(unittest.TestCase):

    def test_backprop(self):
        """
        We check that on small batch (3 items), the accuracy increases over 50 epochs. 
        Subject to randomness, so we average over first 10 and last 10 and look for increase. 
        Note that this test fails with small probability due to randomness in dropout. 
        """
        print("Testing the backpropagation...")
        disable_print()
        # Can only run this with 'distill' on colab. 
        model_type = 'tiny'
        tokenizer, model_source, model_embedding_size = get_model(model_type)
        model = ModelClass(model_source, model_embedding_size, is_distill=model_type == 'distill', freeze=False) 
        model.to(device)
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-5)
        data_path = '../data/processed_headline_data/<=2022-03-01.csv'
        df = get_train_data(data_path)
        ModelTrainer = ModelFineTuner(model=model, loss_function=loss_function, optimizer=optimizer, 
                                    data_source=df, tokenizer=tokenizer,
                                    train_batch_size=1, test_batch_size=1, 
                                    data_limit=3, testing=True)
        
        start_loss = []
        for i in tqdm(range(10)):
            raw_predictions, true_values = ModelTrainer.train(i)
            res, loss = ModelTrainer.valid(i)
            start_loss.append(loss)

        for i in tqdm(range(30)):
            raw_predictions, true_values = ModelTrainer.train(i)
            res, loss = ModelTrainer.valid(i)
        
        end_loss = []
        for i in tqdm(range(10)):
            raw_predictions, true_values = ModelTrainer.train(i)
            res, loss = ModelTrainer.valid(i)
            end_loss.append(loss)
        
        # Check that ending accuracy is greater than starting accuracy. 
        self.assertTrue(mean(start_loss) > mean(end_loss))
        always_print(f"Accuracy went from {mean(start_loss)} to {mean(end_loss)}")
    
if __name__ == '__main__':
    unittest.main()