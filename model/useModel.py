from model import ModelClass, HeadlineData, MAX_LEN, get_model
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import torch 
import pandas as pd
from plotTrainingData import plot_confusion_matrix
from typing import List, Tuple
from args import get_use_model_args
import os 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ModelPredictor:

    def __init__(self, model: torch.nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = 10
        self.data_loader = None
        self.softmax = torch.nn.Softmax(dim=1)
    
    def initialize_dataloaders(self, data_source: pd.DataFrame):

        print(f"Full Dataset: {data_source.shape}")

        dataset = HeadlineData(data_source, self.tokenizer, MAX_LEN)

        eval_params = {
            'batch_size': self.batch_size, 
            'shuffle': True, 
            'num_workers': 0
        }

        data_loader = DataLoader(dataset, **eval_params)

        return data_loader 
    
    def evaluate(self, data_source) -> Tuple[List[int], List[float]]:
        data_loader = self.initialize_dataloaders(data_source)
        predictions = []
        true_values = []
        confidence_values = []
        with torch.no_grad():
            for _, data in tqdm(enumerate(data_loader, 0), total=len(data_loader)):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.long) # We don't use these. 
                historical_data = data['stock_data'].to(device, dtype=torch.long)
                outputs = self.model(ids, mask, token_type_ids, historical_data)
                confidence, choices = torch.max(outputs, 1)
                for choice in choices:
                    predictions.append(choice.item())

                for target in targets:
                    true_values.append(target.item())
                
                for confidence_val in confidence:
                    confidence_values.append(confidence_val.item())
        conf_matrix = confusion_matrix(true_values, predictions)
        print(classification_report(true_values, predictions))
        return predictions, confidence_values, conf_matrix
 
if __name__ == '__main__':
    args = get_use_model_args()
    weights_file_basename = os.path.basename(args.weights)
    model_type = weights_file_basename.split("_")[0].split(":")[0]
    tokenizer, model_source, embedding_size = get_model(model_type)
    model = ModelClass(model_source, embedding_size, is_distill=model_type == 'distill', freeze=False)
    model.load_state_dict(torch.load(args.weights))
    model.to(device)
    model.eval()
    data_source = pd.read_csv(args.data)
    unique_stocks = data_source['stock'].unique()
    num_unique_stock = len(unique_stocks)
    if not args.cutoff is None:
        for stock in unique_stocks:
            num_occurences = len(data_source[data_source['stock'] == stock].index)
            if num_occurences < args.cutoff:
                # Drop all rows with that stock. 
                num_unique_stock -= 1
                data_source = data_source[data_source['stock'] != stock]
        data_source = data_source.reset_index()
    
    print(f'Predicting on {num_unique_stock} stocks.')

    predictor = ModelPredictor(model, tokenizer)
    labels, confidence, conf_matrix = predictor.evaluate(data_source=data_source)
    
    raw_data = {
        'date': data_source['date'].tolist(),
        'stock': data_source['stock'].tolist(),
        'pred_label': labels,
        'confidence': confidence,
    }
    plot_confusion_matrix(conf_matrix)
    output = pd.DataFrame(raw_data)
    output.to_csv(args.output)
    




