from fullModel import RobertaClass, HeadlineData, MAX_LEN
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
import torch 
import pandas as pd
from typing import List, Tuple

#model = torch.load('3labelmodel')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)

class ModelPredictor:

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.batch_size = 10
        self.data_loader = None
    
    def initialize_dataloaders(self, data_source: pd.DataFrame):

        print(f"Full Dataset: {data_source.shape}")

        dataset = HeadlineData(data_source, tokenizer, MAX_LEN)

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
                outputs = self.model(ids, mask, token_type_ids, historical_data).squeeze()
                confidence, choices = torch.max(outputs, 1)
                for choice in choices:
                    predictions.append(choice.item())

                for target in targets:
                    true_values.append(target.item())
                
                for confidence_val in confidence:
                    confidence_values.append(confidence_val.item())

        print(classification_report(true_values, predictions))
        return predictions, confidence_values
 
if __name__ == '__main__':
    model = torch.load('../../3labelstockmodel.bin', map_location=torch.device('cpu'))
    datapath = '../data/processed_headline_data/>2022-03-01.csv'
    data_source = pd.read_csv(datapath).head(30)
    predictor = ModelPredictor(model)
    res = predictor.evaluate(data_source=data_source)
    
    output = pd.DataFrame([data_source['data'].tolist(), res[0], res[1]], columns=['date', 'pred_label', 'confidence'])
    print(output)
    




