from fullModel import RobertaClass, HeadlineData, MAX_LEN
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch 
import pandas as pd

model = torch.load('3labelmodel')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)

class ModelPredictor:

    def __init__(self, model: torch.nn.Module):
        self.model = model 
        self.data_loader = None
    
    def initialize_dataloaders(self, data_source: pd.DataFrame):

        print(f"Full Dataset: {data_source.shape}")

        dataset = HeadlineData(data_source, tokenizer, MAX_LEN)

        data_loader = DataLoader(dataset)

        return data_loader 
    
    def evaluate(self, data_source: pd.DataFrame):
        data_loader = self.initialize_dataloaders(data_source)
        predictions = []
        with torch.no_grad():
            for _, data in tqdm(enumerate(data_loader, 0), total=len(data_loader)):
                date = data['date']
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.long) # We don't use these. 
                historical_data = data['stock_data'].to(device, dtype=torch.long)
                outputs = self.model(ids, mask, token_type_ids, historical_data).squeeze()

                predictions.append(date, outputs.item())

        results = pd.DataFrame(predictions, columns=['date', 'pred_label'])
        return results 



