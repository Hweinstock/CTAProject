from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from torch import cuda
import pandas as pd 
from tqdm import tqdm
from typing import Tuple

# code taken from: https://colab.research.google.com/github/DhavalTaunk08/NLP_scripts/blob/master/sentiment_analysis_using_roberta.ipynb

MAX_LEN = 256
TRAIN_BATCH_SIZE = 10 
VALID_BATCH_SIZE = 4 
LEARNING_RATE = 1e-05

device = 'cuda' if cuda.is_available() else 'cpu'
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

def get_train_data(train_data_path):
    train_data = pd.read_csv(train_data_path)
    new_df = train_data[["title", "label"]]
    return new_df

class HeadlineData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer 
        self.data = dataframe
        self.text = dataframe.title 
        self.targets = dataframe.label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
class RobertaClass(torch.nn.Module):
    def __init__(self, freeze_roberta=False):
        super(RobertaClass, self).__init__()
        self.ll = RobertaModel.from_pretrained('roberta-base')
        if freeze_roberta:
            self.ll.requires_grad_(False)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.ll(input_ids=input_ids, 
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

class RobertaFineTuner:

    def __init__(self, model: torch.nn.Module, 
                 loss_function, optimizer, data_source: pd.DataFrame, 
                 data_limit: None or int = None):
        self.model = model 
        self.loss_function = loss_function 
        self.optimizer = optimizer
        if data_limit is not None:
            data_source = data_source.head(data_limit)
        self.training_loader, self.testing_loader = self.initialize_dataloaders(data_source)
        
    
    def initialize_dataloaders(self, data_source: pd.DataFrame) -> Tuple[DataLoader]:
        train_size = 0.8
        train_data = data_source.sample(frac=train_size, random_state=200)
        test_data = data_source.drop(train_data.index).reset_index(drop=True)
        train_data = train_data.reset_index(drop=True)

        print(f"Full Dataset: {data_source.shape}")
        print(f"Train Dataset: {train_data.shape}")
        print(f"Test Dataset: {test_data.shape}")

        train_params = {
            'batch_size': TRAIN_BATCH_SIZE, 
            'shuffle': True, 
            'num_workers': 0
        }

        test_params = {
            'batch_size': VALID_BATCH_SIZE, 
            'shuffle': True, 
            'num_workers': 0
        }

        training_set = HeadlineData(train_data, tokenizer, MAX_LEN)
        testing_set = HeadlineData(test_data, tokenizer, MAX_LEN)

        training_loader = DataLoader(training_set, **train_params)
        testing_loader = DataLoader(testing_set, **test_params)

        return training_loader, testing_loader

    def train(self, epoch: int):
        tr_loss = 0
        n_correct = 0 
        nb_tr_steps = 0
        nb_tr_examples = 0
        self.model.train()
        for _, data in tqdm(enumerate(self.training_loader, 0), total=len(self.training_loader)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype = torch.long)

            outputs = self.model(ids, mask, token_type_ids)
            loss = self.loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calculate_accuracy(big_idx, targets)

            nb_tr_steps += 1 
            nb_tr_examples += targets.size(0)

            if _%5000==0:
                loss_step = tr_loss/nb_tr_steps 
                accu_step = (n_correct*100)/nb_tr_examples 
                print(f"Training Loss per 5000 steps:  {loss_step}")
                print(f"Training Accuracy per 5000 steps: {accu_step}")
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
        epoch_loss = tr_loss/nb_tr_steps 
        epoch_accu = (n_correct*100)/nb_tr_examples
        print(f"Training Loss Epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {epoch_accu}")

        return 

    def valid(self):
        self.model.eval()
        n_correct = 0
        n_wrong = 0
        total = 0
        tr_loss = 0
        nb_tr_steps = 0 
        nb_tr_examples= 0 
        with torch.no_grad():
            for _, data in tqdm(enumerate(self.testing_loader, 0)):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.long)
                outputs = self.model(ids, mask, token_type_ids).squeeze()
                loss = self.loss_function(outputs, targets)
                tr_loss += loss.item()
                big_val, big_idx = torch.max(outputs.data, dim=1)
                n_correct += calculate_accuracy(big_idx, targets)

                nb_tr_steps += 1
                nb_tr_examples += targets.size(0)

                if _% 5000 == 0:
                    loss_step = tr_loss / nb_tr_steps 
                    accu_step = (n_correct*100) / nb_tr_examples 
                    print(f"Validation Loss per 100 steps: {loss_step}")
                    print(f"Validation Accuracy per 100 steps: {accu_step}")
        epoch_loss = tr_loss / nb_tr_steps 
        epoch_accu = (n_correct*100) / nb_tr_examples 
        print(f"Validation Loss Epoch: {epoch_loss}")
        print(f"Validation Accuracy Epoch: {epoch_accu}")

        return epoch_accu
    
    def save_model(self, model_path: str, vocab_path: str):

        torch.save(self.model, model_path)
        tokenizer.save_vocabulary(vocab_path)

        print('All files saved')
    
def calculate_accuracy(preds, targets):
    n_correct = (preds == targets).sum().item()
    return n_correct

def main():

    model = RobertaClass(False) 
    model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = LEARNING_RATE)

    train_data_path = '../processed_stock_data/headline-data-filtered.csv'
    df = get_train_data(train_data_path)
    SPModel = RobertaFineTuner(model, loss_function, optimizer, df, data_limit=50000)
    EPOCHS = 1
    for epoch in range(EPOCHS):
        SPModel.train(epoch)

    acc = SPModel.valid()
    print("Accuracy on test data = %0.2f%%" % acc)

    output_model_file = 'pytorch_roberta_sentiment.bin'
    output_vocab_file = './'
    SPModel.save_model(output_model_file, output_vocab_file)

if __name__ == '__main__':
    main()