from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from torch import cuda
import pandas as pd 
from tqdm import tqdm
from typing import Tuple
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, classification_report

from args import get_model_args

# code taken from: https://colab.research.google.com/github/DhavalTaunk08/NLP_scripts/blob/master/sentiment_analysis_using_roberta.ipynb

MAX_LEN = 256 
HISTORICAL_DELTA = 5

device = 'cuda' if cuda.is_available() else 'cpu'
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)

def get_historical_headers():
    return [f"{i}_past_close" for i in range(1, HISTORICAL_DELTA+1)]

def get_train_data(train_data_path):
    train_data = pd.read_csv(train_data_path)
    return train_data

class HeadlineData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer 
        self.data = dataframe
        self.text = dataframe.text 
        self.targets = dataframe.label
        self.max_len = max_len
        self.historical_data = dataframe[get_historical_headers()]

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
        stock_data = self.historical_data.iloc[[index]].values.flatten().tolist()

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float), 
            'stock_data': torch.tensor(stock_data, dtype=torch.float),
        }
    
class RobertaClass(torch.nn.Module):
    def __init__(self, freeze_roberta=False):
        super(RobertaClass, self).__init__()
        self.ll = DistilBertModel.from_pretrained('distilbert-base-uncased')
        if freeze_roberta:
            self.ll.requires_grad_(False)
        self.pre_classifier = torch.nn.Linear(768 + HISTORICAL_DELTA, 768 + HISTORICAL_DELTA)

        #self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768 + HISTORICAL_DELTA, 3)
    
    def forward(self, input_ids, attention_mask, token_type_ids, historical_data):
        output_1 = self.ll(input_ids=input_ids, 
                           attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        # Add historical data to the layer. 
        pooler = torch.cat((pooler, historical_data), 1)
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        #pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

class RobertaFineTuner:

    def __init__(self, model: torch.nn.Module, 
                 loss_function, optimizer, data_source: pd.DataFrame, 
                 train_batch_size: int, test_batch_size: int, 
                 data_limit: None or int = None):
        self.model = model 
        self.loss_function = loss_function 
        self.optimizer = optimizer
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        if data_limit is not None:
            data_source = data_source.head(data_limit)
        self.training_loader, self.testing_loader = self.initialize_dataloaders(data_source)
        
    
    def initialize_dataloaders(self, data_source: pd.DataFrame) -> Tuple[DataLoader]:
        train_size = 0.8
        train_data = data_source.sample(frac=train_size, random_state=200)
        test_data = data_source.drop(train_data.index).reset_index(drop=True)
        train_data = train_data.reset_index(drop=True)

        data_source_pos = data_source[data_source['label'] == 1]
        data_source_neg = data_source[data_source['label'] == 0]
        data_source_neutral = data_source[data_source['label'] == 2]

        print(f"Full Dataset: {data_source.shape}")
        print(f"\t Positive cases: {data_source_pos.shape}, Negative cases: {data_source_neg.shape}, Neutral cases: {data_source_neutral.shape}")
        print(f"Train Dataset: {train_data.shape}")
        print(f"Test Dataset: {test_data.shape}")

        train_params = {
            'batch_size': self.train_batch_size, 
            'shuffle': True, 
            'num_workers': 0
        }

        test_params = {
            'batch_size': self.test_batch_size, 
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
            historical_data = data['stock_data'].to(device, dtype=torch.long)

            outputs = self.model(ids, mask, token_type_ids, historical_data)
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
        true_positives = 0 
        true_negatives = 0
        false_positives = 0 
        false_negatives = 0

        n_correct = 0
        n_wrong = 0
        total = 0
        tr_loss = 0
        nb_tr_steps = 0 
        nb_tr_examples= 0 

        true_values = []
        predicted_values = []
        with torch.no_grad():
            for _, data in tqdm(enumerate(self.testing_loader, 0), total=len(self.testing_loader)):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.long)
                historical_data = data['stock_data'].to(device, dtype=torch.long)
                outputs = self.model(ids, mask, token_type_ids, historical_data).squeeze()
                # Got error on colab
                try:
                    loss = self.loss_function(outputs, targets)
                except:
                    print("Got error in loss function, skipping data point.")
                    continue

                tr_loss += loss.item()
                big_val, big_idx = torch.max(outputs.data, dim=1)
                n_correct += calculate_accuracy(big_idx, targets)

                nb_tr_steps += 1
                nb_tr_examples += targets.size(0)
                
                # Track false negatives/false positives to look at precision and recall. 
                confidence, choices = torch.max(outputs, 1)

                for index, choice in enumerate(choices):
                    label = targets[index]
                    true_values.append(label.item())
                    predicted_values.append(choice.item())

                if _% 5000 == 0:
                    loss_step = tr_loss / nb_tr_steps 
                    accu_step = (n_correct*100) / nb_tr_examples 
                    print(f"Validation Loss per 100 steps: {loss_step}")
                    print(f"Validation Accuracy per 100 steps: {accu_step}")
        epoch_conf_matrix = confusion_matrix(true_values, predicted_values, labels=[0, 1, 2])
        epoch_loss = tr_loss / nb_tr_steps 
        epoch_accu = accuracy_score(true_values, predicted_values)
        epoch_f1 = f1_score(true_values, predicted_values, average="micro")
        epoch_prec = precision_score(true_values, predicted_values, average="micro")
        epoch_recall = recall_score(true_values, predicted_values, average="micro")

        print(classification_report(true_values, predicted_values))
        print(f"Validation Loss Epoch: {epoch_loss}")
        print(f"Validation Accuracy Epoch: {epoch_accu}")
        print(f"Validation Precision Epoch: {epoch_prec}")
        print(f"Validation Recall Epoch: {epoch_recall}")
        print(f"F1 score: {epoch_f1}")
        print(epoch_conf_matrix)

        return epoch_accu
    
    def save_model(self, model_path: str, vocab_path: str):

        torch.save(self.model, model_path)
        tokenizer.save_vocabulary(vocab_path)

        print('All files saved')
    
def calculate_accuracy(preds, targets):
    n_correct = (preds == targets).sum().item()
    return n_correct

def main():
    args = get_model_args()
    model = RobertaClass(args.freeze_model) 
    model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.learning_rate)
    if args.data_source == 'tweet':
        data_path = '../data/processed_tweet_data/tweet-data-f.csv'
    else:
        data_path = '../data/processed_headline_data/<=2022-03-01.csv'
    df = get_train_data(data_path)
    if args.data_limit is None:
        SPModel = RobertaFineTuner(model=model, loss_function=loss_function, optimizer=optimizer, 
                                   data_source=df, 
                                   train_batch_size= args.train_batch_size, test_batch_size =args.test_batch_size)
    else:
        SPModel = RobertaFineTuner(model=model, loss_function=loss_function, optimizer=optimizer, 
                                   data_source=df, 
                                   train_batch_size= args.train_batch_size, test_batch_size =args.test_batch_size, 
                                   data_limit=args.data_limit)
    for epoch in range(args.epochs):
        SPModel.train(epoch)

        acc = SPModel.valid()
        print("Accuracy on test data = %0.2f%%" % acc)

        output_model_file = 'pytorch_roberta_sentiment.bin'
        output_vocab_file = './'
        SPModel.save_model(output_model_file, output_vocab_file)

if __name__ == '__main__':
    main()