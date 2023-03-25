from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd 
from tqdm import tqdm

# https://colab.research.google.com/github/DhavalTaunk08/NLP_scripts/blob/master/sentiment_analysis_using_roberta.ipynb

MAX_LEN = 256
TRAIN_BATCH_SIZE = 8 
VALID_BATCH_SIZE = 4 
LEARNING_RATE = 1e-05

from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
train_data_path = '../processed_stock_data/AAPL-data.csv'
train_data = pd.read_csv(train_data_path)

new_df = train_data[["title", "label"]]

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
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.ll = RobertaModel.from_pretrained('roberta-base')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 3)
    
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

def calculate_accuracy(preds, targets):
    n_correct = (preds == targets).sum().item()
    return n_correct

def train(epoch):
    tr_loss = 0
    n_correct = 0 
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _, data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
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
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps 
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {epoch_accu}")

    return 

def valid(model, testing_loader):
    model.eval()
    n_correct = 0
    n_wrong = 0
    total = 0
    tr_loss = 0
    nb_tr_steps = 0 
    nb_tr_examples= 0 
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids).squeeze()
            loss = loss_function(outputs, targets)
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

train_size = 0.8
train_data = new_df.sample(frac=train_size, random_state=200)
test_data = new_df.drop(train_data.index).reset_index(drop=True)
train_data = train_data.reset_index(drop=True)

print(f"Full Dataset: {new_df.shape}")
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

model = RobertaClass() 
model.to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = LEARNING_RATE)

EPOCHS = 1
for epoch in range(EPOCHS):
    train(epoch)

acc = valid(model, testing_loader)
print("Accuracy on test data = %0.2f%%" % acc)

output_model_file = 'pytorch_roberta_sentiment.bin'
output_vocab_file = './'

model_to_save = model
torch.save(model_to_save, output_model_file)
tokenizer.save_vocabulary(output_vocab_file)

print('All files saved')