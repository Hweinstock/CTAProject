import torch 
from torch.utils.data import Dataset, DataLoader, random_split
from torch import cuda
from transformers import RobertaModel, AutoTokenizer 
import pandas as pd
from tqdm import tqdm
from datetime import datetime

class HeadlineData(Dataset):
    def __init__(self, csvpath: str, tokenizer, limit=None):
        self.tokenizer = tokenizer 
        if limit is not None:
            self.data = pd.read_csv(csvpath).head(limit)
        else:
            self.data = pd.read_csv(csvpath).head(limit)
        self.text = self.data.title 
        self.targets = self.data.label

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        tokenized_text = self.tokenizer(self.text[index], return_tensors="pt")
        target = self.targets[index]
    
        return tokenized_text, torch.tensor(target, dtype=torch.long)

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.ll = RobertaModel.from_pretrained('roberta-base')
        self.ll.requires_grad_(False)

        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 3)
    
    def forward(self, input):
        output_1 = self.ll(**input)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

class Trainer:

    def __init__(self, model, trainset: DataLoader, testset: DataLoader, optimizer, loss_function):
        self.model = model 
        self.trainset = trainset
        self.testset = testset 
        self.optimizer = optimizer 
        self.loss_function = loss_function
    
    def train_one_epoch(self, epoch_index: int):
        running_loss = 0
        last_loss = 0

        for i, data in tqdm(enumerate(self.trainset), total=len(self.trainset)):
            inputs, labels = data 
            model_inputs = inputs.to(device)
            model_labels = labels.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(model_inputs)
            loss = self.loss_function(outputs[0], model_labels)
            loss.backward()
            self.optimizer.step() 

            running_loss += loss.item() 

            if i % 1000 == 0:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0
        
        return last_loss

    def validate(self):
        running_vloss = 0

        for i, vdata in tqdm(enumerate(self.testset), total=len(self.testset)):
            vinputs, vlabels = vdata 
            model_vinputs = vinputs.to(device)
            model_vlabels = vlabels.to(device)
            voutputs = model(model_vinputs)
            vloss = loss_fn(voutputs[0], model_vlabels)
            running_vloss += vloss 
        
        return running_vloss / (i+1)
        


device = 'cuda' if cuda.is_available() else 'cpu'
model = RobertaClass()
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
train_split = 0.8

data_path = "../processed_stock_data/headline-data.csv"
print(f"Reading in dataset from {data_path}")

dataset = HeadlineData(data_path, tokenizer)
trainset, testset = random_split(dataset, [train_split, 1-train_split])
print(f"Training samples: {len(trainset)}")
print(f"Testing samples: {len(testset)}")

train_dataloader = DataLoader(trainset, batch_size = 10, shuffle=True, num_workers=0)
test_dataloader = DataLoader(testset, batch_size=10, shuffle=True, num_workers=0)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

ModelTrainer = Trainer(model, trainset, testset, optimizer, loss_fn)
print("Training One Epoch")

EPOCHS = 3

for epoch in range(EPOCHS):

    model.train(True)
    train_avg_loss = ModelTrainer.train_one_epoch(0)
    model.train(False)
    test_avg_loss = ModelTrainer.validate()
    print(f"Train loss:{train_avg_loss}, Validation loss: {test_avg_loss}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = 'model_{}_{}'.format(timestamp, 0)
    torch.save(model.state_dict(), model_path)
print("done")