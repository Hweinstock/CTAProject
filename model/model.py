from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import torch
from torch import cuda
import pandas as pd 
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, classification_report
from plotTrainingData import plot_training_data, plot_loss_data
import os
from args import get_model_args
from typing import Dict, Tuple, Any, List
from dateutil import parser

# adapted from: https://colab.research.google.com/github/DhavalTaunk08/NLP_scripts/blob/master/sentiment_analysis_using_roberta.ipynb

MAX_LEN = 256 
HISTORICAL_DELTA = 10

device = 'cuda' if cuda.is_available() else 'cpu'

def get_model(id: str) -> Tuple[Any]:
    if id == 'distill':
        from transformers import DistilBertModel, DistilBertTokenizer

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
        model = DistilBertModel.from_pretrained("distilbert-base-cased")

        return tokenizer, model, 768
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"prajjwal1/bert-{id}")
        model = AutoModel.from_pretrained(f"prajjwal1/bert-{id}")
        model_sizes = {
            'tiny': 128, 
            'small': 512, 
            'medium': 512
        }
        return tokenizer, model, model_sizes[id]
    
def parse_weights_file(filepath: str) -> Tuple[int, str]:
    epoch = int(filepath.split("_")[-1].split(".")[0])
    return epoch, filepath

def add_true_and_pred_values(new_choices, new_targets, true_values, predicted_values):
    for index, choice in enumerate(new_choices):
        label = new_targets[index]
        true_values.append(label.item())
        predicted_values.append(choice.item())

def get_historical_headers():
    return [f"{i}_past_close" for i in range(1, HISTORICAL_DELTA+1)]

def get_train_data(train_data_path: str) -> pd.DataFrame:
    train_data = pd.read_csv(train_data_path, lineterminator='\n')
    return train_data

def read_in_chunked_data(dir_path: str, prefix: str) -> pd.DataFrame:
    """Read in all files with prefix from directory as pandas df and concat them into one df. 

    Args:
        dir_path (str): path to search for files. 
        prefix (str): prefix to match in path. 

    Returns:
        pd.DataFrame: concatted/combined df
    """
    data_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.startswith(prefix) and f.endswith('.csv')]
    combined_df = pd.concat([get_train_data(f) for f in data_files])
    combined_df.reset_index(inplace=True, drop=True)
    return combined_df

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
            #pad_to_max_length=True,
            padding='max_length',
            truncation=True,
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
    
class ModelClass(torch.nn.Module):
    def __init__(self, model_source, model_embedding_size: int, is_distill: bool, freeze=False):
        super(ModelClass, self).__init__()
        self.ll = model_source
        if freeze:
            self.ll.requires_grad_(False)
        middle_layer_size = int((model_embedding_size + HISTORICAL_DELTA) / 2.0)
        second_middle_layer_size = int(middle_layer_size / 2.0)
        # MLP
        self.layer_1 = torch.nn.Linear(model_embedding_size + HISTORICAL_DELTA, middle_layer_size)
        self.layer_2 = torch.nn.Linear(middle_layer_size, second_middle_layer_size)
        self.dropout_1 = torch.nn.Dropout(0.1)
        self.dropout_2 = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(second_middle_layer_size, 3)
        self.ac_final = torch.nn.Softmax(dim=1)
        self.activation_function = torch.nn.Tanh()
        self.is_distill = is_distill
    
    def forward(self, input_ids, attention_mask, token_type_ids, historical_data):
        if self.is_distill:
            output_1 = self.ll(input_ids=input_ids,
                            attention_mask=attention_mask)
        else:
            output_1 = self.ll(input_ids=input_ids, 
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        # Add historical data to the layer. 
        pooler = torch.cat((pooler, historical_data.float()), 1)
        # Apply it so that they are on same scale. 
        #pooler = torch.nn.GELU()(pooler)
        # Feed to MLP
        pooler = self.layer_1(pooler)
        pooler = self.activation_function(pooler)
        pooler = self.dropout_1(pooler)

        pooler = self.layer_2(pooler)
        pooler = self.activation_function(pooler)
        pooler = self.dropout_2(pooler)

        pooler = self.classifier(pooler)
        # Apply softmax to final layer. 
        output = self.ac_final(pooler)
        return output

class ModelFineTuner:

    def __init__(self, model: torch.nn.Module, 
                 loss_function, optimizer, data_source: pd.DataFrame, 
                 train_batch_size: int, test_batch_size: int, 
                 tokenizer,
                 data_limit: None or int = None, 
                 testing: bool= False):
        self.model = model 
        self.loss_function = loss_function 
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.testing = testing

        if data_limit is not None:
            data_source = data_source.head(data_limit)
        self.training_loader, self.testing_loader = self.initialize_dataloaders(data_source)
        
    
    def initialize_dataloaders(self, data_source: pd.DataFrame) -> Tuple[DataLoader]:
        train_size = 0.9
        data_source['datetime'] = data_source['date'].map(lambda x: parser.parse(x))
        data_source.sort_values(by='datetime', inplace=True)
        data_source.drop('datetime', axis=1, inplace=True)

        data_points = len(data_source.index)
        cutoff = int(data_points*train_size)
        train_data = data_source[:cutoff].reset_index(drop=True)
        test_data = data_source[cutoff:].reset_index(drop=True)

        print(f"Train Dates: {train_data.iloc[0]['date']} to {train_data.iloc[-1]['date']}.")
        print(f"Test Dates: {test_data.iloc[0]['date']} to {test_data.iloc[-1]['date']}.")
        
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

        training_set = HeadlineData(train_data, self.tokenizer, MAX_LEN)
        testing_set = HeadlineData(test_data, self.tokenizer, MAX_LEN)

        training_loader = DataLoader(training_set, **train_params)
        testing_loader = DataLoader(testing_set, **test_params)

        if self.testing:
            train_params['shuffle'] = False

        return training_loader, testing_loader

    def train(self, epoch: int):
        tr_loss = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        true_values = [] 
        predicted_values = []
        self.model.train()

        if self.testing: 
            loaded_data = enumerate(self.training_loader, 0)
        else:
            loaded_data = tqdm(enumerate(self.training_loader, 0), total=len(self.training_loader))

        for _, data in loaded_data:
            # Input data for the model
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            historical_data = data['stock_data'].to(device, dtype=torch.long)

            # Pass through, compute loss. 
            outputs = self.model(ids, mask, token_type_ids, historical_data)
            loss = self.loss_function(outputs, targets)
            tr_loss += loss.item()
            confidence_values, choices = torch.max(outputs.data, dim=1)

            # Add predictions and results to lists. 
            add_true_and_pred_values(choices, targets, true_values, predicted_values)

            nb_tr_steps += 1 
            nb_tr_examples += targets.size(0)

            if _%1000==999:
                loss_step = tr_loss/nb_tr_steps 
                accu_step = accuracy_score(true_values, predicted_values) * 100
                print(f"Training Loss per 1000 steps:  {loss_step}")
                print(f"Training Accuracy per 1000 steps: {accu_step}")
            # Perform backpropagation. 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        epoch_accu = accuracy_score(true_values, predicted_values)
        print(f'The Total Accuracy for Epoch {epoch}: {epoch_accu * 100}')
        epoch_loss = tr_loss/nb_tr_steps 
        print(f"Training Loss Epoch {epoch}: {epoch_loss}")
        print("\n")

        return true_values

    def valid(self, epoch: int):
        self.model.eval()

        tr_loss = 0
        nb_tr_steps = 0 
        nb_tr_examples= 0 

        true_values = []
        predicted_values = []

        if self.testing: 
            loaded_data = enumerate(self.training_loader, 0)
        else:
            loaded_data = tqdm(enumerate(self.testing_loader, 0), total=len(self.testing_loader))

        with torch.no_grad():
            for _, data in loaded_data:
                # Define inputs to the model. 
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.long)
                historical_data = data['stock_data'].to(device, dtype=torch.long)
                outputs = self.model(ids, mask, token_type_ids, historical_data)

                # Got error on colab so just skip if it happens. 
                try:
                    loss = self.loss_function(outputs, targets)
                except:
                    print("Got error in loss function, skipping data point.")
                    continue

                tr_loss += loss.item()
                confidences, choices = torch.max(outputs, 1)

                nb_tr_steps += 1
                nb_tr_examples += targets.size(0) 

                add_true_and_pred_values(choices, targets, true_values, predicted_values)

                if _% 1000 == 999:
                    loss_step = tr_loss / nb_tr_steps 
                    accu_step = accuracy_score(true_values, predicted_values) * 100
                    print(f"Validation Loss per 1000 steps: {loss_step}")
                    print(f"Validation Accuracy per 1000 steps: {accu_step}")

        epoch_conf_matrix = confusion_matrix(true_values, predicted_values, labels=[0, 1, 2])
        epoch_loss = tr_loss / nb_tr_steps 
        epoch_accu = accuracy_score(true_values, predicted_values)
        epoch_f1 = f1_score(true_values, predicted_values, average="micro", zero_division=0)
        epoch_prec = precision_score(true_values, predicted_values, average="micro", zero_division=0)
        epoch_recall = recall_score(true_values, predicted_values, average="micro", zero_division=0)
        labels = ['Increasing', 'Decreasing', 'Neutral']

        # This line throws an error if only labels 0-2 are present. 
        try: 
            report = classification_report(true_values, predicted_values, target_names=labels, labels=[0, 1, 2], zero_division=0)
            report_dict = classification_report(true_values, predicted_values, target_names=labels, labels=[0, 1, 2], output_dict=True, zero_division=0)
        except ValueError:
            report_dict = {}

        print(report)
        print(f"Validation Loss Epoch {epoch}: {epoch_loss}")
        print(f"Validation Accuracy Epoch {epoch}: {epoch_accu}")
        print(f"Validation Precision Epoch {epoch}: {epoch_prec}")
        print(f"Validation Recall Epoch {epoch}: {epoch_recall}")
        print(f"F1 score for epoch {epoch}: {epoch_f1}")
        print(epoch_conf_matrix)
        print("\n")

        return report_dict, epoch_loss

def flatten_report(report: Dict[str, str or Dict[str, float]]):
    labels = ['Increasing', 'Decreasing', 'Neutral', 'macro avg', 'micro avg', 'weighted avg']
    data = {}
    for key, value in report.items():
        if key in labels:
            for subkey, subvalue in value.items():
                data[f"{key}_{subkey}"] = subvalue
        else:
            data[key] = value 
    # In testing, we sometimes get accu = 0, in which case it isnt in the dict. 
    if 'accuracy' not in data:
        data['accuracy'] = 0.0
    return data

def main():
    args = get_model_args()
    tokenizer, model_source, model_embedding_size = get_model(args.model_type)
    model = ModelClass(model_source, model_embedding_size, is_distill=args.model_type == 'distill', freeze=args.freeze_model) 
    model.to(device)

    # Load in existing weights if specified. 
    if args.starting_weights is not None:
        starting_epoch, weights = parse_weights_file(f'weights/{args.starting_weights}')
        model.load_state_dict(torch.load(weights))
        training_data = list(pd.read_csv(args.stats_filepath).T.to_dict().values())
        stats_filename = args.stats_filepath
    else:
        starting_epoch = 0
        training_data = []
        stats_filename = args.stats_filename

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.learning_rate, weight_decay=1e-03)
    print(f"Running model with learning rate {args.learning_rate} and train batch size {args.train_batch_size}")
    
    tweet_data_path = '../data/processed_tweet_data/tweet-data.csv'
    # This is due to chunking the files. 
    headline_data_path = '../data/processed_headline_data/'
    kaggle_data_path = '../data/processed_kaggle_data/'

    headline_matched_prefix = '<=2022-03-01' 
    kaggle_matched_prefix = '<=2019-06-01' 

    if args.data_source == 'tweet':
        df = get_train_data(tweet_data_path)
    elif args.data_source == 'all':
        tweet_df = get_train_data(tweet_data_path)
        headline_df = read_in_chunked_data(headline_data_path, headline_matched_prefix)
        df = pd.concat([tweet_df, headline_df], ignore_index=True, sort=False)
    elif args.data_source == 'kaggle':
        df = read_in_chunked_data(kaggle_data_path, kaggle_matched_prefix)
    else:
        df = read_in_chunked_data(headline_data_path, headline_matched_prefix)
 
    ModelTrainer = ModelFineTuner(model=model, loss_function=loss_function, optimizer=optimizer, 
                                data_source=df, tokenizer=tokenizer,
                                train_batch_size= args.train_batch_size, test_batch_size =args.test_batch_size, 
                                data_limit=args.data_limit, testing=args.testing)
    
    for epoch in range(starting_epoch+1, args.epochs+starting_epoch+1):
        ModelTrainer.train(epoch)

        report, loss = ModelTrainer.valid(epoch)
        flattened_report = flatten_report(report)
        flattened_report['loss'] = loss
        training_data.append(flattened_report)

        print("Saving model...")
        model_str = "".join([x for x in f"{args.model_type}:{args.learning_rate}:{args.train_batch_size}" if x != '.'])
        output_model_file = f'+{model_str}_{epoch}.pt'
        weights_dir = os.path.join(args.output_dir, 'weights/')
        
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)
            
        torch.save(model.state_dict(), os.path.join(weights_dir, output_model_file))
        print("Saving statistics...")
        training_df = pd.DataFrame(training_data)
        training_df.to_csv(os.path.join('./run_summaries/', stats_filename), index=False)
        print("all files saved.")

    plot_training_data(training_df, args)
    plot_loss_data(training_df, args)
if __name__ == '__main__':
    main()
