import numpy as np
import pandas as pd
import seaborn as sns
import argparse

def plot_training_data(data: pd.DataFrame, args: argparse.Namespace = None):
    plot_data = data[['accuracy', 'macro avg_precision', 'macro avg_recall' ,'macro avg_f1-score',
                      'weighted avg_precision','weighted avg_recall','weighted avg_f1-score']]
    if args is not None:
        title = f"{args.model_type}, Lr: {args.learning_rate}, trb: {args.train_batch_size}"
    else:
        title = 'Training Plot'
    
    sns.set_theme(style="whitegrid")
    fig = sns.lineplot(data=plot_data, palette="tab10", linewidth=1.0)
    fig.set_title(title)
    fig.get_figure().savefig(title)

if __name__ == '__main__':
    data_path = '../data/training_data.csv'
    df = pd.read_csv(data_path)
    plot_training_data(df)
