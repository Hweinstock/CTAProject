import pandas as pd
import seaborn as sns
import argparse
import os

from typing import List, Any

PLOTS_DIR = 'plots/'

def plot_training_data(data: pd.DataFrame, args: argparse.Namespace = None):
    plot_data = data[['accuracy', 'macro avg_precision', 'macro avg_recall' ,'macro avg_f1-score',
                      'weighted avg_precision','weighted avg_recall','weighted avg_f1-score']]
    if args is not None:
        title = f"+{args.model_type}, Lr: {args.learning_rate}, trb: {args.train_batch_size}"
    else:
        title = 'Training Plot'
    output_path = os.path.join(PLOTS_DIR, title)
    sns.set_theme(style="whitegrid")
    fig = sns.lineplot(data=plot_data, palette="tab10", linewidth=1.0)
    fig.set_title(title)
    fig.get_figure().savefig(output_path)
    fig.get_figure().clf()

def plot_loss_data(data: pd.DataFrame, args: argparse.Namespace = None):
    plot_data2 = data[['loss']]
    if args is not None:
        title = f"+Loss:{args.model_type}, Lr: {args.learning_rate}, trb: {args.train_batch_size}"
    else:
        title = 'Training Plot'
    output_path = os.path.join(PLOTS_DIR, title)
    sns.set_theme(style='darkgrid')
    fig2 = sns.lineplot(data=plot_data2, palette='tab10', linewidth=2.0)
    fig2.set_title(title)
    fig2.get_figure().savefig(output_path)
    fig2.get_figure().clf()

def plot_confusion_matrix(matrix: List[Any], output_path: str = 'confusion_matrix') -> None:
    df_cm = pd.DataFrame(matrix, index = range(3),
                  columns = [i for i in range(3)])
    fig3 = sns.heatmap(df_cm, annot=True)
    fig3.get_figure().savefig(output_path)
    fig3.get_figure().clf()

if __name__ == '__main__':
    data_path = '../data/training_data.csv'
    df = pd.read_csv(data_path)
    plot_training_data(df)
