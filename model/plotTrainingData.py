import numpy as np
import pandas as pd
import seaborn as sns

def plot_training_data(data: pd.DataFrame):

    plot_data = data[['accuracy', 'macro avg_precision', 'macro avg_recall' ,'macro avg_f1-score',
                      'weighted avg_precision','weighted avg_recall','weighted avg_f1-score']]
    sns.set_theme(style="whitegrid")
    fig = sns.lineplot(data=plot_data, palette="tab10", linewidth=1.0)
    fig.get_figure().savefig('test')

if __name__ == '__main__':
    data_path = '../data/training_data.csv'
    df = pd.read_csv(data_path)
    plot_training_data(df)
