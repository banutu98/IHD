import os
import pandas as pd

def get_csv_train(data_prefix='./data', verbose=False):
    train_df = pd.read_csv(os.path.join(data_prefix, 'stage_1_train.csv'))
    train_df[['ID', 'subtype']] = train_df['ID'].str.rsplit('_', 1,
                                                            expand=True)
    train_df = train_df.rename(columns={'ID': 'id', 'Label': 'label'})
    train_df = pd.pivot_table(train_df, index='id',
                              columns='subtype', values='label')
    return train_df
