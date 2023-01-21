import pandas as pd
import numpy as np
import pickle
import os
def prepare_data():
    if not os.path.exists(r'D:\data\data_hackaton\label_dict.pickle'):
        df = pd.read_csv(r'D:\data\data_hackaton\dataset.csv')
        label_dict = {label: idx for idx, label in enumerate(df['Set'].unique())}
        df['label'] = df['Set'].map(label_dict)
        # del unnamed
        del df['Unnamed: 0'], df['Set']
        questionnaire_idx_list = []
        for idx, i in enumerate(range(0, len(df), 500)):
            questionnaire_idx_list.extend([idx]*500)
        df['questionnaire_idx'] = questionnaire_idx_list
        # split randomly
        unique_idx = df.questionnaire_idx.unique()
        np.random.shuffle(unique_idx)
        train_idx = unique_idx[:int(len(unique_idx)*0.8)]
        val_idx = unique_idx[int(len(unique_idx)*0.8):int(len(unique_idx)*0.9)]
        test_idx = unique_idx[int(len(unique_idx)*0.9):]
        train_df = df[df['questionnaire_idx'].isin(train_idx)]
        val_df = df[df['questionnaire_idx'].isin(val_idx)]
        test_df = df[df['questionnaire_idx'].isin(test_idx)]
        # reset index
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        # save
        train_df.to_csv(r'D:\data\data_hackaton\train.csv')
        val_df.to_csv(r'D:\data\data_hackaton\val.csv')
        test_df.to_csv(r'D:\data\data_hackaton\test.csv')
        # save label_dict
        with open(r'D:\data\data_hackaton\label_dict.pickle', 'wb') as handle:
            pickle.dump(label_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        train_df = pd.read_csv(r'D:\data\data_hackaton\train.csv')
        del train_df['Unnamed: 0']
        val_df = pd.read_csv(r'D:\data\data_hackaton\val.csv')
        del val_df['Unnamed: 0']
        test_df = pd.read_csv(r'D:\data\data_hackaton\test.csv')
        del test_df['Unnamed: 0']
        with open(r'D:\data\data_hackaton\label_dict.pickle', 'rb') as handle:
            label_dict = pickle.load(handle)


    return train_df, val_df, test_df, label_dict