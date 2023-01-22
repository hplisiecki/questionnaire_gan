import numpy as np
import pandas as pd
import csv
import os
import glob
import pathlib
from scipy.io import readsav
from sklearn import preprocessing

def get_delimiter(file_path, bytes = 4096):
    sniffer = csv.Sniffer()
    data = open(file_path, "r").read(bytes)
    delimiter = sniffer.sniff(data).delimiter
    return delimiter      

def cleanup_frame(df):
    scales = []
    df.dropna(how='all', inplace=True)
    for col in df.columns:
        le = preprocessing.LabelEncoder()
        df[col] = le.fit_transform(df[col])
        if len(df[col].unique()) > 20:
            df.drop(col, axis=1, inplace=True)
        else:
            scales.append(len(df[col].unique()))
    return df.to_numpy(), np.array(scales)

def process_files(path):
    for dir in os.listdir(path):
        for file in pathlib.Path(path+dir).glob('*.csv'):
            
            try:
                df = pd.read_csv(file, sep=get_delimiter(file), on_bad_lines = 'warn')
                df, scales = cleanup_frame(df)
            except:
                pass
            np.save(str(file)[:-4]+'.npy', df)
            np.save(str(file)[:-4] + '_scales'+'.npy', scales)
        for file in glob.glob(dir+'\*.sav'):
            try:
                df = scipy.io.readsav(file, idict=None, python_dict=False, uncompressed_file_name=None, verbose=False)
                df, scales = cleanup_frame(df)
            except:
                pass
            np.save(str(file)[:-4]+'.npy', df)
            np.save(str(file)[:-4] + '_scales'+'.npy', scales)
            

process_files(r'C:\Users\lniedzwiedzki\Desktop\ergodicity_1991\surveys\\')