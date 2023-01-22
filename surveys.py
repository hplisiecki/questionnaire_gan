
import numpy as np
import pandas as pd

s1 = np.load("Surveys/survey_2/responses.npy")


def sampling_from_survey(df):
    df = pd.DataFrame(df)
    n_items = np.random.randint(1, s1.shape[1], size=1)
    n_rows  = np.random.randint(1, s1.shape[0], size=1)
    df = df.sample(n = n_rows, replace = False, axis=0)
    df = df.sample(n = n_items, replace = False, axis=1)
    
    return df

import os
scalesdf = pd.DataFrame()

surveys_dir = "Surveys"
survey_list = []
length_survey_list = []
length_answer_list = []
for file in os.listdir(surveys_dir):
    for file_2 in os.listdir(os.path.join(surveys_dir, file)):
        if 'scale' in file_2:
            scale_array = np.load(os.path.join(surveys_dir, file, file_2))
            
            file_2 = file_2.replace('_scale.npy', '.npy')
            data_array = np.load(os.path.join(surveys_dir, file, file_2))
            
            survey_list.append((data_array, scale_array))
            
            if len(data_array.shape) >1:
                length_survey_list.append(data_array.shape[1])
            else: 
                length_survey_list.append("1")
            length_answer_list.append(data_array.shape[0])
            

# llow picke = tru