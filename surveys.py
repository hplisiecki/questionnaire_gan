import numpy as np
import os
surveys_dir = r'D:\data\surveys'
survey_list = []
length_survey_list = []
length_answer_list = []
for file in os.listdir(surveys_dir):
    for file_2 in os.listdir(os.path.join(surveys_dir, file)):
        if 'scale' in file_2:
            scale_array = np.load(os.path.join(surveys_dir, file, file_2), allow_pickle=True)

            file_2 = file_2.replace('_scales.npy', '.npy')
            data_array = np.load(os.path.join(surveys_dir, file, file_2), allow_pickle=True)
            
            survey_list.append((data_array, scale_array))
            
            if len(data_array.shape) >1:
                length_survey_list.append(data_array.shape[1])
            else:
                length_survey_list.append("1")
            length_answer_list.append(data_array.shape[0])
            

# save
import pickle
with open(r'D:\data\surveys\survey_list.pickle', 'wb') as handle:
    pickle.dump(survey_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(r'D:\data\surveys\length_survey_list.pickle', 'wb') as handle:
    pickle.dump(length_survey_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(r'D:\data\surveys\length_answer_list.pickle', 'wb') as handle:
    pickle.dump(length_answer_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

