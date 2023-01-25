import sys
sys.path.insert(0, r'D:\GitHub\questionnaire_gan\detection_gan')
from dataloader import real_dataloader
import pickle
import os
from tqdm import tqdm

batch_size = 1000
epochs = 100
epoch_data_list = []
for epoch in tqdm(range(epochs)):
    real_batches = real_dataloader(batch_size)
    epoch_data_list.append(real_batches)


data_dir = r'D:\data\data_hackaton\gan_data'

with open(os.path.join(data_dir, 'data.pkl'), 'wb') as f:
    pickle.dump(epoch_data_list, f)