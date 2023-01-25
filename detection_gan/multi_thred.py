import sys
sys.path.insert(0, r'D:\GitHub\questionnaire_gan\detection_gan')
from dataloader import real_dataloader
import pickle
import os
import multiprocessing


if __name__ == '__main__':

    batch_size = 1000
    data_splits = 40

    for epoch_range in range(data_splits):
        pool = multiprocessing.Pool(2)
        epoch_data_list = pool.map(real_dataloader, [batch_size] * 10)
        pool.close()
        del pool
        epoch_data_list = list(epoch_data_list)
        data_dir = r'D:\data\data_hackaton\gan_data'

        with open(os.path.join(data_dir, f'data_{epoch_range}.pkl'), 'wb') as f:
            pickle.dump(epoch_data_list, f)
43