import sys, os
sys.path.insert(r'D:\GitHub\ergodicity_1991\simple_net')

import Dataset, Simple_Net from dataset_and_model
import pandas as pd
from training_loop import training_loop

df = pd.read_csv()



start_layer = 300
layer_list = [100, 100, 100]

model = Simple_Net(dropout = 0.3, layer_list)

training_loop(train_dataloader, test_datalaoder, model, criterion, optimizer, scheduler, epochs, device, savedir)
