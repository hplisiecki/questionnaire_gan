import sys, os
import torch
sys.path.insert(0, r'D:\GitHub\ergodicity_1991\simple_net')
from dataset_and_model import Dataset, Simple_Net
from training_loop import training_loop
from preparing_data import prepare_data
import wandb
from transformers import get_linear_schedule_with_warmup

df_train, df_val, df_test, label_dict = prepare_data()

train, val, test = Dataset(df_train), Dataset(df_val), Dataset(df_test)

epochs = 100
layer_list = [400, 400, 400, 400, 400, 400, 400]
input_size = 500 * 40
savedir = r'D:\data\data_hackaton\models\test'
batch_size = 800
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)


model = Simple_Net(0.3, input_size, layer_list, len(label_dict))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),
                  lr=5e-4,
                  eps=1e-8,  # Epsilon
                  weight_decay=0.3,
                  amsgrad=True,
                  betas = (0.9, 0.999))

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=500,
                                            num_training_steps= len(train_dataloader) * epochs)
# model to device
model = model.to(device)

wandb.init(project="simple_net", entity="hubertp")
wandb.watch(model, log_freq=5)

training_loop(train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, epochs, device, savedir)
