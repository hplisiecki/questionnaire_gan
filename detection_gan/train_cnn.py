import sys
sys.path.insert(0, r'D:\GitHub\questionnaire_gan\detection_gan')
from cnn_training_loop import training_loop
from generator_discriminator import CNN_Generator, CNN_Discriminator, Generator
from transformers import get_linear_schedule_with_warmup
import wandb
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



generator = CNN_Generator()
layer_sizes = [100, 8000, 4000]
# layer_sizes = [100, 2000, 4000]

generator = Generator(layer_sizes, 40)

# initialize weights to be normally distributed
# with mean 0 and standard deviation 0.02
generator.linear_0.weight.data.normal_(0, 0.02)
generator.linear_0.bias.data.fill_(0)
generator.linear_1.weight.data.normal_(0, 0.02)
generator.linear_1.bias.data.fill_(0)


discriminator = CNN_Discriminator()

generator = generator.to(device)
discriminator = discriminator.to(device)

optimizer_generator = torch.optim.AdamW(generator.parameters(),
                    lr=5e-5,
                    eps=1e-8,  # Epsilon
                    weight_decay=0.3,
                    amsgrad=True,
                    betas = (0.9, 0.999))

optimizer_discriminator = torch.optim.AdamW(discriminator.parameters(),
                    lr=5e-5,
                    eps=1e-8,  # Epsilon
                    weight_decay=0.3,
                    amsgrad=True,
                    betas = (0.9, 0.999))

# modified minimax loss

criterion = torch.nn.BCELoss()
critic_range = 1
mimic_range = 1
save_dir = r'D:\GitHub\questionnaire_gan\detection_gan\models\test_cnn_2'
epochs = 160
batch_size = 1000

# load
# generator.load_state_dict(torch.load(r'D:\GitHub\questionnaire_gan\detection_gan\models\test_cnn_1generator'))
# discriminator.load_state_dict(torch.load(r'D:\GitHub\questionnaire_gan\detection_gan\models\test_cnn_1discriminator'))

scheduler_generator = get_linear_schedule_with_warmup(optimizer_generator,
                                            num_warmup_steps=500,
                                            num_training_steps= 4 * 10 * epochs)

scheduler_discriminator = get_linear_schedule_with_warmup(optimizer_discriminator,
                                            num_warmup_steps=500,
                                            num_training_steps= 4 * 10 * epochs)



wandb.init(project="detection_gan", entity="hubertp")
wandb.watch(generator, log_freq=5)
wandb.watch(discriminator, log_freq=5)

training_loop(generator, discriminator, epochs, batch_size, device, save_dir,
              optimizer_generator, optimizer_discriminator, criterion,
              scheduler_generator, scheduler_discriminator, critic_range,
              mimic_range)


