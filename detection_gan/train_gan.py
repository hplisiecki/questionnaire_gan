import sys
sys.path.insert(0, r'D:\GitHub\ergodicity_1991\detection_gan')
from training_loop import training_loop
from generator_discriminator import Generator, Discriminator
from transformers import get_linear_schedule_with_warmup
import wandb
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

layer_sizes = [100, 500, 1000, 2000, 4000]

generator = Generator(layer_sizes, 40)

layer_sizes = [4000, 2000, 500, 100, 1]
discriminator = Discriminator(layer_sizes, 40)

generator = generator.to(device)
discriminator = discriminator.to(device)

optimizer_generator = torch.optim.AdamW(generator.parameters(),
                    lr=5e-4,
                    eps=1e-8,  # Epsilon
                    weight_decay=0.3,
                    amsgrad=True,
                    betas = (0.9, 0.999))

optimizer_discriminator = torch.optim.AdamW(discriminator.parameters(),
                    lr=5e-4,
                    eps=1e-8,  # Epsilon
                    weight_decay=0.3,
                    amsgrad=True,
                    betas = (0.9, 0.999))

criterion = torch.nn.BCELoss()
critic_range = 3
save_dir = r'D:\GitHub\ergodicity_1991\detection_gan\models\test_1'
epochs = 10
batch_size = 1000

scheduler_generator = get_linear_schedule_with_warmup(optimizer_generator,
                                            num_warmup_steps=500,
                                            num_training_steps= 10000 * epochs)

scheduler_discriminator = get_linear_schedule_with_warmup(optimizer_discriminator,
                                            num_warmup_steps=500,
                                            num_training_steps= 10000 * epochs)



wandb.init(project="detection_gan", entity="hubertp")
wandb.watch(generator, log_freq=5)
wandb.watch(discriminator, log_freq=5)

training_loop(generator, discriminator, epochs, batch_size, device, save_dir,
              optimizer_generator, optimizer_discriminator, criterion,
              scheduler_generator, scheduler_discriminator, critic_range)


