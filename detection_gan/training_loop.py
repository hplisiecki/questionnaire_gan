import torch
from data_mixer import data_mixer
import wandb
from dataloader import real_dataloader
import wandb
from tqdm import tqdm
from utils import box_numbers
def training_loop(generator, discriminator, epochs, batch_size, device, save_dir,
                    optimizer_generator, optimizer_discriminator, criterion, scheduler_gen, scheduler_disc,
                  critic_range):

    # random vector for generator input
    for epoch in range(epochs):
        real_batches = real_dataloader(batch_size)
        for n_critic in tqdm(range(critic_range)):
            for idx, (inputs, scales) in enumerate(real_batches):
                inputs = torch.tensor(inputs).to(device)
                real_scales = torch.tensor(scales).to(device).to(torch.float32)

                z = torch.randn(batch_size, 100).to(device)

                # generate some fake data
                fake_data = generator(z, real_scales)
                print(fake_data.shape)
                print(torch.flatten(real_scales).shape)
                new_scale = torch.cat([torch.flatten(i.repeat(100, 1)) for i in real_scales]).to(device)
                fake_data = box_numbers(torch.flatten(fake_data), new_scale)

                fake_data = fake_data.view(batch_size, -1, 40)
                mixed_data, labels = data_mixer(inputs, fake_data, device)
                mixed_data = mixed_data.to(torch.float32)
                double_scales = torch.cat((real_scales, real_scales), dim=0).view(-1, 1, 40)

                discriminator_output = discriminator(mixed_data, double_scales)
                labels = labels.to(torch.float32)
                discriminator_loss = criterion(discriminator_output, labels)
                discriminator_loss.backward()
                optimizer_discriminator.step()
                optimizer_discriminator.zero_grad()
                if idx % n_critic == 0:
                    scheduler_disc.step()


        # train generator
        for inputs, scales in real_batches:
            # to tensor, to device
            inputs = torch.tensor(inputs).to(device)
            real_scales = torch.tensor(scales).to(device).to(torch.float32)

            z = torch.randn(batch_size, 100).to(device)
            fake_data = generator(z, real_scales)
            fake_data = fake_data.view(batch_size, -1, 40)
            mixed_data, labels = data_mixer(inputs, fake_data, device)
            mixed_data = mixed_data.to(torch.float32)
            double_scales = torch.cat((real_scales, real_scales), dim=0).view(-1, 1, 40)
            # reverse labels
            labels = torch.abs(labels - 1)
            labels = labels.to(torch.float32)

            discriminator_output = discriminator(mixed_data, double_scales)
            generator_loss = criterion(discriminator_output, labels)

            generator_loss.backward()
            optimizer_generator.step()
            optimizer_generator.zero_grad()
            scheduler_gen.step()

        if epoch % 2 == 0:
            wandb.log({"Generator Loss": generator_loss,
                       "Discriminator Loss": discriminator_loss})

        if epoch % 10 == 0:
            torch.save(generator.state_dict(), save_dir)
            torch.save(discriminator.state_dict(), save_dir)





