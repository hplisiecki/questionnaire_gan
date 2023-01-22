import torch
from data_mixer import data_mixer
import wandb
from dataloader import real_dataloader
import wandb
from tqdm import tqdm

def training_loop(generator, discriminator, epochs, batch_size, device, save_dir,
                    optimizer_generator, optimizer_discriminator, criterion, scheduler_gen, scheduler_disc,
                  critic_range):

    # random vector for generator input
    for epoch in range(epochs):
        real_batches = real_dataloader(batch_size)
        for n_critic in tqdm(range(critic_range)):
            for inputs, scales in real_batches:
                inputs = torch.tensor(inputs).to(device)
                real_scales = torch.tensor(scales).to(device).to(torch.float32)

                z = torch.randn(batch_size, 100).to(device)

                # generate some fake data
                fake_data = generator(z, real_scales)

                fake_data = fake_data.view(batch_size, -1, 40)
                mixed_data, labels = data_mixer(inputs, fake_data, device)

                discriminator_output = discriminator(mixed_data, real_scales)
                discriminator_loss = criterion(discriminator_output, labels)
                discriminator_loss.backward()
                optimizer_discriminator.step()
                optimizer_discriminator.zero_grad()
                scheduler_disc.step()


        # train generator
        for real_surveys, real_scales in tqdm(real_dataloader):
            # to tensor, to device
            real_surveys = torch.tensor(real_surveys).to(device)
            real_scales = torch.tensor(real_scales).to(device)
            z = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_data = generator(z)
            mixed_data, labels = data_mixer(real_surveys, fake_data)
            # reverse labels
            labels = [1 if label == 0 else 0 for label in labels]
            discriminator_output = discriminator(real_surveys, real_scales)
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





