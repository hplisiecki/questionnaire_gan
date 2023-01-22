import torch
from utils import data_mixer
import wandb
from dataloader import real_dataloader
import wandb

def training_loop(generator, discriminator, epochs, batch_size, device, save_dir,
                    optimizer_generator, optimizer_discriminator, criterion, scheduler,
                  critic_range):

    # random vector for generator input
    for epoch in range(epochs):
        real_batches = real_dataloader(batch_size, device)
        for n_critic in range(critic_range):
            for real_surveys, real_scales in real_batches:
                z = torch.randn(batch_size, 100, 1, 1, device=device)

                # generate some fake data
                fake_data = generator(z, real_scales)
                # train discriminator on fake data
                mixed_data, labels = data_mixer(real_surveys, fake_data)

                discriminator_output = discriminator(mixed_data, real_scales)
                discriminator_loss = criterion(discriminator_output, labels)
                discriminator_loss.backward()
                optimizer_discriminator.step()
                optimizer_discriminator.zero_grad()

        # train generator
        for real_batch in real_dataloader:
            z = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_data = generator(z)
            mixed_data, labels = data_mixer(real_batch, fake_data)
            # reverse labels
            labels = [1 if label == 0 else 0 for label in labels]
            discriminator_output = discriminator(mixed_data)
            generator_loss = criterion(discriminator_output, labels)

            generator_loss.backward()
            optimizer_generator.step()
            optimizer_generator.zero_grad()
            scheduler.step()

        if epoch % 2 == 0:
        wandb.log({"Generator Loss": generator_loss,
                   "Discriminator Loss": discriminator_loss})

        if epoch % 10 == 0:
            torch.save(generator.state_dict(), save_dir)
            torch.save(discriminator.state_dict(), save_dir)





