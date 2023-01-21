import torch
from utils import data_mixer
import wandb

def training_loop(generator, discriminator, real_dataloader, epochs, batch_size, device, savedir,
                    generator_optimizer, discriminator_optimizer, generator_criterion, discriminator_criterion,
                  critic_range):

    # random vector for generator input
    for epoch in epochs:
        for n_critic in range(critic_range):
            for batch in real_dataloader:
                z = torch.randn(batch_size, 100, 1, 1, device=device)

                # generate some fake data
                fake_data = generator(z)
                # train discriminator on fake data
                mixed_data, labels = data_mixer(real_data, fake_data)

                discriminator_output = discriminator(mixed_data)
                discriminator_loss = discriminator_criterion(discriminator_output, labels)
                discriminator_loss.backward()
                discriminator_optimizer.step()
                discriminator_optimizer.zero_grad()

        # train generator
        z = torch.randn(batch_size, 100, 1, 1, device=device)
        fake_data = generator(z)
        mixed_data, labels = data_mixer(real_data, fake_data)
        # reverse labels
        labels = [1 if label == 0 else 0 for label in labels]
        discriminator_output = discriminator(mixed_data)
        generator_loss = generator_criterion(discriminator_output, labels)

        generator_loss.backward()
        generator_optimizer.step()
        generator_optimizer.zero_grad()

        if epoch % 2 == 0:
        wandb.log({"Generator Loss": generator_loss,
                   "Discriminator Loss": discriminator_loss})

        if epoch % 10 == 0:
            torch.save(generator.state_dict(), savedir)
            torch.save(discriminator.state_dict(), savedir)





