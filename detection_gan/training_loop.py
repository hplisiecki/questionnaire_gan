import sys
sys.path.insert(0, r'D:\GitHub\questionnaire_gan\detection_gan')
import torch
from data_mixer import data_mixer
import wandb
from dataloader import real_dataloader
import wandb
from tqdm import tqdm
from utils import box_numbers
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def training_loop(generator, discriminator, epochs, batch_size, device, save_dir,
                    optimizer_generator, optimizer_discriminator, criterion, scheduler_gen, scheduler_disc,
                  critic_range):

    # random vector for generator input
    for epoch in range(epochs):
        total_loss_disc = 0
        total_loss_gen = 0
        real_batches = real_dataloader(batch_size)
        for n_critic in tqdm(range(critic_range)):
            for idx, (inputs, scales) in enumerate(real_batches):
                inputs = torch.tensor(inputs).to(device)
                real_scales = torch.tensor(scales).to(device).to(torch.float32)

                z = torch.randn(batch_size, 100).to(device)

                # generate some fake data
                fake_data = generator(z, real_scales)
                new_scale = torch.cat([torch.flatten(i.repeat(100, 1)) for i in real_scales]).to(device)
                fake_data = box_numbers(torch.flatten(fake_data), new_scale)
                fake_data = fake_data.view(batch_size, -1, 40)
                mixed_data, labels = data_mixer(inputs, fake_data, device)
                mixed_data = mixed_data.to(torch.float32)
                double_scales = torch.cat((real_scales, real_scales), dim=0).view(-1, 1, 40)

                discriminator_output = discriminator(mixed_data, double_scales)
                labels = labels.to(torch.float32)
                discriminator_loss = criterion(discriminator_output, labels)
                total_loss_disc += discriminator_loss.item()
                discriminator_loss.backward()
                optimizer_discriminator.step()
                optimizer_discriminator.zero_grad()
                if idx % critic_range == 0:
                    scheduler_disc.step()


        # train generator
        for inputs, scales in real_batches:
            # to tensor, to device
            inputs = torch.tensor(inputs).to(device)
            real_scales = torch.tensor(scales).to(device).to(torch.float32)

            z = torch.randn(batch_size, 100).to(device)
            fake_data = generator(z, real_scales)

            fake_data = fake_data.view(batch_size, -1, 40)
            new_scale = torch.cat([torch.flatten(i.repeat(100, 1)) for i in real_scales]).to(device)
            fake_data = box_numbers(torch.flatten(fake_data), new_scale)
            fake_data = fake_data.view(batch_size, -1, 40)
            mixed_data, labels = data_mixer(inputs, fake_data, device)
            mixed_data = mixed_data.to(torch.float32)
            double_scales = torch.cat((real_scales, real_scales), dim=0).view(-1, 1, 40)
            # reverse labels
            labels = torch.abs(labels - 1)
            labels = labels.to(torch.float32)

            discriminator_output = discriminator(mixed_data, double_scales)
            generator_loss = criterion(discriminator_output, labels)
            total_loss_gen += generator_loss.item()

            generator_loss.backward()
            optimizer_generator.step()
            optimizer_generator.zero_grad()
            scheduler_gen.step()

        if epoch % 1 == 0:
            wandb.log({"Generator Loss": total_loss_gen / len(real_batches),
                       "Discriminator Loss": total_loss_disc / (len(real_batches) * critic_range),
                       "Discriminator Learning Rate": optimizer_discriminator.param_groups[0]['lr'],
                      "Generator Learning Rate": optimizer_discriminator.param_groups[0]['lr']})


        if epoch % 10 == 0:
            torch.save(generator.state_dict(), save_dir + 'generator')
            torch.save(discriminator.state_dict(), save_dir + 'discriminator')

            fig, axs = plt.subplots(3,3)
            cnt = 0
            for ax_ in axs:
                for ax in ax_:
                    ax.hist(fake_data[cnt][:,0].cpu().detach().numpy())
                    cnt+1

            fig.savefig(fr'D:\GitHub\questionnaire_gan\detection_gan\charts\hist_{epoch}')
            plt.close()








