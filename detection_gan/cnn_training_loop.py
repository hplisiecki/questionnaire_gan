import sys
sys.path.insert(0, r'D:\GitHub\questionnaire_gan\detection_gan')
import torch
from data_mixer import data_mixer
import wandb
from dataloader import load_data
import wandb
from tqdm import tqdm
from utils import box_numbers
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def training_loop(generator, discriminator, epochs, batch_size, device, save_dir,
                    optimizer_generator, optimizer_discriminator, criterion, scheduler_gen, scheduler_disc,
                  critic_range, mimic_range):

    # random vector for generator input
    epoch = 0
    for data_file in tqdm(range(epochs)):
        data_id = data_file % 40
        data_list = load_data(data_id)
        for idx in range(10):
            real_batches = data_list[idx]
            total_loss_disc = 0
            total_loss_gen = 0
            for n_critic in range(critic_range):
                for idx, (inputs, scales) in enumerate(real_batches):
                    # zero the parameter gradients
                    discriminator.zero_grad()

                    inputs = torch.tensor(inputs).to(device)
                    real_scales = torch.tensor(scales).to(device).to(torch.float32)

                    z = torch.randn(batch_size, 100).to(device)
                    # z = z.view(batch_size, -1, 1)

                    # generate some fake data
                    fake_data = generator(z, real_scales)
                    new_scale = torch.cat([torch.flatten(i.repeat(100, 1)) for i in real_scales]).to(device)
                    fake_data = box_numbers(torch.flatten(fake_data), new_scale)
                    zeros = torch.where(new_scale == 0)[0]
                    fake_data[zeros] = 0
                    fake_data = fake_data.view(batch_size, -1, 40)

                    mixed_data, labels = data_mixer(inputs, fake_data, device)
                    mixed_data = mixed_data.to(torch.float32)

                    discriminator_output = discriminator(mixed_data)
                    labels = labels.to(torch.float32)
                    discriminator_loss = criterion(discriminator_output, labels)
                    total_loss_disc += discriminator_loss.item()
                    discriminator_loss.backward()
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                    if idx % critic_range == 0:
                        scheduler_disc.step()


            # train generator
            for n_mimic in range(mimic_range):
                for idx, (inputs, scales) in enumerate(real_batches):
                    # to tensor, to device
                    generator.zero_grad()
                    inputs = torch.tensor(inputs).to(device)
                    real_scales = torch.tensor(scales).to(device).to(torch.float32)

                    z = torch.randn(batch_size, 100).to(device)
                    # z = z.view(batch_size, -1, 1)

                    fake_data = generator(z, real_scales)

                    fake_data = fake_data.view(batch_size, -1, 40)
                    new_scale = torch.cat([torch.flatten(i.repeat(100, 1)) for i in real_scales]).to(device)
                    fake_data = box_numbers(torch.flatten(fake_data), new_scale)
                    fake_data = fake_data.view(batch_size, -1, 40)
                    mixed_data, labels = data_mixer(inputs, fake_data, device)
                    mixed_data = mixed_data.to(torch.float32)
                    # reverse labels

                    labels = torch.ones(labels.shape[0], labels.shape[1]).to(device)
                    labels = labels.to(torch.float32)

                    discriminator_output = discriminator(mixed_data)
                    generator_loss = criterion(discriminator_output, labels)
                    total_loss_gen += generator_loss.item()

                    generator_loss.backward()
                    optimizer_generator.step()
                    optimizer_generator.zero_grad()
                    if idx % mimic_range == 0:
                        scheduler_gen.step()

            if epoch % 1 == 0:
                wandb.log({"Generator Loss": total_loss_gen / len(real_batches),
                           "Discriminator Loss": total_loss_disc / (len(real_batches) * critic_range),
                           "Discriminator Learning Rate": optimizer_discriminator.param_groups[0]['lr'],
                          "Generator Learning Rate": optimizer_generator.param_groups[0]['lr'],
                           "Epoch": epoch})


            if epoch % 10 == 0:
                torch.save(generator.state_dict(), save_dir + 'generator')
                torch.save(discriminator.state_dict(), save_dir + 'discriminator')

                fig, axs = plt.subplots(9,9)
                cnt = 0
                for ax_ in axs:
                    for ax in ax_:
                        ax.hist(mixed_data[cnt][:,0].cpu().detach().numpy())
                        cnt+=1

                fig.savefig(fr'D:\GitHub\questionnaire_gan\detection_gan\charts_cnn\mixed_hist_{epoch}')
                plt.close()
                fig, axs = plt.subplots(9,9)
                cnt = 0
                for ax_ in axs:
                    for ax in ax_:
                        ax.hist(fake_data[cnt][:,0].cpu().detach().numpy())
                        cnt+=1

                fig.savefig(fr'D:\GitHub\questionnaire_gan\detection_gan\charts_cnn\fake_hist_{epoch}')
                plt.close()

            epoch += 1








