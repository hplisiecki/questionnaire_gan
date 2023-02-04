import sys
sys.path.insert(0, r'D:\GitHub\questionnaire_gan\detection_gan')
import torch
from data_mixer import data_mixer
from dataloader import real_dataloader
import wandb
from tqdm import tqdm
from utils import box_numbers
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def training_loop(generator, discriminator, epochs, batch_size, device, save_dir,
                    optimizer_generator, optimizer_discriminator, criterion, scheduler_gen, scheduler_disc,
                  critic_range, mimic_range):

    generator.train()
    discriminator.train()

    # random vector for generator input
    for epoch in range(epochs):
        data = real_dataloader()
        # batch the data
        real_batches = [(data[0][i:i + batch_size], data[1][i:i + batch_size]) for i in range(0, len(data[0]), batch_size)]
        total_loss_disc = 0
        total_loss_gen = 0
        total_disc_acc = 0

        # set generator to eval mode
        for inputs, scales in tqdm(real_batches):
            # zero the parameter gradients
            discriminator.zero_grad()
            discriminator.requires_grad_(True)
            generator.requires_grad_(False)

            inputs = torch.tensor(inputs).to(device)
            real_scales = torch.tensor(scales).to(device).to(torch.float32)

            z = torch.randn(batch_size, 100, device = device)
            # z = z.view(batch_size, -1, 1)

            # generate some fake data
            fake_data = generator(z, real_scales)
            fake_data = fake_data.view(batch_size, 100, 40, -1)
            # softmax
            fake_data = torch.nn.functional.softmax(fake_data, dim=3)
            # get max

            mixed_data, labels = data_mixer(inputs, fake_data, device)
            mixed_data = mixed_data.to(torch.float32)
            labels = labels.to(torch.float32)

            discriminator_output = discriminator(mixed_data)
            del mixed_data, inputs, fake_data, z
            # where_not_padding = torch.where(mixed_data.sum(dim=2).view(-1) > 0)[0]
            discriminator_output = discriminator_output.view(-1)
            # discriminator_output = discriminator_output[where_not_padding]
            labels = labels.view(-1)
            # change 2 in labels to 1
            labels = torch.where(labels == 2, torch.ones_like(labels), labels)
            # labels = labels[where_not_padding]
            discriminator_loss = criterion(discriminator_output, labels)


            acc = torch.sum(torch.round(discriminator_output) == labels) / len(labels)
            del labels, discriminator_output
            total_disc_acc += acc.item()
            total_loss_disc += discriminator_loss.item()
            discriminator_loss.backward()
            optimizer_discriminator.step()
            optimizer_discriminator.zero_grad()
            scheduler_disc.step()

        for inputs, scales in tqdm(real_batches):
            # to tensor, to device
            generator.zero_grad()
            inputs = torch.tensor(inputs).to(device)

            real_scales = torch.tensor(scales).to(device).to(torch.float32)

            z = torch.randn(batch_size, 100, device = device)
            fake_data = generator(z, real_scales)
            fake_data = fake_data.view(batch_size, 100, 40, 21)
            # softmax
            fake_data = torch.nn.functional.softmax(fake_data, dim=3)
            # get max

            mixed_data, labels = data_mixer(inputs, fake_data, device)
            mixed_data = mixed_data.to(torch.float32)

            mixed_data = mixed_data.view(-1, 100, 40, 21)

            mixed_data = mixed_data.to(torch.float32)
            label_zeros = torch.where(labels != 2)[0]
            discriminator_output = discriminator(mixed_data)

            discriminator_output = discriminator_output.view(-1)

            fake_discriminator = discriminator_output[label_zeros]

            labels = torch.ones(fake_discriminator.shape[0]).to(device)

            generator_loss = criterion(fake_discriminator, labels)
            total_loss_gen += generator_loss.item()

            generator_loss.backward()
            optimizer_generator.step()
            optimizer_generator.zero_grad()
            scheduler_gen.step()

        if epoch % 1 == 0:
            wandb.log({"Generator Loss": total_loss_gen / (len(real_batches) * mimic_range),
                       "Discriminator Loss": total_loss_disc / (len(real_batches) * critic_range),
                        "Discriminator Accuracy": total_disc_acc / (len(real_batches) * critic_range),
                       "Discriminator Learning Rate": optimizer_discriminator.param_groups[0]['lr'],
                      "Generator Learning Rate": optimizer_generator.param_groups[0]['lr'],
                       "Epoch": epoch})

            print(f'Epoch: {epoch}, Generator Loss: {total_loss_gen / (len(real_batches) * mimic_range)}, Discriminator Loss: {total_loss_disc / (len(real_batches) * critic_range)}, Discriminator Accuracy: {total_disc_acc / (len(real_batches) * critic_range)}')


        if epoch % 10 == 0:
            torch.save(generator.state_dict(), save_dir + 'generator')
            torch.save(discriminator.state_dict(), save_dir + 'discriminator')

            mixed_data = torch.argmax(mixed_data, 3)
            fake_data = torch.argmax(fake_data, 3)
            inputs = torch.argmax(inputs, 3)
            inputs = inputs.view(-1, 40)
            inputs = inputs[inputs.sum(1) != 0]
            mixed_data = mixed_data.view(-1, 40)
            mixed_data = mixed_data[mixed_data.sum(1) != 0]



            fig, axs = plt.subplots(3,3)
            cnt = 0
            for ax_ in axs:
                for ax in ax_:
                    data = mixed_data[0:100][:, cnt].cpu().detach().numpy()
                    data = data.tolist()
                    while data[-1] == 0:
                        data = data[:-1]
                    ax.hist(data)
                    cnt+=1

            fig.savefig(fr'D:\GitHub\questionnaire_gan\detection_gan\charts_cnn\mixed_hist_{epoch}')
            plt.close()
            fig, axs = plt.subplots(3,3)
            cnt = 0
            for ax_ in axs:
                for ax in ax_:
                    data = fake_data[0][:, cnt].cpu().detach().numpy()
                    data = data.tolist()
                    while data[-1] == 0:
                        data = data[:-1]
                    ax.hist(data)
                    cnt+=1

            fig.savefig(fr'D:\GitHub\questionnaire_gan\detection_gan\charts_cnn\fake_hist_{epoch}')
            plt.close()









