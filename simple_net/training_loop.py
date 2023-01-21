import torch
import wandb

# training loop for binary classification with one dimensional output
def training_loop(train_dataloader, test_datalaoder, model, criterion, optimizer, scheduler, epochs, device, savedir):
    for epoch in range(epochs):
        total_loss_train = 0
        best_acc = 0
        # to device
        model.train()
        for train_input, train_label in train_dataloader:
            train_label = train_label.to(device).long()
            # train input are vectors
            train_input = train_input.to(device).float()
            output = model(train_input)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()


            del output, train_label, input_id, mask

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
        total_acc_val = 0
        total_loss_val = 0
        model.eval()
        with torch.no_grad():
            for val_input, val_label in test_datalaoder:
                val_label = val_label.to(device).long()
                val_input = val_input.to(device).float()

                output = model(val_input)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item() / len(val_label)

                total_acc_val += acc

        if total_acc_val / len(test_datalaoder) > best_acc:
            best_acc = total_acc_val / len(test_datalaoder)
            torch.save(model.state_dict(), savedir)

        wandb.log({"Train Loss": total_loss_train / len(train_dataloader),
                   "Validation Loss": total_loss_val / len(test_datalaoder),
                   "Validation Accuracy": total_acc_val / len(test_datalaoder)})

        # print results
        print(f"Epoch {epoch + 1} of {epochs} / Train Loss: {total_loss_train / len(train_dataloader)} "
              f"/ Validation Loss: {total_loss_val / len(test_datalaoder)} / Validation Accuracy: "
              f"{total_acc_val / len(test_datalaoder)}")
