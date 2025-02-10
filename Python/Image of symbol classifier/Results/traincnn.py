import torch

def train_cnn(model, dataloaders, datasets, criterion, optimizer, epochs, save_path):
    # Put model in gpu if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f'Model is on device: {next(model.parameters()).device}')

    _, _, classes = datasets[0]._get_names()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        # Loop through batch
        for item, label in iter(dataloaders[0]):
            # Put data in the same device as the model
            item = item.to(device)
            label = label.to(device)
            # Set the gradient to zero
            optimizer.zero_grad()

            # Run model on input, calculate error and update weights
            output = model(item)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * item.size(0)

            _, predicted = torch.max(output, 1)
            correct_train += (predicted == label).sum().item()

        train_loss /= len(datasets[0])
        train_losses.append(train_loss)
        train_accuracy = 100 * correct_train/len(datasets[0])
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0
        correct_val = 0
        with torch.no_grad():
            for item, label in iter(dataloaders[1]):
                item = item.to(device)
                label = label.to(device)

                output = model(item)
                loss = criterion(output, label)
                val_loss += loss.item() * item.size(0)

                _, predicted = torch.max(output, 1)
                correct_val += (predicted == label).sum().item()

            val_loss /= len(datasets[1])
            val_losses.append(val_loss)
            val_accuracy = 100 * correct_val/len(datasets[1])
            val_accuracies.append(val_accuracy)


        print(f"Epoch {epoch+1}/{epochs}, Train_loss: {train_loss}, Train_acc: {train_accuracy}, Val_loss: {val_loss}, Val_acc: {val_accuracy}")
    torch.save(model.state_dict(), save_path)
    #, val_losses, val_accuracies
    return train_losses, train_accuracies
