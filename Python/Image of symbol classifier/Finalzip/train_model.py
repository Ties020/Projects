import xml.etree.ElementTree as ET
import choose_data
from customdataclasses import torch
import custommodels
import torch.nn as nn
import argparse

def train_cnn(datatype, model, dataloaders, datasets, criterion, optimizer, epochs, save_path):
    # Put model in gpu if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f'Model is on device: {next(model.parameters()).device}')

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        # Loop through batches
        for item, label in iter(dataloaders[0]):
            # Put data in the same device as the model
            item = item.to(device)
            label = label.to(device)
            # Set the gradient to zero
            optimizer.zero_grad()

            # Run model on input, calculate error and update weights
            output = model(item)
            if datatype=='mathwriting':
                output = output.view(-1, 229)
                label = label.view(-1)
                label = label.long()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * item.size(0)

            _, predicted = torch.max(output, 1)
            if datatype=='mathwriting':
                predicted_reshaped = predicted.view(16, 4)
                labels_reshaped = label.view(16, 4)
                correct_train += (torch.all(predicted_reshaped == labels_reshaped, dim=1)).sum().item()
            elif datatype=='mnist':
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
                if datatype=='mathwriting':
                    output = output.view(-1, 229)
                    label = label.view(-1)
                    label = label.long()
                loss = criterion(output, label)
                val_loss += loss.item() * item.size(0)

                _, predicted = torch.max(output, 1)
                if datatype=='mathwriting':
                    predicted_reshaped = predicted.view(16, 4)
                    labels_reshaped = label.view(16, 4)
                    correct_train += (torch.all(predicted_reshaped == labels_reshaped, dim=1)).sum().item()
                elif datatype=='mnist':
                    correct_train += (predicted == label).sum().item()

            val_loss /= len(datasets[1])
            val_losses.append(val_loss)
            val_accuracy = 100 * correct_val/len(datasets[1])
            val_accuracies.append(val_accuracy)


        print(f"Epoch {epoch+1}/{epochs}, Train_loss: {train_loss}, Train_acc: {train_accuracy}, Val_loss: {val_loss}, Val_acc: {val_accuracy}")
    torch.save(model.state_dict(), save_path)
    return train_losses, train_accuracies, val_losses, val_accuracies

parser = argparse.ArgumentParser(description="Run Model")
parser.add_argument('--datatype', type=str, required=True, help='Which data to use: mnist or mathwriting')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--optim', type=str, required=True, help='Optimizer: sgd or adam')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--saveweights', type=str, required=True, help='Path to save the model weights')
args = parser.parse_args()

if args.datatype == 'mnist':
    trainset, trainloader, valset, valloader, _, _ = choose_data.load_data('mnist')
    cnn = custommodels.CNN1train(10, 'mnist')
elif args.datatype == 'mathwriting':
    symbols, trainset, trainloader, valset, valloader, _, _ = choose_data.load_data('mathwriting')
    cnn = custommodels.CNN3train(len(symbols._get_unique_labels()), 'mathwriting')
criterion = nn.CrossEntropyLoss()
if args.optim == 'sgd':
    optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr)
elif args.optim == 'adam':
    optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr)
train_cnn(args.datatype, cnn, [trainloader, valloader], [trainset, valset], criterion, optimizer, 1, 'test.pth')
