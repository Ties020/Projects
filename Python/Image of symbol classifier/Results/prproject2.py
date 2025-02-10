import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import zipfile
import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder
import inkml2img2 as inkml2img2
import functools
import operator
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))  
])

transform_simple = transforms.Compose([
    transforms.ToTensor(),  
])

def tokenize_label(label):
    tokens = []
    i = 0
    # Loop through label
    while i < len(label):
        char = label[i]
        # If symbol is part of a Latex command, collect all parts of the command
        if char == "\\":
            i += 1
            collectchar = char
            while i < len(label) and label[i] != "{" and label[i] != " ":
                collectchar += label[i]
                i += 1
            tokens.append(collectchar)
        # Else, append symbol to the token list
        else:
            if char != " ":
                tokens.append(char) 
            i += 1
    return tokens

class MNISTSubset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image) 
        return image, label
    
class Subset(Dataset):
    def __init__(self, file, data, targets, transform=None):
        self.file = file
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data[idx]
        label = self.targets[idx]
        try:
            with zipfile.ZipFile(self.file, 'r') as zip_data:
                with zip_data.open(file_name) as file:
                    image = Image.open(file).convert('L')
                    if self.transform:
                        image = self.transform(image)
                    else:
                        image = transforms.ToTensor()(image)
        except KeyError:
            print(f"Error: Failed to open {file_name} from the archive.")
            image = Image.new('L', (256, 256), color=255) 
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
        return image, label

# # Get subset of train set
# trainset = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform)
# X = trainset.data.numpy()  
# Y = trainset.targets.numpy()
# X_big, X_small, Y_big, Y_small = train_test_split(X, Y, stratify=Y, test_size=0.05)
# train_subset = MNISTSubset(X_small, Y_small, transform=transform_simple)
# # Get subset of test set
# testset = torchvision.datasets.MNIST(root='./', train=False, download=True, transform=transform)
# X = testset.data.numpy()  
# Y = testset.targets.numpy()
# X_bigtest, X_smalltest, Y_bigtest, Y_smalltest = train_test_split(X, Y, stratify=Y, test_size=0.2)
# X_test, X_val, Y_test, Y_val = train_test_split(X_smalltest, Y_smalltest, stratify=Y_smalltest, test_size=0.25)
# val_subset = MNISTSubset(X_val, Y_val, transform=transform_simple)
# test_subset = MNISTSubset(X_test, Y_test, transform=transform_simple)
# # Check dataset sizes
# print(len(train_subset))
# print(len(val_subset))
# print(len(test_subset))

# trainloader = DataLoader(train_subset, batch_size=16, shuffle=True)
# valloader = DataLoader(val_subset, batch_size=16, shuffle=True)
# testloader = DataLoader(test_subset, batch_size=16, shuffle=False)

class HME(Dataset):

    def __init__(self, images, data, path_to_data, label_encoder, datatype, transform=None):
        self.images = images
        self.data = data
        self.path_to_data = path_to_data
        self.label_encoder = label_encoder
        self.transform = transform
        self.datatype = datatype
        self.file_names, self.label_names, self.label_mapping = self._get_names()

    def __tokenize_label(self, label):
        tokens = []
        i = 0
        # Loop through label
        while i < len(label):
            char = label[i]
            # If symbol is part of a Latex command, collect all parts of the command
            if char == "\\":
                i += 1
                collectchar = char
                while i < len(label) and label[i] != "{" and label[i] != " ":
                    collectchar += label[i]
                    i += 1
                tokens.append(collectchar)
            # Else, append symbol to the token list
            else:
                if char != " ":
                    tokens.append(char) 
                i += 1
        return tokens

    def _get_names(self):
        # Collect files names and labels
        file_names = []
        label_names = []
        # Go through zip file
        with zipfile.ZipFile(self.data, 'r') as zip_data:
            all_content = zip_data.namelist()
            for item in all_content:
                 # Make sure item is not a directory
                 if item.startswith(self.path_to_data) and not item.endswith('/'):
                     # Only check inkml files
                     if item.endswith(('.inkml')):
                        # Add file to list
                        base = os.path.splitext(item)[0]
                        file_names.append(base+'.png')
                        # Open the inkml file
                        with zip_data.open(item) as inkml_file:
                            # Parse inkml file to get the annotation
                            tree = ET.parse(inkml_file)
                            root = tree.getroot()
                            namespaces = {'inkml': 'http://www.w3.org/2003/InkML'}
                            if self.datatype =='symbols':
                                annotation = root.find('.//inkml:annotation[@type="label"]', namespaces)
                            else: 
                                annotation = root.find('.//inkml:annotation[@type="normalizedLabel"]', namespaces)
                            # Tokenize label
                            if annotation is not None:
                                annotation = annotation.text
                                annotationlst = self.__tokenize_label(annotation)
                            label_names.append(annotationlst)
        # Encode label_names and pad it so all labels have the same length
        label_names_encoded = [self.label_encoder.transform(lst) for lst in label_names]
        padding_value = len(self.label_encoder.classes_) 
        label_names_encoded = [torch.tensor(lst) for lst in label_names_encoded]
        padded_labels = pad_sequence(label_names_encoded, batch_first=True, padding_value=padding_value)
        # Collect label mapping
        label_mapping = {code: label for label, code in zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_)))}
        return file_names, padded_labels, label_mapping

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        label = self.label_names[idx]
        with zipfile.ZipFile(self.images, 'r') as zip_data:
            with zip_data.open(file_name) as file:
                image = Image.open(file).convert('L')
                if self.transform:
                    image = self.transform(image)
                else:
                    image = transforms.ToTensor()(image)

        return image, label
    
    def _getlabel(self, encoded_label):
        label = ""
        # Loop through encoded label and translate everything but the padding
        for elem in encoded_label:
            if elem != len(self.label_encoder.classes_):
                label += self.label_mapping[int(elem)]
        return label
    
    def _get_unique_labels(self):
        unique_labels = set()
        for label in self.label_names:
            converted_tuple = tuple(t.item() for t in label)
            unique_labels.add(converted_tuple)
        return unique_labels

class CNN3train(nn.Module):
    def __init__(self, classes):
        super(CNN3train, self).__init__()
        self.classes = classes
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 100, 3, padding=1),  
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 100, 3, padding=1),  
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 100, 3, padding=1), 
            nn.BatchNorm2d(100),
            nn.ReLU(),
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(100, 200, 3, padding=1),  
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, 3, padding=1),  
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, 3, padding=1), 
            nn.BatchNorm2d(200),
            nn.ReLU(),
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(200, 300, 3, padding=1), 
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(300, 300, 3, padding=1), 
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(300, 300, 3, padding=1), 
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(300, 400, 3, padding=1),  
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(400, 400, 3, padding=1),  
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(400, 400, 3, padding=1), 
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(400, 512, 3, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.fc1 = nn.Linear(512 * 28 * 28, 1024) 
        self.fc2 = nn.Linear(1024, 4*self.classes)

    def forward(self, x):
        x = self.conv_block_1(x)  
        x = self.conv_block_2(x) 
        x = self.conv_block_3(x)  
        x = self.conv_block_4(x)  
        x = self.conv_block_5(x) 

        x = x.view(x.size(0), -1)  
        x = self.fc1(x)  
        x = nn.ReLU()(x)  
        x = self.fc2(x)
        x = x.view(16, 4, 229)
        return x   

class CNN2train(nn.Module):
    def __init__(self, classes):
        super(CNN2train, self).__init__()
        self.classes = classes
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, padding=1),  
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Conv2d(50, 100, 3, padding=1),  
            nn.BatchNorm2d(100),
            nn.ReLU(),
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(100, 150, 3, padding=1),  
            nn.BatchNorm2d(150),
            nn.ReLU(),
            nn.Conv2d(150, 200, 3, padding=1),  
            nn.BatchNorm2d(200),
            nn.ReLU(),
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(200, 250, 3, padding=1), 
            nn.BatchNorm2d(250),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(250, 300, 3, padding=1), 
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(300, 350, 3, padding=1),  
            nn.BatchNorm2d(350),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(350, 400, 3, padding=1),  
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(400, 512, 3, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.fc1 = nn.Linear(512 * 28 * 28, 1024) 
        self.fc2 = nn.Linear(1024, 4 * self.classes)

    def forward(self, x):
        x = self.conv_block_1(x)  
        x = self.conv_block_2(x) 
        x = self.conv_block_3(x)  
        x = self.conv_block_4(x)  
        x = self.conv_block_5(x) 

        x = x.view(x.size(0), -1)  
        x = self.fc1(x)  
        x = nn.ReLU()(x)  
        x = self.fc2(x)
        x = x.view(16, 4, 229)
        return x   

class CNN1train(nn.Module):
    def __init__(self, classes):
        super(CNN1train, self).__init__()
        self.classes = classes
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),  
            nn.ReLU(),
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),  
            nn.ReLU(),
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), 
            nn.ReLU(),
        )
        
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(512 * 28 * 28, 1024) 
        self.fc2 = nn.Linear(1024, 4 * self.classes)

    def forward(self, x):
        x = self.conv_block_1(x)  
        x = self.conv_block_2(x) 
        x = self.conv_block_3(x)  
        x = self.conv_block_4(x)  
        x = self.conv_block_5(x) 

        x = x.view(x.size(0), -1)  
        x = self.fc1(x)  
        x = nn.ReLU()(x)  
        x = self.fc2(x)
        x = x.view(16, 4, 229)
        return x   

def train_cnn(model, dataloaders, datasets, criterion, optimizer, epochs, save_path):
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
            output = output.view(-1, 229)
            label = label.view(-1)
            label = label.long()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * item.size(0)

            _, predicted = torch.max(output, 1)
            predicted_reshaped = predicted.view(16, 4)
            labels_reshaped = label.view(16, 4)
            correct_train += (torch.all(predicted_reshaped == labels_reshaped, dim=1)).sum().item()
            
        print(correct_train)
        print(len(datasets[0]))
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
                output = output.view(-1, 229)
                label = label.view(-1)
                label = label.long()
                loss = criterion(output, label)
                val_loss += loss.item() * item.size(0)

                _, predicted = torch.max(output, 1)
                predicted_reshaped = predicted.view(16, 4)
                labels_reshaped = label.view(16, 4)
                correct_val += (torch.all(predicted_reshaped == labels_reshaped, dim=1)).sum().item()

            val_loss /= len(datasets[1])
            val_losses.append(val_loss)
            val_accuracy = 100 * correct_val/len(datasets[1])
            val_accuracies.append(val_accuracy)


        print(f"Epoch {epoch+1}/{epochs}, Train_loss: {train_loss}, Train_acc: {train_accuracy}, Val_loss: {val_loss}, Val_acc: {val_accuracy}")
    torch.save(model.state_dict(), save_path)
    return train_losses, train_accuracies, val_losses, val_accuracies

# Create labelencoder 
label_names = []
with zipfile.ZipFile("mathwriting-2024symbols.zip", 'r') as zip_data:
    all_content = zip_data.namelist()
    for item in all_content:
        # Make sure item is not a directory
        if not item.endswith('/'):
            # Only check inkml files
            if item.endswith(('.inkml')):
                # Open the inkml file
                with zip_data.open(item) as inkml_file:
                    # Parse inkml file to get the annotation
                    tree = ET.parse(inkml_file)
                    root = tree.getroot()
                    namespaces = {'inkml': 'http://www.w3.org/2003/InkML'}
                    if item.startswith("symbols/"):
                        annotation = root.find('.//inkml:annotation[@type="label"]', namespaces)
                    else: 
                        annotation = root.find('.//inkml:annotation[@type="normalizedLabel"]', namespaces)
                    # Tokenize label
                    if annotation is not None:
                        annotation = annotation.text
                        annotation = tokenize_label(annotation)
                        label_names.append(annotation)
# Encode label
label_encoder = LabelEncoder()
# Flatten label_names, so all symbols can be encoded separately
label_names_flat = functools.reduce(operator.concat, label_names)
label_encoder.fit_transform(label_names_flat)

symbols = HME("mathwriting-2024symbols_imgr3.zip", "mathwriting-2024symbols.zip", "symbols/", label_encoder, datatype='symbols')
print(len(symbols))
print(len(symbols._get_unique_labels()))
X, Y, _ = symbols._get_names()
X_big, X_small, Y_big, Y_small = train_test_split(X, Y, test_size=0.4)
X_val, X_test, Y_val, Y_test = train_test_split(X_small, Y_small, test_size=0.75)
train_subset = Subset("mathwriting-2024symbols_imgr3.zip", X_big, Y_big, transform=transform)
val_subset = Subset("mathwriting-2024symbols_imgr3.zip", X_val, Y_val, transform=transform)
test_subset = Subset("mathwriting-2024symbols_imgr3.zip", X_test, Y_test, transform=transform)
print(len(train_subset), len(val_subset), len(test_subset))
symbolstrain_dataloader = DataLoader(train_subset, batch_size=16, shuffle=True, drop_last=True)
symbolsval_dataloader = DataLoader(val_subset, batch_size=16, shuffle=True, drop_last=True)
symbolstest_dataloader = DataLoader(test_subset, batch_size=16, shuffle=False, drop_last=True)
img, la = next(iter(symbolstest_dataloader))
# for i, l in zip(img, la):
#     print(l)
#     print(symbols._getlabel(l))

cnn = CNN3train(len(symbols._get_unique_labels()))
criterion = nn.CrossEntropyLoss(ignore_index=215)
optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01)
# train_losses, train_accuracies, val_losses, val_accuracies = train_cnn(cnn, [symbolstrain_dataloader, symbolsval_dataloader], 
#                                                                         [train_subset, val_subset], criterion, optimizer, 10, 
#                                                                         'try2/symbolscnn3_s9.pth')
# print(train_losses)
# print(train_accuracies)
# print(val_losses)
# print(val_accuracies)
     
def calculate_metrics(model, weightspath, dataloader, dataset):
    # Load in weights
    model.load_state_dict(torch.load(weightspath, map_location='cpu',  weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Collect labels and predictions
    truelabels = []
    predlabels = []
    with torch.no_grad():
        for input, label in dataloader:
            input = input.to(device)
            label = label.to(device)

            output = model(input)
            output = output.view(-1, 229)
            label = label.view(-1)
            label = label.long()
            _, predicted = torch.max(output, 1)

            truelabels.extend(label.cpu().numpy())
            predlabels.extend(predicted.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(truelabels, predlabels)
    precision = precision_score(truelabels, predlabels, average='macro')
    recall = recall_score(truelabels, predlabels, average='macro')
    f1 = f1_score(truelabels, predlabels, average='macro')
    cm = confusion_matrix(truelabels, predlabels)

    return accuracy, precision, recall, f1, cm

# _, _, _, _, matrix = (calculate_metrics(cnn, './try2/symbolscnn3_s9.pth', symbolstest_dataloader, test_subset))
