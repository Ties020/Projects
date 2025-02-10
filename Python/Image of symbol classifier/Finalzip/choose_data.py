import customdataclasses
from customdataclasses import zipfile
import xml.etree.ElementTree as ET
from torch.utils.data import DataLoader
import torchvision 
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# Helper function
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

# Define transform functions
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))  
])
transform_simple = transforms.Compose([
    transforms.ToTensor(),  
])

def load_data(datatype):
    if datatype=='mnist':
        # Load in MNIST datset
        trainset = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform)
        # Get subset of train data
        X = trainset.data.numpy()  
        Y = trainset.targets.numpy()
        _, X_small, _, Y_small = train_test_split(X, Y, stratify=Y, test_size=0.05)
        train_subset = customdataclasses.MNISTSubset(X_small, Y_small, transform=transform_simple)
        # Get subset of test set
        testset = torchvision.datasets.MNIST(root='./', train=False, download=True, transform=transform)
        X = testset.data.numpy()  
        Y = testset.targets.numpy()
        _, X_smalltest, _, Y_smalltest = train_test_split(X, Y, stratify=Y, test_size=0.2)
        # Split subset in validation and test set
        X_test, X_val, Y_test, Y_val = train_test_split(X_smalltest, Y_smalltest, stratify=Y_smalltest, test_size=0.25)
        val_subset = customdataclasses.MNISTSubset(X_val, Y_val, transform=transform_simple)
        test_subset = customdataclasses.MNISTSubset(X_test, Y_test, transform=transform_simple)
        # Make dataloaders
        trainloader = DataLoader(train_subset, batch_size=16, shuffle=True)
        valloader = DataLoader(val_subset, batch_size=16, shuffle=True)
        testloader = DataLoader(test_subset, batch_size=16, shuffle=False)
        return train_subset, trainloader, val_subset, valloader, test_subset, testloader
    elif datatype=='mathwriting':
        symbols = customdataclasses.HME("mathwriting-2024symbols_imgr.zip", "mathwriting-2024symbols.zip", "symbols/")
        X, Y, _ = symbols._get_names()
        X_big, X_small, Y_big, Y_small = train_test_split(X, Y, test_size=0.4)
        X_val, X_test, Y_val, Y_test = train_test_split(X_small, Y_small, test_size=0.75)
        train_subset = customdataclasses.Subset("mathwriting-2024symbols_imgr.zip", X_big, Y_big, transform=transform)
        val_subset = customdataclasses.Subset("mathwriting-2024symbols_imgr.zip", X_val, Y_val, transform=transform)
        test_subset = customdataclasses.Subset("mathwriting-2024symbols_imgr.zip", X_test, Y_test, transform=transform)
        trainloader = DataLoader(train_subset, batch_size=16, shuffle=True, drop_last=True)
        valloader = DataLoader(val_subset, batch_size=16, shuffle=True, drop_last=True)
        testloader = DataLoader(test_subset, batch_size=16, shuffle=False, drop_last=True)
        return symbols, train_subset, trainloader, val_subset, valloader, test_subset, testloader