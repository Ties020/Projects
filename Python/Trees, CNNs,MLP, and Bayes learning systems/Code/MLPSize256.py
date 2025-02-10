import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


torch.manual_seed(0) #Ensure weights and biases are initialized the same every time when instantiating the MLP

#Load data, convert feature vectors from numpy arrays back to tensors since this is what dataloaders use
train_features_reduced = torch.tensor(torch.load('train_features_reduced.pt', weights_only = False), dtype = torch.float32) 
test_features_reduced = torch.tensor(torch.load('test_features_reduced.pt', weights_only = False), dtype = torch.float32)
train_labels = torch.load('train_labels.pt', weights_only = False)
test_labels = torch.load('test_labels.pt', weights_only = False)

#Make sure that the labels are of type long for CrossEntropyLoss
train_labels = train_labels.long()
test_labels = test_labels.long()

#We use batches to process the data more efficiently
train_dataset = TensorDataset(train_features_reduced, train_labels)
test_dataset = TensorDataset(test_features_reduced, test_labels)

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)

#Define model
class MLP(nn.Module): #Inherit from nn.Module which is the base class for NNs in Pytorch
    def __init__(self):
        super(MLP, self).__init__() #Initialize base functionality
        self.model = nn.Sequential( #Define the layers of the model
            nn.Linear(50, 256),
            nn.ReLU(), #Replaces all negative values of output of layer above it to 0, since negative values are of no importance to feature selection
            nn.Linear(256,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x): #Just an inherited method that needs to be implemented
        return self.model(x)

#Instantiate model, loss function, and SGD optimizer with momentum 0.09
model = MLP()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9) #Parameters of model are weights and bias which will be updated during training

#Training the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train() #Set model to training mode

    for features, labels in train_loader: #Go through all batches
        outputs = model(features)         #Passes features of batch to model, returns array with 10 values per sample (1 per class), the higher the value the more likely to belong to that class
        loss = loss_function(outputs, labels) #Compute loss by comparing predicted and true labels of the samples

        optimizer.zero_grad() #Clear gradients that were computed before
        loss.backward()       #Compute gradients for weights and biases of the model
        optimizer.step()      #Update weights and biases based on the gradients

#Evaluate the model on the test dataset
model.eval()  
all_predictions = []  #To store all predicted labels

with torch.no_grad(): 
    for features, _ in test_loader:  #Iterate through test batches
        outputs = model(features)  #Get model predictions
        predictions = torch.argmax(outputs, dim = 1)  #Get predicted class labels
        all_predictions.append(predictions)

#Concatenate all predictions into a single tensor
all_predictions = torch.cat(all_predictions).numpy()

#Generate confusion matrix
cm = confusion_matrix(test_labels.numpy(), all_predictions, labels = np.arange(10)) #Have class labels be integers for classes 1-10 as 0-9

#Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = np.arange(10))
disp.plot(cmap = plt.cm.Blues)
plt.title("Confusion Matrix MLP")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

def calculate_accuracy(cm):
    correct_predictions = np.trace(cm) #Returns sum of cells on diagonal, these are correctly identified classes
    total_predictions = np.sum(cm)
    return correct_predictions / total_predictions

def calculate_precision(cm):
    precision_per_class = []
    for c in range(len(cm)): #Loop through each class
        tp = cm[c, c]        #Returns the cell that represents where true class c was correctly predicted as class c
        fp = np.sum(cm[:, c]) - tp  #Returns sum of all numbers in column c, that were predicted as c. Minus the correctly predicted class 
        precision_per_class.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return np.mean(precision_per_class)

def calculate_recall(cm):
    recall_per_class = []
    for c in range(len(cm)):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        recall_per_class.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return np.mean(recall_per_class)

def calculate_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


#Compute metrics
accuracy = calculate_accuracy(cm)
precision = calculate_precision(cm)
recall = calculate_recall(cm)
f1 = calculate_f1(precision, recall)

print("Metrics:")
print(f"Accuracy: ", accuracy)
print(f"Precision: ", precision)
print(f"Recall: ", recall)
print(f"F1-Measure: ", f1)