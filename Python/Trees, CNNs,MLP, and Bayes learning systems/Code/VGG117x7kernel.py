import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

torch.manual_seed(0)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1), #This is to have input channel for first conv layer be 1, without this it would need 3 since image is originally in rgb
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) #Normalize greyscale image, taken from: https://discuss.pytorch.org/t/transoforms-normalize-grayscale-image/161843 
])

full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train = True, download = True, transform = transform)
full_test_dataset = torchvision.datasets.CIFAR10(root='./data', train = False, download = True, transform = transform)

#Filter the dataset to include only the first 500 training and 100 test images per class
def filter_dataset(dataset, max_per_class):
    class_counts = {i: 0 for i in range(10)}  
    filtered_data = []
    
    for img, label in dataset:
        if class_counts[label] < max_per_class:
            filtered_data.append((img, label))
            class_counts[label] += 1
        if all(count >= max_per_class for count in class_counts.values()):
            break
    
    images, labels = zip(*filtered_data) #Group the images and labels using zip, *filtered_data unpacks the tuples and zip groups the elements of the tuples
    return torch.stack(images), torch.tensor(labels) #Return images and labels as tensors

train_images, train_labels = filter_dataset(full_train_dataset, max_per_class = 500)
test_images, test_labels = filter_dataset(full_test_dataset, max_per_class = 100)

#Create DataLoader for filtered datasets
train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = False)

#Define VGG11 Model
class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 7, 1, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 7, 1, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 7, 1, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 7, 1, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 7, 1, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 7, 1, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 7, 1, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 7, 1, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.calculate_features_size() #Adjusts input size for fully-connected layers when removing layers and changing kernel sizes since the output of the conv layers would also change


        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    def calculate_features_size(self):
        dummy_input = torch.zeros(1, 1, 32, 32)  #Size of grayscale CIFAR images
        features_out = self.features(dummy_input)  #Get the output size
        self.fc_input_size = features_out.view(1, -1).size(1) #Flatten the features to get the input size for the first fully connected layer
    
    def forward(self, x): #Passes images throught the network
        x = self.features(x)   #Extract features
        x = x.view(x.size(0), -1)  #Flatten features into a 2D tensor for fully connected layers
        x = self.classifier(x) #Pass features through fully connected layers, output is tensor of size (batch_size,10)
        return x

#Instantiate model, loss function, and optimizer
model = VGG11()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)

#Train the model, same principle as MLP
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        print(epoch)
        outputs = model(images)
        loss = loss_function(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            

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
plt.title("Confusion Matrix VGG11")
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
