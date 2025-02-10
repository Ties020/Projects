import torch
import torchvision
import time
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA


transform = transforms.Compose([
    transforms.Resize((224,224)),  #This resizes the image to 224x224x3, third argument isn't needed since that was already 3
    transforms.ToTensor(), #Convert image to pytorch tensor format 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #Normalize for ResNet-18
])

full_trainset = torchvision.datasets.CIFAR10(root='./data', train = True, download = True, transform = transform)
full_testset = torchvision.datasets.CIFAR10(root='./data', train = False, download = True, transform = transform)

def get_class_indices(dataset, samples_per_class):
    counts = {class_id: 0 for class_id in range(10)}  # Tracks the count of samples per class
    selected_indices = []
    index = 0  

    for data in dataset:
        label = data[1]  #Extract label belonging to samples in the class

        #Check if we've collected enough samples for this class
        if counts[label] < samples_per_class:
            selected_indices.append(index)  
            counts[label] += 1 

        #Stop once we've gathered enough samples for every class
        if all(count >= samples_per_class for count in counts.values()):
            break

        index += 1

    return selected_indices

#Get the first 500 training images and 100 test images of each class
train_indices = get_class_indices(full_trainset, 500)
test_indices = get_class_indices(full_testset, 100)

#Create subset datasets with these indices
trainset = Subset(full_trainset, train_indices)
testset = Subset(full_testset, test_indices)

resnet_model = torchvision.models.resnet18(weights = torchvision.models.ResNet18_Weights.DEFAULT)   #Import the pre-trained model, argument means the model is imported with pre-trained weights
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])  #Removed last layer, inspiration from: https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/5
resnet_model.eval()  

train_loader = DataLoader(trainset, batch_size = 1, shuffle = False, num_workers = 1) 
test_loader = DataLoader(testset, batch_size = 1, shuffle = False, num_workers = 1)

#Function to extract feature vectors from the model
def extract_features(loader):
    feature_vectors = []
    labels = []
    index = 0

    with torch.no_grad():  #No need to compute gradients, this would slow down device significantly
        for image, label in loader:
            print(index)
            
            index += 1
           
            features = resnet_model(image)  #Extract features, output is shape: [batch_size, 512, 1, 1]

            features = features.view(features.size(0), -1) #Flatten the features from [batch_size, 512, 1, 1] to [batch_size, 512], this is so that it is easier to use the vectors for later classifying tasks

            feature_vectors.append(features)  
            labels.append(label)  

        return torch.cat(feature_vectors, dim = 0), torch.cat(labels, dim = 0) #This returns 1 single tensor for all feature vectors from all batches and 1 tensor with all feature labels

train_features, train_labels = extract_features(train_loader)
test_features, test_labels = extract_features(test_loader)

#Reduce dimensionality of features from 512 to 50
pca = PCA(n_components = 50)
train_features_reduced = pca.fit_transform(train_features)
test_features_reduced = pca.fit_transform(test_features)

#Save both reduced vectors into memory such that the code above doesn't have to rerun everytime 
torch.save(train_features_reduced, 'train_features_reduced.pt')
torch.save(test_features_reduced, 'test_features_reduced.pt')
torch.save(train_labels, 'train_labels.pt')
torch.save(test_labels,'test_labels.pt')
