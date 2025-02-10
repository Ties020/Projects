import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


train_features_reduced = torch.load('train_features_reduced.pt', weights_only = False)
test_features_reduced = torch.load('test_features_reduced.pt', weights_only = False)
train_labels = torch.load('train_labels.pt', weights_only = False)
test_labels = torch.load('test_labels.pt', weights_only = False)


class GaussianNaiveBayes:
    #This method will calculate the statistics needed for computing probabilities 
    def fit(self, X, y):
        #X is the feature matrix
        #y is the label array
        
        #These variables are used to store statistics per class
        self.classes = np.unique(y) 
        self.class_means = {} 
        self.class_variances = {}  
        self.class_priors = {}   
        
        #Calculate mean, variance, and prior for each class
        for c in self.classes:
            X_c = X[y == c]  #Extract the class that is currently being considered
            self.class_means[c] = np.mean(X_c, axis=0) #Axis is 0 to compute the stats column-wise, which represent the features
            self.class_variances[c] = np.var(X_c, axis=0)
            self.class_priors[c] = X_c.shape[0] / X.shape[0]  #This is number of samples in class / total number of samples in dataset, leads to prior
    
    def calculate_likelihood(self, class_mean, class_variance, x):
        coefficient = 1.0 / np.sqrt(2.0 * np.pi * (class_variance))
        exponent = np.exp(- (x - class_mean) ** 2 / (2 * (class_variance)))
        return coefficient * exponent
    
    def calculate_posterior(self, x):
        #Calculate posterior probability for each class
        posteriors = {}
        for c in self.classes:
            #Use logs to work with small numbers, avoids underflow
            log_prior = np.log(self.class_priors[c])
            log_likelihood = np.sum(np.log(self.calculate_likelihood(self.class_means[c], self.class_variances[c], x))) #Likelihood of class is sum of likelihoods per feature
            posteriors[c] = log_prior + log_likelihood
        return posteriors

    def predict(self, X):
        x_pred = [self.predict_single(x) for x in X]
        return np.array(x_pred)  #Returns array with predicted class label for each sample in X
    
    def predict_single(self, x):
        posteriors = self.calculate_posterior(x)
        return max(posteriors, key = posteriors.get) #Return class with highest posterior

#Don't need to convert feature vectors to numpy since they are already of that type, done by the PCA transformation
#Labels are of type tensor so they do need to be transformed to numpy arrays
train_labels = train_labels.numpy()
test_labels = test_labels.numpy()

#Create and fit/prepare the model
gnb = GaussianNaiveBayes()
gnb.fit(train_features_reduced, train_labels)

#Predict on the test set, we get an array with the predicted labels for each sample. So first label belongs to first sample.
predictions = gnb.predict(test_features_reduced) 

#Generate confusion matrix
cm = confusion_matrix(test_labels, predictions, labels = np.arange(10)) #Have class labels be integers for classes 1-10 as 0-9

#Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = np.arange(10))
disp.plot(cmap = plt.cm.Blues) #Plot gradient for better readability

#Add titles and labels for better visualization
plt.title("Confusion Matrix Bayes")
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

