import torch
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

train_features_reduced = torch.load('train_features_reduced.pt', weights_only = False)
test_features_reduced = torch.load('test_features_reduced.pt', weights_only = False)
train_labels = torch.load('train_labels.pt', weights_only = False).numpy()
test_labels = torch.load('test_labels.pt', weights_only = False).numpy()

#Create and train the decision tree classifier
dt_classifier = DecisionTreeClassifier(criterion = 'gini', max_depth = 5, random_state = 42)
dt_classifier.fit(train_features_reduced, train_labels)

predictions = dt_classifier.predict(test_features_reduced)

#Generate confusion matrix
cm = confusion_matrix(test_labels, predictions, labels = np.arange(10)) #Have class labels be integers for classes 1-10 as 0-9

#Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix = cm,  display_labels = np.arange(10))
disp.plot(cmap = plt.cm.Blues) #Plot gradient for better readability

#Add titles and labels for better visualization
plt.title("Confusion Matrix DT 2")
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
