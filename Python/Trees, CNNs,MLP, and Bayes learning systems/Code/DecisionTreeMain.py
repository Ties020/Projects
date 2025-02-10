import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

train_features_reduced = torch.load('train_features_reduced.pt', weights_only = False)
test_features_reduced = torch.load('test_features_reduced.pt', weights_only = False)
train_labels = torch.load('train_labels.pt', weights_only = False)
test_labels = torch.load('test_labels.pt', weights_only = False)
train_labels = train_labels.numpy()
test_labels = test_labels.numpy()


class DecisionTreeClassifier:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self.build_tree(X, y, depth = 0) #Start building tree from depth 0

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.tree) for x in X]) #This returns an array with all the predicted labels

    def gini(self, y):
        #Calculate gini impurity for set of labels
        classes, counts = np.unique(y, return_counts = True) #Need return_counts argument to return number of occurences of certain label value corresponding to a class
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def split(self, X, y, feature_index, threshold):

        left_indices = X[:, feature_index] <= threshold #Checks if each feature of X in column feature_index are <= threshold and puts them in array like [True, True, False....]
        right_indices = ~left_indices #Inverse matrix of left_indices
        return X[left_indices], X[right_indices], y[left_indices], y[right_indices] #Returns samples on each side (on right X[right_indices]) with corresponding labels (y[right_indices])

    def best_split(self, X, y):

        best_gini = float("inf")  #Lower the gini -> better the split -> better separation between classes, this is why we start with infinity since it can only get better
        best_split = None
        n_features = X.shape[1]

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index]) #For the current feature, extract all its unique values

            #Try each of these values (thresholds) to split the data and see which one is best
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self.split(X, y, feature_index, threshold)
                if len(y_left) == 0 or len(y_right) == 0: #Skip if invalid split, since one side would have no samples
                    continue
                
                #Calculate gini impurities
                gini_left = self.gini(y_left)
                gini_right = self.gini(y_right)

                weighted_gini = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)

                #Check if this current split has the better gini, if so: store information of this split
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "left_indices": (X_left, y_left),
                        "right_indices": (X_right, y_right),
                    }
        return best_split

    def build_tree(self, X, y, depth):
        print(f"Building tree at depth {depth}, samples: {len(y)}")

        if len(np.unique(y)) == 1 or depth == self.max_depth or len(X) == 0: #Stop recursion if all labels are the same, max_depth is reached or no further samples remain
            return {"leaf": True, "class": np.argmax(np.bincount(y))} #Create leaf node of the most common class, done by checking array of labels and returning most occuring class 

        best_split = self.best_split(X, y)

        if best_split is None: #Return leaf node with most common class if no split is found
            return {"leaf": True, "class": np.argmax(np.bincount(y))}
        
        #Recursively build left and right sub-trees
        left_tree = self.build_tree(*best_split["left_indices"], depth + 1) # * unpacks the elements of best_split, so in case of "right_indices" it will return (X_right, y_right) which is passed to build_tree
        right_tree = self.build_tree(*best_split["right_indices"], depth + 1)

        #After building sub-tree, return current node with its sub-trees, initial call will return info of the whole tree
        return {
            "leaf": False,
            "feature_index": best_split["feature_index"],
            "threshold": best_split["threshold"],
            "left": left_tree,
            "right": right_tree,
            }

    def traverse_tree(self, x, node):
        if node["leaf"]:
            return node["class"] #Return predicted class label
        
        if x[node["feature_index"]] <= node["threshold"]:
            return self.traverse_tree(x, node["left"])

        else:
            return self.traverse_tree(x, node["right"])
        


dt = DecisionTreeClassifier(max_depth = 50)
dt.fit(train_features_reduced, train_labels) 

predictions = dt.predict(test_features_reduced)

#Generate confusion matrix
cm = confusion_matrix(test_labels, predictions, labels = np.arange(10)) #Have class labels be integers for classes 1-10 as 0-9

#Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = np.arange(10))
disp.plot(cmap = plt.cm.Blues) #Plot gradient for better readability

#Add titles and labels for better visualization
plt.title("Confusion Matrix DT 1")
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
