In the zip file I provided there is one folder called "Code". 
In this folder the code for all models and the data-preprocessing can be found.

Each model has a clear name, for example, MLPMain.py is the code to be ran for the MLP model to train, evaluate, and apply the model. MLPSize1024.py would be the file for a variant of the MLP model.
All models have a single file per variant, running this file will train, evaluate, and apply its corresponding model. For clarity, I ran the file in vscode with Python 3.11.6. 

To preprocess the data, the file DataLoadingResNet.py has to be ran. This will save the data in 4 files named: test_features_reduced.pt, train_features_reduced.pt, test_labels.pt, and train_labels.pt. 
The data from these files will be loaded in and used by the Bayes, Decision Tree, and MLP model. This is to only have to preprocess the CIFAR-10 dataset once. 

