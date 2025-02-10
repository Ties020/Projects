import processdata
import models
import traincnn

import argparse
import torch

def run_model(path, trainimages, traindata, trainpath, valimages, valdata, valpath, transform, batch_size, model, criterion, optimizer, epochs, saveweights):
  train = processdata.HME_onelabel(images=path+trainimages, data=path+traindata, path_to_data=trainpath, transform=transform)
  val = processdata.HME_onelabel(images=path+valimages, data=path+valdata, path_to_data=valpath, transform=transform)
  train_dataloader = processdata.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
  val_dataloader = processdata.DataLoader(val, batch_size=batch_size, shuffle=True, drop_last=True)
  train_losses, train_accuracies, val_losses, val_accuracies = traincnn.train_cnn(model, [train_dataloader, val_dataloader],
 [train, val], criterion, optimizer, epochs, path+saveweights)
  print(train_losses)
  print(train_accuracies)
  print(val_losses)
  print(val_accuracies)

"""# To make sure function can be run in a command prompt"""

parser = argparse.ArgumentParser(description="Run Model")
parser.add_argument('--path', type=str, required=True, help='Path to data directory')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--saveweights', type=str, required=True, help='Path to save the model weights')
parser.add_argument('--optimizer', type=str, required=True, help='Optimizer')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
args = parser.parse_args()

hme_train_cnn = processdata.HME_onelabel("mathwriting-2024mini_imgr.tgz", "mathwriting-2024mini.tgz", "mathwriting-2024/train/")
_, _, classes = hme_train_cnn._get_names()
model = models.CNN3train(classes)
if args.optimizer.lower() == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.lower() == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
else: 
   raise ValueError("Optimizer not supported!")
criterion = torch.nn.CrossEntropyLoss()
run_model(
  path=args.path,
  trainimages= "mathwriting-2024mini_imgr.tgz",
  traindata="mathwriting-2024mini.tgz",
  trainpath="mathwriting-2024/train/",
  valimages= "mathwriting-2024mini_imgr.tgz",
  valdata="mathwriting-2024mini.tgz",
  valpath="mathwriting-2024/valid/",
  transform=None,
  batch_size=args.batch_size,
  model=model,
  criterion=criterion,
  optimizer=optimizer,
  epochs=args.epochs,
  saveweights=args.saveweights
)
