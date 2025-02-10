from customdataclasses import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse
import choose_data
import custommodels

def calculate_metrics(datatype, model, weightspath, dataloader, dataset):
    # Load in weights
    model.load_state_dict(torch.load(weightspath, map_location='cpu',  weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Collect labels and predictions
    truelabels = []
    predlabels = []
    correct_val = 0
    with torch.no_grad():
        for input, label in iter(dataloader):
            input = input.to(device)
            label = label.to(device)

            output = model(input)
            if datatype=='mathwriting':
                output = output.view(-1, 229)
                label = label.view(-1)
                label = label.long()
            _, predicted = torch.max(output, 1)
            if datatype=='mathwriting':
                predicted_reshaped = predicted.view(16, 4)
                labels_reshaped = label.view(16, 4)
                correct_val += (torch.all(predicted_reshaped == labels_reshaped, dim=1)).sum().item()

            truelabels.extend(label.cpu().numpy())
            predlabels.extend(predicted.cpu().numpy())
        eq_accuracy = 100 * correct_val/len(dataset)

    # Calculate metrics
    accuracy = accuracy_score(truelabels, predlabels)
    precision = precision_score(truelabels, predlabels, average='macro')
    recall = recall_score(truelabels, predlabels, average='macro')
    f1 = f1_score(truelabels, predlabels, average='macro')
    cm = confusion_matrix(truelabels, predlabels)

    return eq_accuracy, accuracy, precision, recall, f1, cm

parser = argparse.ArgumentParser(description="Evaluate Model")
parser.add_argument('--datatype', type=str, required=True, help='Which data to use: mnist or mathwriting')
parser.add_argument('--weights', type=str, required=True, help='Path to weights')
args = parser.parse_args()

if args.datatype == 'mnist':
    _, _, _, _, testset, testloader = choose_data.load_data('mnist')
    cnn = custommodels.CNN1train(10, 'mnist')
elif args.datatype == 'mathwriting':
    symbols, _, _, _, _, testset, testloader = choose_data.load_data('mathwriting')
    cnn = custommodels.CNN3train(len(symbols._get_unique_labels()), 'mathwriting')
eq_accuracy, accuracy, precision, recall, f1, cm = calculate_metrics(args.datatype, cnn, args.weights, testloader, testset)
print('Equation accuracy: ', eq_accuracy, "(n.a. for mnist)")
print(f"Token accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1-score: {f1}")
print("Token confusion matrix:")
print(cm)