from customdataclasses import torch
import argparse
import choose_data
import custommodels

def predict(datatype, model, weightspath, dataloader):
    # Load in weights
    model.load_state_dict(torch.load(weightspath, map_location='cpu',  weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Collect labels and predictions
    truelabels = []
    predlabels = []
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
                predicted = predicted.view(16, 4)
                label = label.view(16, 4)

            truelabels.extend(label.cpu().numpy())
            predlabels.extend(predicted.cpu().numpy())
            # Only get first batch
            break

    return truelabels, predlabels

parser = argparse.ArgumentParser(description="Evaluate Model")
parser.add_argument('--datatype', type=str, required=True, help='Which data to use: mnist or mathwriting')
parser.add_argument('--weights', type=str, required=True, help='Path to weights')
args = parser.parse_args()

if args.datatype == 'mnist':
    _, _, _, _, _, testloader = choose_data.load_data('mnist')
    cnn = custommodels.CNN1train(10, 'mnist')
elif args.datatype == 'mathwriting':
    symbols, _, _, _, _, _, testloader = choose_data.load_data('mathwriting')
    cnn = custommodels.CNN3train(len(symbols._get_unique_labels()), 'mathwriting')
true, pred = predict(args.datatype, cnn, args.weights, testloader)
for t, p in zip(true, pred):
    print('True: ', t)
    print('Predicted:', p)