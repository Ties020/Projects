import zipfile
import xml.etree.ElementTree as ET
import re
from sklearn.preprocessing import LabelEncoder
import inkml2img2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
from xml.dom import minidom
import functools
import operator


def tokenize_label(label):
    tokens = []
    i = 0
    # Loop through label
    while i < len(label):
        char = label[i]
        # If symbol is part of a Latex command, collect all parts of the command
        if char == "\\":
            i += 1
            collectchar = char
            while i < len(label) and label[i] != "{" and label[i] != " ":
                collectchar += label[i]
                i += 1
            tokens.append(collectchar)
        # Else, append symbol to the token list
        else:
            if char != " ":
                tokens.append(char) 
            i += 1
    return tokens

class HME(Dataset):

    def __init__(self, data, path_to_data, label_encoder, datatype, transform=None):
        self.data = data
        self.path_to_data = path_to_data
        self.label_encoder = label_encoder
        self.transform = transform
        self.datatype = datatype
        self.file_names, self.label_names, self.label_mapping = self._get_names()

    def __tokenize_label(self, label):
        tokens = []
        i = 0
        # Loop through label
        while i < len(label):
            char = label[i]
            # If symbol is part of a Latex command, collect all parts of the command
            if char == "\\":
                i += 1
                collectchar = char
                while i < len(label) and label[i] != "{" and label[i] != " ":
                    collectchar += label[i]
                    i += 1
                tokens.append(collectchar)
            # Else, append symbol to the token list
            else:
                if char != " ":
                    tokens.append(char) 
                i += 1
        return tokens

    def _get_names(self):
        # Collect files names and labels
        file_names = []
        label_names = []
        # Go through zip file
        with zipfile.ZipFile(self.data, 'r') as zip_data:
            all_content = zip_data.namelist()
            for item in all_content:
                 # Make sure item is not a directory
                 if item.startswith(self.path_to_data) and not item.endswith('/'):
                     # Only check inkml files
                     if item.endswith(('.inkml')):
                        # Add file to list
                        file_names.append(item)
                        # Open the inkml file
                        with zip_data.open(item) as inkml_file:
                            # Parse inkml file to get the annotation
                            tree = ET.parse(inkml_file)
                            root = tree.getroot()
                            namespaces = {'inkml': 'http://www.w3.org/2003/InkML'}
                            if self.datatype =='symbols':
                                annotation = root.find('.//inkml:annotation[@type="label"]', namespaces)
                            else: 
                                annotation = root.find('.//inkml:annotation[@type="normalizedLabel"]', namespaces)
                            # Tokenize label
                            if annotation is not None:
                                annotation = annotation.text
                                annotationlst = self.__tokenize_label(annotation)
                            label_names.append(annotationlst)
        # Encode label_names and pad it so all labels have the same length
        label_names_encoded = [self.label_encoder.transform(lst) for lst in label_names]
        padding_value = len(self.label_encoder.classes_) 
        label_names_encoded = [torch.tensor(lst) for lst in label_names_encoded]
        padded_labels = pad_sequence(label_names_encoded, batch_first=True, padding_value=padding_value)
        # Collect label mapping
        label_mapping = {code: label for label, code in zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_)))}
        return file_names, padded_labels, label_mapping

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        label = self.label_names[idx]
        with zipfile.ZipFile(self.data, 'r') as zip_data:
            with zip_data.open(file_name) as file:
                # Convert inkml file to image
                root = ET.fromstring(file.read())
                image = inkml2img2.inkml_to_image(root)
                if self.transform:
                    image = self.transform(image)
                else:
                    image = transforms.ToTensor()(image)

        return image, label
    
    def _getlabel(self, encoded_label):
        label = ""
        # Loop through encoded label and translate everything but the padding
        for elem in encoded_label:
            if elem != len(self.label_encoder.classes_):
                label += self.label_mapping[int(elem)]
        return label

class HME_onelabel(Dataset):

    def __init__(self, data, path_to_data, transform=None):
        self.data = data
        self.path_to_data = path_to_data
        self.transform = transform
        self.file_names, self.label_names, self.label_mapping = self._get_names()

    def _get_names(self):
        # Collect files names and labels
        file_names = []
        label_names = []
        # Go through zip file
        with zipfile.ZipFile(self.data, 'r') as zip_data:
            all_content = zip_data.namelist()
            for item in all_content:
                 # Make sure item is not a directory
                 if item.startswith(self.path_to_data) and not item.endswith('/'):
                     # Only check inkml files
                     if item.endswith(('.inkml')):
                        # Add file to list
                        file_names.append(item)
                        # Open the inkml file
                        with zip_data.open(item) as inkml_file:
                            # Parse inkml file to get the annotation
                            tree = ET.parse(inkml_file)
                            root = tree.getroot()
                            namespaces = {'inkml': 'http://www.w3.org/2003/InkML'}
                            annotation = root.find('.//inkml:annotation[@type="normalizedLabel"]', namespaces)
                            if annotation is not None:
                                annotation = annotation.text
                            label_names.append(annotation)
        # Encode label
        label_encoder = LabelEncoder()
        label_names = label_encoder.fit_transform(label_names)
        label_mapping = {code: label for label, code in zip(label_encoder.classes_, range(len(label_encoder.classes_)))}
        return file_names, label_names, label_mapping

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        label = self.label_names[idx]
        with zipfile.ZipFile(self.data, 'r') as zip_data:
            with zip_data.open(file_name) as file:
                root = ET.fromstring(file.read())
                # xml_str = ET.tostring(root, encoding='utf-8').decode()
                # pretty_xml = minidom.parseString(xml_str).toprettyxml()
                # print(pretty_xml)
                image = inkml2img2.inkml_to_image(root)
                if self.transform:
                    image = self.transform(image)
                else:
                    image = transforms.ToTensor()(image)

        return image, label

    def _getlabel(self, encoded_label):
        print(encoded_label)
        return self.label_mapping[encoded_label]
    

def train_cnn(model, dataloaders, datasets, criterion, optimizer, epochs, save_path):
    # Put model in gpu if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f'Model is on device: {next(model.parameters()).device}')

    _, _, classes = datasets[0]._get_names()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        # Loop through batch
        for item, label in iter(dataloaders[0]):
            # Put data in the same device as the model
            item = item.to(device)
            label = label.to(device)
            # Set the gradient to zero
            optimizer.zero_grad()

            # Run model on input, calculate error and update weights
            output = model(item)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * item.size(0)

            _, predicted = torch.max(output, 1)
            correct_train += (predicted == label).sum().item()

        train_loss /= len(datasets[0])
        train_losses.append(train_loss)
        train_accuracy = 100 * correct_train/len(datasets[0])
        train_accuracies.append(train_accuracy)

        # model.eval()
        # val_loss = 0
        # correct_val = 0
        # with torch.no_grad():
        #     for item, label in iter(dataloaders[1]):
        #         item = item.to(device)
        #         label = label.to(device)

        #         output = model(item)
        #         loss = criterion(output, label)
        #         val_loss += loss.item() * item.size(0)

        #         _, predicted = torch.max(output, 1)
        #         correct_val += (predicted == label).sum().item()

        #     val_loss /= len(datasets[1])
        #     val_losses.append(val_loss)
        #     val_accuracy = 100 * correct_val/len(datasets[1])
        #     val_accuracies.append(val_accuracy)


        print(f"Epoch {epoch+1}/{epochs}, Train_loss: {train_loss}, Train_acc: {train_accuracy}")
        #, Val_loss: {val_loss}, Val_acc: {val_accuracy}
    torch.save(model.state_dict(), save_path)
    #, val_losses, val_accuracies
    return train_losses, train_accuracies, val_losses, val_accuracies

class CNN3train(nn.Module):
    def __init__(self, classes):
        super(CNN3train, self).__init__()
        self.classes = classes
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 100, 3),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 100, 3),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 100, 3),
            nn.BatchNorm2d(100),
            nn.ReLU(),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(100, 200, 3),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, 3),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, 3),
            nn.BatchNorm2d(200),
            nn.ReLU(),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(200, 300, 3),
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(300, 300, 3),
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(300, 300, 3),
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(300, 400, 3),
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(400, 400, 3),
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(400, 400, 3),
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(400, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(1, 2)
        self.pool3 = nn.MaxPool2d(2, 1)
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)  
        self.fc2 = nn.Linear(1024, len(self.classes))

    def forward(self, x):  
        x = self.pool1(self.conv_block_1(x))
        x = self.pool1(self.conv_block_2(x))
        x = self.pool2(self.conv_block_3(x))
        x = self.pool3(self.conv_block_4(x))
        x = self.conv_block_5(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.ReLU()(x)  
        x = self.fc2(x) 
        return x
    
class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 100, 3),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 100, 3),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 100, 3),
            nn.BatchNorm2d(100),
            nn.ReLU(),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(100, 200, 3),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, 3),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, 3),
            nn.BatchNorm2d(200),
            nn.ReLU(),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(200, 300, 3),
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(300, 300, 3),
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(300, 300, 3),
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(300, 400, 3),
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(400, 400, 3),
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(400, 400, 3),
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(400, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.pool1 = nn.MaxPool2d(2, 1)
        self.pool2 = nn.MaxPool2d((1, 2), 1)
        self.pool3 = nn.MaxPool2d((2, 1), 2)

    def forward(self, x):  
        x = self.pool1(self.conv_block_1(x))
        x = self.pool1(self.conv_block_2(x))
        x = self.pool2(self.conv_block_3(x))
        x = self.pool3(self.conv_block_4(x))
        x = self.conv_block_5(x) 
        return x

class RowEncoder(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(RowEncoder, self).__init__()
        self.blstm = nn.LSTM(num_features, hidden_size, num_layers = 1, bidirectional = True, batch_first = True) 

    def forward(self, F):
        batch_size, feature_size, height, width = F.shape
        F_reshaped = F.view(batch_size * height, width, feature_size) #Reshape for row-wise BLSTM: (batch_size * K, L, D)

        blstm_out, _ = self.blstm(F_reshaped)  #Output: (batch_size * K, L, hidden_dim * 2)
        F_prime = blstm_out.view(batch_size, height, width, -1) #Reshape back to (batch_size, K, L, hidden_dim

        F_prime = F_prime.permute(0, 3, 1, 2)
        return F_prime


class RowDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_dim):
        super(RowDecoder, self).__init__()
        self.hidden_size = hidden_size

        self.GridTransf = nn.Linear(512, 512)

        #Attention weights
        self.W_h = nn.Linear(hidden_size, hidden_size) 
        self.W_F = nn.Linear(512, hidden_size) 
        self.attn_softmax = nn.Softmax(dim=1)

        #LSTM for decoding
        self.lstm = nn.LSTMCell(embedding_dim + hidden_size, hidden_size)

        #Output layers
        self.W_c = nn.Linear(2 * hidden_size, hidden_size)
        self.W_out = nn.Linear(hidden_size, output_size)

    def forward(self, input_token, hidden_state, cell_state, encoded_image):
       
        batch_size, feature_size, height, width = encoded_image.shape

        #Flatten the encoded image to match hidden state
        encoded_image_flat = encoded_image.reshape(batch_size, feature_size, -1)  #New shape: [batch_size, 512, height*width]

        h_t_transformed = self.W_h(hidden_state) 

        # Now compute attention scores for each position (u,v) in the grid
        scores = []
        for i in range(height * width):
            F_prime_u_v = encoded_image_flat[:, :, i]  

            F_prime_u_v_transformed = self.W_F(F_prime_u_v) 

            #Compute the attention score for each (u,v) in the feature grid
            score = torch.tanh(h_t_transformed + F_prime_u_v_transformed) 
            scores.append(score)

        scores = torch.stack(scores, dim=1)  

        attention_weights = self.attn_softmax(scores.mean(dim=-1)/0.1)  # Shape: [batch_size, height*width]

        attention_weights = attention_weights.unsqueeze(-1)  # Shape: [batch_size, height*width, 1]

        encoded_image_flat = encoded_image_flat.permute(0, 2, 1)  # Shape: [batch_size, height*width, 512]
        context_vector = torch.sum(attention_weights * encoded_image_flat, dim=1)  # Shape: [batch_size, 512]
 
        #Concatenate context vector with input_token and pass through LSTM
        lstm_input = torch.cat([input_token, context_vector], dim=-1)   

        hidden_state, cell_state = self.lstm(lstm_input, (hidden_state, cell_state))

        #Compute the output
        output = self.W_out(hidden_state)  # Shape: [batch_size, output_size]
        output_probs = torch.softmax(output, dim=1)  # Shape: [1, output_size]

        return output_probs, hidden_state, cell_state, output

def train_encdecoder(cnn, cnnweights, datasets, dataloaders, criterion, epochs, save_path, symbols, symbols_dataloader):
    #cnn.load_state_dict(torch.load(cnnweights), strict=False)
    cnn.eval()
    image, _ = next(iter(dataloaders[0]))    
    cnnoutput = cnn(image)                   #shape per batch: [batch_size of 16, 512 features, 49 height, 49 width]
    all_symbol_cnn_outputs = []

    #Below would save the features of all symbols in a tensor, this way based on the index predicted by the decoder, we could get the predicted symbol's features
    #And feed it to the decoder for further generation
    with torch.no_grad():
       for idx, (images, _) in enumerate(symbols_dataloader):
            print("loaded")
            output = cnn(images)
            all_symbol_cnn_outputs.append(output.cpu()) 
            torch.cuda.empty_cache()
    
    torch.save(all_symbol_cnn_outputs, "all_symbol_cnn_outputs.pt") 

    encoder = RowEncoder(num_features = cnnoutput.shape[1], hidden_size = 256)
    decoder = RowDecoder(hidden_size=512, output_size=len(symbols)+1, embedding_dim=512)
    batch_size = 16

    #Initialize hidden state, cell state, and start_token with random values since first symbol to be generated has no previous information to base the symbol on
    initial_hidden = torch.randn(1, 512)   
    initial_cell = torch.randn(1, 512) 
    start_token = torch.randn(1, 512)  
    optimizer = optim.SGD(list(cnn.parameters()) + list(encoder.parameters()) + list(decoder.parameters()), lr=0.1)

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        correct_hmes = 0 
        total_images = 0  
        train_loss = 0

        for items, labels in iter(dataloaders[0]):
            optimizer.zero_grad()
            cnnoutput = cnn(items)
            encoded_data = []
            
            #Process each image from the CNN output batch-wise
            for i in range(batch_size):
                new = encoder(cnnoutput[i:i+1])
                encoded_data.append(new)  # Process one image at a time


            for im in range(len(encoded_data)):
                hidden_state = initial_hidden
                cell_state = initial_cell
                decoder_input = start_token

                predicted_labels = [] 

                # Generate HME for the current image
                for t in range(labels.size(1)):  # Loop over time steps (tokens)
                    decoder_output, hidden_state, cell_state, raw_logits = decoder(decoder_input, hidden_state, cell_state, encoded_data[im])
                    
                    predicted_label = torch.argmax(decoder_output, dim=1)  #Shape: [batch_size] (index of the highest prob)
                    predicted_labels.append(predicted_label)
                    symbol_index = predicted_label.item()  #Convert tensor to scalar
                    symbol_image = all_symbol_cnn_outputs[symbol_index] 
                    decoder_input = symbol_image
                    decoder_input = symbol_image.sum(dim=[1, 2], keepdim=True) 
                    decoder_input = decoder_input.view(1, 512)  # Flatten the tensor to [1, 512] to have the feature vector of the generated symbol as input for the next symbol

                predicted_labels = torch.stack(predicted_labels).to(torch.float32) 
                
                labels_image = labels[im].view(-1).long() 

                #Loss calculation
                loss = criterion(predicted_labels, labels_image)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                true_labels = labels[im]  #True labels for this image

                if (predicted_labels.cpu().numpy() == true_labels.cpu().numpy()).all():
                    correct_hmes += 1  #Increment count of correct HMEs


            total_images += len(encoded_data)  # Update total image count

        accuracy = (correct_hmes / total_images) * 100  
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss / len(dataloaders[0]):.4f}, Accuracy: {accuracy:.2f}%")
    
    return None


# Create labelencoder 
label_names = []
with zipfile.ZipFile("mathwriting-2024-excerpt.zip", 'r') as zip_data:
    all_content = zip_data.namelist()
    for item in all_content:
        # Make sure item is not a directory
        if (item.startswith("test/") or 
            item.startswith("train/") or
            item.startswith("valid/") or
            item.startswith("symbols/")) and not item.endswith('/'):
            # Only check inkml files
            if item.endswith(('.inkml')):
                # Open the inkml file
                with zip_data.open(item) as inkml_file:
                    # Parse inkml file to get the annotation
                    tree = ET.parse(inkml_file)
                    root = tree.getroot()
                    namespaces = {'inkml': 'http://www.w3.org/2003/InkML'}
                    if item.startswith("symbols/"):
                        annotation = root.find('.//inkml:annotation[@type="label"]', namespaces)
                    else: 
                        annotation = root.find('.//inkml:annotation[@type="normalizedLabel"]', namespaces)
                    # Tokenize label
                    if annotation is not None:
                        annotation = annotation.text
                        annotation = tokenize_label(annotation)
                        label_names.append(annotation)

# Encode label
label_encoder = LabelEncoder()
# Flatten label_names, so all symbols can be encoded separately
label_names_flat = functools.reduce(operator.concat, label_names)
label_encoder.fit_transform(label_names_flat)

hme_test = HME("mathwriting-2024-excerpt.zip", "test/", label_encoder, datatype='k')
hme_test_dataloader = DataLoader(hme_test, batch_size=16, shuffle=False, drop_last=True)
torch.cuda.empty_cache()

symbols = HME("mathwriting-2024-excerpt.zip", "symbols/", label_encoder, datatype='symbols')
symbols_dataloader = DataLoader(symbols, batch_size=32, shuffle=False, drop_last=False)

train_encdecoder(CNN3(), "cnn_sgd1_3e.pth", [hme_test], [hme_test_dataloader], nn.CrossEntropyLoss(), 10, "", [symbols], symbols_dataloader)
