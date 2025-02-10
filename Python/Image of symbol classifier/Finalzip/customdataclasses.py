import os
import zipfile
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
from PIL import Image
import functools
import operator
from sklearn.preprocessing import LabelEncoder

# To create subset of MNIST
class MNISTSubset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image) 
        else:
            image = transforms.ToTensor()(image)
        return image, label

# To create subset of Mathwriting-2024
class Subset(Dataset):
    def __init__(self, file, data, targets, transform=None):
        self.file = file
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data[idx]
        label = self.targets[idx]
        try:
            with zipfile.ZipFile(self.file, 'r') as zip_data:
                with zip_data.open(file_name) as file:
                    image = Image.open(file).convert('L')
                    if self.transform:
                        image = self.transform(image)
                    else:
                        image = transforms.ToTensor()(image)
        except KeyError:
            # If image cannot be found, return an empty image
            print(f"Error: Failed to open {file_name}.")
            image = Image.new('L', (256, 256), color=255) 
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
        return image, label

# To process Mathwriting-2024 
class HME(Dataset):

    def __init__(self, images, data, path_to_data, transform=None):
        self.images = images
        self.data = data
        self.path_to_data = path_to_data
        self.transform = transform
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
                        base = os.path.splitext(item)[0]
                        file_names.append(base+'.png')
                        # Open the inkml file
                        with zip_data.open(item) as inkml_file:
                            # Parse inkml file to get the annotation
                            tree = ET.parse(inkml_file)
                            root = tree.getroot()
                            namespaces = {'inkml': 'http://www.w3.org/2003/InkML'}
                            annotation = root.find('.//inkml:annotation[@type="label"]', namespaces)
                            # Tokenize label
                            if annotation is not None:
                                annotation = annotation.text
                                annotationlst = self.__tokenize_label(annotation)
                            label_names.append(annotationlst)
        # Encode label
        self.label_encoder = LabelEncoder()
        # Flatten label_names, so all symbols can be encoded separately
        label_names_flat = functools.reduce(operator.concat, label_names)
        self.label_encoder.fit_transform(label_names_flat)
        # Encode label_names and pad it so all labels have the same length
        label_names_encoded = [self.label_encoder.transform(lst) for lst in label_names]
        label_names_encoded = [torch.tensor(lst) for lst in label_names_encoded]
        padded_labels = pad_sequence(label_names_encoded, batch_first=True, padding_value=0)
        # Collect label mapping
        label_mapping = {code: label for label, code in zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_)))}
        return file_names, padded_labels, label_mapping

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        label = self.label_names[idx]
        with zipfile.ZipFile(self.images, 'r') as zip_data:
            with zip_data.open(file_name) as file:
                image = Image.open(file).convert('L')
                if self.transform:
                    image = self.transform(image)
                else:
                    image = transforms.ToTensor()(image)

        return image, label
    
    def _getlabel(self, encoded_label):
        label = ""
        # Loop through encoded label and translate everything
        for elem in encoded_label:
            if elem != len(self.label_encoder.classes_):
                label += self.label_mapping[int(elem)]
        return label
    
    def _get_unique_labels(self):
        unique_labels = set()
        for label in self.label_names:
            converted_tuple = tuple(t.item() for t in label)
            unique_labels.add(converted_tuple)
        return unique_labels
