import tarfile
import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder
import functools
import operator
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import os
from PIL import Image

class HME(Dataset):

    def __init__(self, images, data, path_to_data, transform=None):
        self.data = data
        self.images = images
        self.path_to_data = path_to_data
        self.transform = transform
        self.file_names, self.label_names, self.label_mapping = self._get_names()

    def __tokenize_label(self, label):
        print(label)
        print(len(label))
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
        with tarfile.open(self.data, 'r:gz') as zip_data:
            all_content = zip_data.getnames()
            for item in all_content:
                 base_name = os.path.splitext(item)[0]
                 file_names.append(base_name + ".png")
                 # Make sure item is not a directory
                 if item.startswith(self.path_to_data) and not item.endswith('/'):
                     # Only check inkml files
                     if item.endswith(('.inkml')):
                        # Open the inkml file
                        with zip_data.extractfile(item) as inkml_file:
                            # Parse inkml file to get the annotation
                            tree = ET.parse(inkml_file)
                            root = tree.getroot()
                            namespaces = {'inkml': 'http://www.w3.org/2003/InkML'}
                            annotation = root.find('.//inkml:annotation[@type="normalizedLabel"]', namespaces)
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
        with tarfile.open(self.images, 'r:gz') as zip_data:
            with zip_data.extractfile(file_name) as file:
                image = Image.open(file)
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

    def __init__(self, images, data, path_to_data, transform=None):
        self.data = data
        self.images = images
        self.path_to_data = path_to_data
        self.transform = transform
        self.file_names, self.label_names, self.label_mapping = self._get_names()

    def _get_names(self):
        # Collect files names and labels
        file_names = []
        label_names = []
        # Go through zip file
        with tarfile.open(self.data, 'r:gz') as zip_data:
            all_content = zip_data.getnames()
            for item in all_content:
                 # Make sure item is not a directory
                 if item.startswith(self.path_to_data) and not item.endswith('/'):
                     # Only check inkml files
                     if item.endswith(('.inkml')):
                        # Add file to list
                        base_name = os.path.splitext(item)[0]
                        file_names.append(base_name + ".png")
                        # Open the inkml file
                        with zip_data.extractfile(item) as inkml_file:
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
        with tarfile.open(self.images, 'r:gz') as zip_data:
            with zip_data.extractfile(file_name) as file:
                image = Image.open(file)
                if self.transform:
                    image = self.transform(image)
                else:
                    image = transforms.ToTensor()(image)

        return image, label

    def _getlabel(self, encoded_label):
        print(encoded_label)
        return self.label_mapping[encoded_label]
    
