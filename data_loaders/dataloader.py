import os
import glob
import csv
from PIL import Image
import numpy as np

# PyTorch
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

# Plotting
import matplotlib.pyplot as plt


class Pathology(Dataset):
    def __init__(self, img_dir, label_file, mode):
        # Read label file
        self.img_dir = img_dir
        self.mode = mode
        label_file = open(label_file)
        csv_reader = csv.reader(label_file)
        headers = next(csv_reader)
        print('All headers of label file ', headers)
        self.img_names = []
        self.labels = []

        for row in csv_reader:
            self.img_names.append(row[0])
            self.labels.append(int(row[2]))

        self.img_transforms = transforms.Compose(
                [
                    # transforms.RandomResizedCrop(224),     # uncomment for data augmentation
                    # transforms.RandomHorizontalFlip(),     # uncomment for data augmentation
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    # Normalize input channels using mean values and standard deviations of ImageNet.
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        if self.mode == 'train':
            img_name = 'toptile_k1N2_train_' + self.img_names[item] + '.png'
        elif self.mode == 'valid':
            img_name = 'toptile_k1N2_val_' + self.img_names[item] + '.png'
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        label = self.labels[item]

        # Convert to tensor
        img_tensor = self.img_transforms(image)
        label_tensor = torch.tensor(label)

        return img_tensor, label_tensor


class PathologyLoader(object):
    def __init__(self, batch_size, shuffle, img_dir, label_file, mode):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = Pathology(img_dir, label_file, mode)

    def loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)


if __name__ == '__main__':
    img_dir = '/home/hades/Desktop/quantum-neural-network/data/Quantum/k1_train'
    label_file = '/home/hades/Desktop/quantum-neural-network/data/Quantum/k1_train/top1_k1N2_train.csv'

    # Test data loader
    dataloader = PathologyLoader(3, True, img_dir, label_file).loader()
    for img, label in dataloader:
        print(label)

