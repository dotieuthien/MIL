import time
import os
import copy
import matplotlib.pyplot as plt
import argparse
import configparser
from data_loaders.dataloader import PathologyLoader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import pennylane as qml
from pennylane import numpy as np
from models.qresnet import DressedQuantumNet


config = configparser.ConfigParser()
config.read('configs/qresnet.ini')

# OpenMP: number of parallel threads.
os.environ["OMP_NUM_THREADS"] = "1"


def imshow_tensor_img(tensor):
    """Display image from tensor
    """
    transform = transforms.ToPILImage()
    img = transform(tensor).convert('RGB')
    plt.imshow(img)
    plt.show()


def train_model():
    parser = argparse.ArgumentParser(description="Trains the quantum Resnet model.")
    parser.add_argument("--training_img_dir", type=str, default='data/tile/k1_train', help="Directory to images")
    parser.add_argument("--training_label_dir", type=str, default='data/tile/k1_train/top1_k1N2_train.csv',
                        help="Directory to labels")
    parser.add_argument("--val_img_dir", type=str, default='data/tile/k1_val', help="Directory to images")
    parser.add_argument("--val_label_dir", type=str, default='data/tile/k1_val/top1_k1N2_val.csv',
                        help="Directory to labels")
    parser.add_argument("--runs", type=str, default='runs/exp_1/exp_quantum',
                        help="Directory to tensorboard folder")
    args = parser.parse_args()

    # Initialize dataloader
    dataloader = PathologyLoader(config.getint('TRAIN', 'batch_size'), True, args.training_img_dir, args.training_label_dir, 'train').loader()
    valid_dataloader = PathologyLoader(1, False, args.val_img_dir, args.val_label_dir, 'valid').loader()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(args.runs)

    # Model
    model_hybrid = torchvision.models.resnet18(pretrained=True)
    if not config.getboolean('TRAIN', 'requires_grad'):
        for param in model_hybrid.parameters():
            param.requires_grad = False

    # Notice that model_hybrid.fc is the last layer of ResNet18
    model_hybrid.fc = DressedQuantumNet()
    # model_hybrid.fc = nn.Linear(512, 2)

    # Use CUDA or CPU according to the "device" object.
    model_hybrid = model_hybrid.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_hybrid.fc.parameters(), lr=config.getfloat('TRAIN', 'step'))
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=config.getfloat('TRAIN', 'gamma_lr_scheduler')
    )

    since = time.time()
    best_acc_train = 0.0
    best_loss_train = 10000.0  # Large arbitrary number

    num_epochs = config.getint('TRAIN', 'num_epochs')

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        model_hybrid.train()
        training_loss = 0
        training_acc = 0

        for batch_id, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            # imshow_tensor_img(inputs[0].cpu())
            labels = labels.to(device)
            optimizer.zero_grad()

            # Track/compute gradient and make an optimization step only when training
            outputs = model_hybrid(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update iteration results
            training_loss += loss.item()
            batch_corrects = torch.sum(preds == labels.data).item()
            training_acc += batch_corrects / labels.size(0)

        training_loss = training_loss / (batch_id + 1)
        training_acc = training_acc / (batch_id + 1)

        print(
            "Train: {}/{} Loss: {:.4f} Acc: {:.4f}".format(
                epoch + 1,
                num_epochs,
                training_loss,
                training_acc,
            )
        )

        writer.add_scalar('training acc', training_acc, epoch)
        writer.add_scalar('training loss', training_loss, epoch)

        valid_loss, valid_acc = valid_model(valid_dataloader, model_hybrid, device, criterion)

        print(
            "Eval: {}/{} Loss: {:.4f} Acc: {:.4f}".format(
                epoch + 1,
                num_epochs,
                valid_loss,
                valid_acc,
            )
        )
        writer.add_scalar('valid acc', valid_acc, epoch)
        writer.add_scalar('valid loss', valid_loss, epoch)

        if valid_acc > best_acc_train and valid_loss < best_loss_train:
            best_acc_train = valid_acc
            best_loss_train = valid_loss
            checkpoint_path = os.path.join(args.runs, "best_checkpoint.pth")
            torch.save(model_hybrid.state_dict(), checkpoint_path)

        # Update learning rate
        exp_lr_scheduler.step()

    time_elapsed = time.time() - since
    print(
        "Training completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
    )


def valid_model(valid_dataloader, model, device, criterion):
    model.eval()
    valid_loss = 0
    valid_acc = 0

    with torch.no_grad():
        for batch_id, (inputs, labels) in enumerate(valid_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()
            batch_corrects = torch.sum(preds == labels.data).item()
            valid_acc += batch_corrects

    valid_loss = valid_loss / (batch_id + 1)
    valid_acc = valid_acc / (batch_id + 1)

    return valid_loss, valid_acc


if __name__ == '__main__':
    train_model()




