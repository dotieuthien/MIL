import time
import os
import copy
import matplotlib.pyplot as plt
import argparse
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
from models.model_transfer_simple import DressedQuantumNet

torch.manual_seed(42)
np.random.seed(42)


# OpenMP: number of parallel threads.
os.environ["OMP_NUM_THREADS"] = "1"

step = 0.0005               # Learning rate
batch_size = 4               # Number of samples for each training step
num_epochs = 400
gamma_lr_scheduler = 0.1    # Learning rate reduction applied every 10 epochs.
start_time = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('runs/exp_4/exp_quantum')

# Initialize dataloader
img_dir = "/home/hades/Desktop/quantum-neural-network/data/tile/k1_train"
label_file = "/home/hades/Desktop/quantum-neural-network/data/tile/k1_train/top1_k1N2_train.csv"
dataloader = PathologyLoader(batch_size, True, img_dir, label_file, 'train').loader()
data_size = dataloader.dataset.__len__()

val_img_dir = '/home/hades/Desktop/quantum-neural-network/data/tile/k1_val'
val_label_file = '/home/hades/Desktop/quantum-neural-network/data/tile/k1_val/top1_k1N2_val.csv'
valid_dataloader = PathologyLoader(1, False, val_img_dir, val_label_file, 'valid').loader()
val_data_size = valid_dataloader.dataset.__len__()


def imshow_tensor_img(tensor):
    """Display image from tensor
    """
    transform = transforms.ToPILImage()
    img = transform(tensor).convert('RGB')
    plt.imshow(img)
    plt.show()


# Model
model_hybrid = torchvision.models.resnet18(pretrained=True)

for param in model_hybrid.parameters():
    param.requires_grad = False

# Notice that model_hybrid.fc is the last layer of ResNet18
model_hybrid.fc = DressedQuantumNet()
# model_hybrid.fc = nn.Linear(512, 2)

# Use CUDA or CPU according to the "device" object.
model_hybrid = model_hybrid.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_hybrid = optim.Adam(model_hybrid.fc.parameters(), lr=step)
exp_lr_scheduler = lr_scheduler.StepLR(
    optimizer_hybrid, step_size=10, gamma=gamma_lr_scheduler
)


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    parser = argparse.ArgumentParser(description="Trains the quantum Resnet model.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--pretrained_weights", type=str,
                        help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 10000.0  # Large arbitrary number
    best_acc_train = 0.0
    best_loss_train = 10000.0  # Large arbitrary number
    print("Training started:")

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for batch_id, (inputs, labels) in enumerate(dataloader):
            batch_size_ = len(inputs)
            inputs = inputs.to(device)
            # imshow_tensor_img(inputs[0].cpu())
            labels = labels.to(device)
            optimizer.zero_grad()

            # Track/compute gradient and make an optimization step only when training
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print iteration results
            running_loss += loss.item() * batch_size_
            batch_corrects = torch.sum(preds == labels.data).item()
            running_corrects += batch_corrects

        # Print epoch results
        epoch_loss = running_loss / data_size
        epoch_acc = running_corrects / data_size


        print(
            "Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}".format(
                epoch + 1,
                num_epochs,
                epoch_loss,
                epoch_acc,
            )
        )

        writer.add_scalar('training acc', epoch_acc, epoch)
        writer.add_scalar('training loss', epoch_loss, epoch)

        if epoch_acc > best_acc_train:
            best_acc_train = epoch_acc

        if epoch_loss < best_loss_train:
            best_loss_train = epoch_loss

        valid_loss, valid_acc = valid_model(valid_dataloader, model, criterion)

        print(
            "Eval Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}".format(
                epoch + 1,
                num_epochs,
                valid_loss,
                valid_acc,
            )
        )
        writer.add_scalar('valid acc', valid_acc, epoch)
        writer.add_scalar('valid loss', valid_loss, epoch)

        # Update learning rate
        # scheduler.step()



    # Print final results
    model.load_state_dict(best_model_wts)
    time_elapsed = time.time() - since
    print(
        "Training completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
    )


def valid_model(valid_dataloader, model, criterion):
    model.eval()
    running_loss = 0
    running_corrects = 0

    with torch.no_grad():
        for batch_id, (inputs, labels) in enumerate(valid_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Track/compute gradient and make an optimization step only when training
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            batch_corrects = torch.sum(preds == labels.data).item()
            running_corrects += batch_corrects

    running_loss = running_loss / val_data_size
    running_corrects = running_corrects / val_data_size

    return running_loss, running_corrects


if __name__ == '__main__':
    train_model(model_hybrid, criterion, optimizer_hybrid, exp_lr_scheduler, num_epochs=num_epochs)




