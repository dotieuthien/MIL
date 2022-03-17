from statistics import mode
import time
import os
import copy
import matplotlib.pyplot as plt
import argparse
import configparser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import pennylane as qml
from pennylane import numpy as np

from sympy import arg
from loaders.tile_loader import PathologyLoader
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
    parser = argparse.ArgumentParser(description="Pytorch Training Classification Model")
    # Dataset
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--data-loader', default='Which data loader', type=str)
    parser.add_argument("--training-img-file", type=str, default='data/tile/k1_train', 
                        help="directory to images")
    parser.add_argument("--training-label-dir", type=str, default='data/tile/k1_train/top1_k1N2_train.csv',
                        help="directory to labels")
    parser.add_argument("--val-img-dir", type=str, default='data/tile/k1_val', 
                        help="directory to images")
    parser.add_argument("--val-label-file", type=str, default='data/tile/k1_val/top1_k1N2_val.csv',
                        help="directory to labels")

    # Optimization
    parser.add_argument('--optim', default='sgd', type=str,
                        help='Which optimizer')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                        help='train batch size (default: 256)')
    parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                        help='test batch size (default: 200)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    # parser.add_argument('--drop', '--dropout', default=0, type=float,
    #                     metavar='Dropout', help='Dropout ratio')
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    # Checkpoints
    parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("--runs", type=str, default='runs/exp_qresnet4',
                        help="Directory to tensorboard folder")

    # Architecture
    parser.add_argument('--arch', metavar='ARCH', default='resnet18',
                        help='model architecture')

    args = parser.parse_args()

    # Initialize dataloader
    if args.data_loader == 'from_csv':
        train_loader = PathologyLoader(
            batch_size=args.train_batch, 
            shuffle=True, 
            img_dir=args.training_img_dir, label_file=args.training_label_file, 
            mode='train').loader()

        val_loader = PathologyLoader(
            batch_size=args.test_batch, 
            shuffle=False,
            img_dir=args.val_img_dir, label_file=args.val_label_file,
            mode='valid').loader()

    elif args.data_loader == 'from_folder':
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(args.training_img_dir, transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.train_batch, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(args.val_img_dir, transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Tensorboard
    writer = SummaryWriter(args.runs)

    # Create model
    model_hybrid = torchvision.models.resnet18(pretrained=True)
    model_hybrid.fc = DressedQuantumNet()
    model_hybrid = model_hybrid.to(device)
    # model_hybrid.load_state_dict(torch.load('/home/hades/Desktop/q/runs/exp_qresnet3/best_checkpoint_14.pth'))

    # Define loss
    criterion = nn.CrossEntropyLoss()
    # Define optimizer
    if args.optim == 'adam':
        optimizer = optim.Adam(model_hybrid.parameters(), lr=config.getfloat('TRAIN', 'step'))
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model_hybrid.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer, 
        step_size=30, 
        gamma=config.getfloat('TRAIN', 'gamma_lr_scheduler')
    )

    since = time.time()
    best_acc_train = 0.0
    best_acc_valid = 0.0
    best_loss_valid = 10000.0  # Large arbitrary number

    num_epochs = config.getint('TRAIN', 'num_epochs')

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        model_hybrid.train()
        training_loss = 0
        training_acc = 0

        for batch_id, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # imshow_tensor_img(inputs[0].cpu())
            optimizer.zero_grad()
            # Feed
            outputs = model_hybrid(inputs)
            _, preds = torch.max(outputs, 1)
            # Compute loss
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

        valid_loss, valid_acc = valid_model(val_loader, model_hybrid, device, criterion)

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

        # Save the best
        if valid_acc > best_acc_valid and valid_loss < best_loss_valid:
            best_acc_valid = valid_acc
            best_loss_valid = valid_loss
            checkpoint_path = os.path.join(args.runs, "best_checkpoint_%s.pth" % str(epoch))
            torch.save(model_hybrid.state_dict(), checkpoint_path)
        elif valid_acc == best_acc_valid and training_acc > best_acc_train:
            best_acc_train = training_acc
            checkpoint_path = os.path.join(args.runs, "best_checkpoint_%s.pth" % str(epoch))
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
