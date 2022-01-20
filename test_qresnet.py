import time
import os
import copy
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
import configparser
from openslide import open_slide
from data_loaders.slide_loader import get_box_from_min_size, rgba2rgb
from data_loaders.tile_loader import PathologyLoader

import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms

import pennylane as qml
from pennylane import numpy as np
from models.qresnet import DressedQuantumNet


def predict(tile_image, checkpoint_path):
    """Tile classification: Tumor or Normal

    Args:
        tile_image ([type]): [description]
        checkpoint_path ([type]): [description]
    Returns:

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('The device for prediction is %s' % device)

    # Load model qresnet
    model_hybrid = torchvision.models.resnet18()
    model_hybrid.fc = DressedQuantumNet()
    model_hybrid.load_state_dict(torch.load(checkpoint_path))

    model_hybrid = model_hybrid.to(device)
    
    with torch.no_grad():
