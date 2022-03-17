import time
import os
import copy
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
import configparser
from openslide import open_slide
from loaders.slide_loader import get_box_from_min_size, rgba2rgb

import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms

import pennylane as qml
from pennylane import numpy as np
from models.qresnet import DressedQuantumNet


config = configparser.ConfigParser()
config.read('configs/qresnet.ini')

# OpenMP: number of parallel threads.
os.environ["OMP_NUM_THREADS"] = "1"


def draw_heatmap(slide_path, checkpoint_path):
    """Draw heatmap 

    Args:
        slide_path (string): [description]
        checkpoint_path (string):
    Returns:
        heatmap (narray):
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('The device for prediction is %s' % device)

    # Load model
    model_hybrid = torchvision.models.resnet18()
    model_hybrid.fc = DressedQuantumNet()
    model_hybrid.load_state_dict(torch.load(checkpoint_path))

    # Use CUDA or CPU according to the "device" object.
    model_hybrid = model_hybrid.to(device)

    # Read slide
    slide = open_slide(slide_path)
    # coords format [width, height]
    coords, w_scale, h_scale = get_box_from_min_size(slide)
    w, h = slide.level_dimensions[-1]
    heatmap1 = np.zeros((h, w), dtype=np.uint8)
    heatmap2 = np.ones((h, w, 3), dtype=np.uint8) * 255

    model_hybrid.eval()
    trans_func = transforms.ToTensor()

    with torch.no_grad():
        for box_id, coord in enumerate(coords):
            print('Process box %d/%d' % (box_id, len(coords)))
            # coord is the position in the low resolution slide
            # convert it to high resolution by mul with scale factor
            max_coord = (int(coord[0] * w_scale - 112), int(coord[1] * h_scale - 112))
            # Get the patch
            box_img = slide.read_region(max_coord, 0, (224, 224))
            box_img = np.array(box_img)
            box_img = rgba2rgb(box_img)
            gray_box_img = cv2.cvtColor(box_img, cv2.COLOR_RGB2GRAY)

            # Convert image to tensor
            box_img = trans_func(box_img)
            box_img = torch.unsqueeze(box_img, dim=0)

            input = box_img.to(device)
            outputs = model_hybrid(input)
            softmax = nn.Softmax(dim=1)
            outputs = softmax(outputs)
            _, preds = torch.max(outputs, 1)

            # Prob of tumor class
            tumor_prob = outputs.cpu().detach().numpy()
            tumor_prob = int(tumor_prob[0][0] * 255)
            tumor_color = [tumor_prob, tumor_prob, tumor_prob]
            heatmap2[coord[1], coord[0]] = tumor_color

            preds = preds.cpu().detach().numpy()

            if preds[0] == 0:
                heatmap1[coord[1], coord[0]] = 100
            else:
                heatmap1[coord[1], coord[0]] = 200

    colors = np.random.randint(0, 255, (int(np.max(heatmap1)) + 1, 3))
    heatmap1 = colors[heatmap1]

    cv2.imwrite('/home/hades/Desktop/q/data/slide/mrxs/B00.12534_A.heatmap1.png', heatmap1)
    cv2.imwrite('/home/hades/Desktop/q/data/slide/mrxs/B00.12534_A.heatmap2.png', heatmap2)
    return


if __name__ == '__main__':
    draw_heatmap('/home/hades/Desktop/q/data/slide/mrxs/B00.12534_A.mrxs',
    '/home/hades/Desktop/q/runs/exp_qresnet3/best_checkpoint_14.pth')
