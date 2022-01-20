import time
import os
import copy
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
import configparser
from openslide import open_slide
from data_loaders.slide_loader import rgba2rgb

import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms

import pennylane as qml
from pennylane import numpy as np
from models.qresnet import DressedQuantumNet


def segment_foreground(img, mthresh=7, sthresh=5, sthresh_up=255):
    """Segment to get foreground of the whole slide

    Args:
        img ([type]): [description]
        mthres
        sthresh (int, optional): [description]. Defaults to 20.
        sthresh_up (int, optional): [description]. Defaults to 255.
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
    img_med = cv2.medianBlur(img_hsv[:,:,1], mthresh)  # Apply median blurring
    
    # Thresholding
    _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)            

    # fig = plt.figure()
    # fig.add_subplot(1, 3, 1)
    # plt.imshow(img)
    # fig.add_subplot(1, 3, 2)
    # plt.imshow(img_med)
    # fig.add_subplot(1, 3, 3)
    # plt.imshow(img_otsu)
    # plt.show()

    return img_otsu


def get_box_from_min_size(slide_img, max_size_box=224):
    """Split images into patches with size = 224

    Args:
        slide_img : whole-slide image
        max_size_box (int, optional): size of each patch to split. Defaults to 224.

    Returns:
        coords [list]: [j: width, i: height]
        w_scale, h_scale
    """
    # Resolution of whole-slide
    max_size = slide_img.level_dimensions[0]
    min_size = slide_img.level_dimensions[-1]

    # Scale factor between min and max resolution
    w_scale = max_size[0] / min_size[0]
    h_scale = max_size[1] / min_size[1]

    min_img = slide_img.read_region((0, 0), slide_img.level_count - 1, slide_img.level_dimensions[-1])
    min_img = np.array(min_img)

    # Get foreground
    foreground_img = segment_foreground(min_img)
    debug_img = np.zeros((min_size[1], min_size[0], 3))

    coords = np.where(foreground_img == 255)

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(min_img)
    fig.add_subplot(1, 2, 2)
    plt.imshow(foreground_img)
    plt.show()
    
    return coords, w_scale, h_scale


def get_batch_of_patches(w_scale, h_scale, coord, slide, trans_func):
    """[summary]
    """
    # A pixel in min image represents for (scale factor / patches size) patches
    patch_per_pixel_w = round(w_scale / 224)
    patch_per_pixel_h = round(h_scale / 224)

    top_left_coords = []
    if patch_per_pixel_w != patch_per_pixel_h:
        top_left = (int(coord[0] * w_scale), int(coord[1] * h_scale))
        top_left_coords.append(top_left)
    else:
        str_w, str_h = int(coord[0] * w_scale), int(coord[1] * h_scale)
        for p_w in range(patch_per_pixel_w):
            for p_h in range(patch_per_pixel_h):
                top_left = (str_w + p_w * 224, str_h + p_h * 224)
                top_left_coords.append(top_left)

    # Get the batch of patch
    batch = None
    for index, top_left in enumerate(top_left_coords):
        box_img = slide.read_region(top_left, 0, (224, 224))
        box_img = np.array(box_img)
        box_img = rgba2rgb(box_img)

        # Convert image to tensor
        box_tensor = trans_func(box_img)
        box_tensor = torch.unsqueeze(box_tensor, dim=0)

        if index == 0:
            batch = box_tensor
        else:
            batch = torch.cat((batch, box_tensor), 0)
    return batch


def patching(slide_path, checkpoint_path):
    """

    Args:
        slide_path ([type]): [description]
        checkpoint_path ([type]): [description]
    """

    print("Creating patches ...")

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
    num_coord = len(coords[0])
    w, h = slide.level_dimensions[-1]
    # classification map for tumor and normal
    heatmap1 = np.ones((h, w, 3), dtype=np.uint8) * 255
    # Prob map
    heatmap2 = np.ones((h, w, 3), dtype=np.uint8) * 255

    model_hybrid.eval()
    trans_func = transforms.ToTensor()

    with torch.no_grad():
        for box_id in range(num_coord):
            print('Process box %d/%d' % (box_id, num_coord))
            coord = [coords[1][box_id], coords[0][box_id]]

            batch = get_batch_of_patches(w_scale, h_scale, coord, slide, trans_func)
            batch = batch.to(device)

            outputs = model_hybrid(batch)
            softmax = nn.Softmax(dim=1)
            outputs = softmax(outputs)
            
            _, preds = torch.max(outputs, 1)

            # Prob of normal class
            outputs = torch.mean(outputs, dim=0, keepdim=True)
            np_outputs = outputs.cpu().detach().numpy()
            normal_prob = int(np_outputs[0][0] * 255)
            normal_color = [255, normal_prob, 255]
            heatmap2[coord[1], coord[0]] = normal_color

            num_tumor = torch.count_nonzero(preds)
            num_tumor = num_tumor.cpu().detach().numpy()
            # 0 is normal class, other is tumor class
            if num_tumor == 0:
                heatmap1[coord[1], coord[0], :] = [200, 10, 10]
            else:
                heatmap1[coord[1], coord[0], :] = [10, 100, 200]

    cv2.imwrite('/home/hades/Desktop/q/data/slide/mrxs/B00.12534_A.heatmap1_.png', heatmap1)
    cv2.imwrite('/home/hades/Desktop/q/data/slide/mrxs/B00.12534_A.heatmap2_.png', heatmap2)
    return


if __name__ == '__main__':
    patching('/home/hades/Desktop/q/data/slide/mrxs/B00.12534_A.mrxs',
    '/home/hades/Desktop/q/runs/exp_qresnet3/best_checkpoint_14.pth')