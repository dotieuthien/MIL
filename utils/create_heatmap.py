from json import tool
import time
import os
import copy
from tokenize import group
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
import argparse
import cv2
from xml.dom import minidom
import numpy as np
import configparser
from openslide import open_slide
from loaders.slide_loader import rgba2rgb

import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms

# import pennylane as qml
# from pennylane import numpy as np
# from models.qresnet import DressedQuantumNet


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


def get_box_from_min_size(slide_img):
    """Get coordinate of foreground and compute the sclae factor

    Args:
        slide_img ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Resolution of whole-slide
    max_size = slide_img.level_dimensions[0]
    min_size = slide_img.level_dimensions[-1]

    print('Size of level 0 is ', max_size)
    print('Size of min level is ', min_size)

    # Scale factor between min and max resolution
    w_scale = max_size[0] / min_size[0]
    h_scale = max_size[1] / min_size[1]

    print('Scale factor is ', w_scale, h_scale)

    min_img = slide_img.read_region((0, 0), slide_img.level_count - 1, slide_img.level_dimensions[-1])
    min_img = np.array(min_img)

    # plt.imshow(min_img)
    # plt.show()

    # Get foreground
    foreground_img = segment_foreground(min_img)
    coords = np.where(foreground_img == 255)
    rows, cols = coords
    coords = (cols, rows)

    # fig = plt.figure()
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(min_img)
    # fig.add_subplot(1, 2, 2)
    # plt.imshow(foreground_img)
    # plt.show()
    
    return coords, w_scale, h_scale


def get_batch_of_patches(w_scale, h_scale, coord, slide, trans_func):
    """Concat all patches in a region to a batch

    Args:
        w_scale ([type]): [description]
        h_scale ([type]): [description]
        coord ([type]): [description]
        slide ([type]): [description]
        trans_func ([type]): [description]

    Returns:
        [type]: [description]
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
        box_img = slide.read_region((top_left[0], top_left[1]), 0, (224, 224))
        print(top_left)
        box_img = np.array(box_img)
        box_img = rgba2rgb(box_img)

        # plt.imshow(box_img)
        # plt.show()

        # Convert image to tensor
        box_tensor = trans_func(box_img)
        box_tensor = torch.unsqueeze(box_tensor, dim=0)

        if index == 0:
            batch = box_tensor
        else:
            batch = torch.cat((batch, box_tensor), 0)
    return batch


def compute_heatmap(slide_path, checkpoint_path, output_path):
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
    slide_name = os.path.basename(slide_path)[:-5]
    heatmap_path1 = os.path.join(output_path, slide_name + '.heatmap1.png')
    heatmap_path2 = os.path.join(output_path, slide_name + '.heatmap2.png')
    print(slide_name, heatmap_path2)

    # coords format [WIDTH, HEIGHT]
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
            # WIDTH, HEIGHT
            coord = [coords[0][box_id], coords[1][box_id]]

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

    cv2.imwrite(heatmap_path1, heatmap1)
    cv2.imwrite(heatmap_path2, heatmap2)
    return


def show_3d(prob_heatmap_path):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    from matplotlib import cm

    prob_heatmap = 255 - np.array(cv2.imread(prob_heatmap_path, 0), np.uint8)
    prob_heatmap = cv2.threshold(prob_heatmap, 70, 255, cv2.THRESH_BINARY)[1]
    plt.imshow(prob_heatmap)
    plt.show()

    h, w = prob_heatmap.shape
    x, y = np.meshgrid(np.linspace(0,w, w), np.linspace(0,h,h))
    print(x.shape, y.shape)
    z = prob_heatmap
    print(z.shape)

    # create the figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()


def create_xml(binary_heatmap_path, prob_heatmap_path, slide_path, outpath):
    """[summary]

    Args:
        binary_heatmap_path ([type]): [description]
        prob_heatmap_path ([type]): [description]
        slide_path ([type]): [description]
    """
    # Read slide
    slide_img = open_slide(slide_path)
    slide_name = os.path.basename(slide_path)[:-5]

    # Resolution of whole-slide
    max_size = slide_img.level_dimensions[0]
    min_size = slide_img.level_dimensions[-1]

    # Scale factor between min and max resolution
    w_scale = max_size[0] / min_size[0]
    h_scale = max_size[1] / min_size[1]
    
    # Read heatmap
    heatmap1 = cv2.imread(binary_heatmap_path, 0)
    plt.imshow(heatmap1)
    plt.show()

    # tumor
    t_X, t_Y = np.where(heatmap1 == 119)

    root = minidom.Document()
    
    # Create ASAP
    asap = root.createElement('ASAP_Annotations') 
    root.appendChild(asap)
    
    # Create Anns
    anns = root.createElement('Annotations')
    asap.appendChild(anns)

    # Create Groups
    ann_groups = root.createElement('AnnotationGroups')
    asap.appendChild(ann_groups)

    # Add Annotation to Annotations
    num_tumor = len(t_X)

    for i in range(num_tumor):
        ann = root.createElement('Annotation')
        anns.appendChild(ann)
        ann.setAttribute('Name', 'tumor' + str(i + 1))
        ann.setAttribute('Type', 'Rectangle')
        ann.setAttribute('PartOfGroup', 'tumor')
        ann.setAttribute('Color', '#f4fa58')

         # Add Coordinates to Annotation
        for j in range(1):
            coords = root.createElement('Coordinates')
            ann.appendChild(coords)

            # Add Coordinate to Coordinates
            min_x = t_X[i] * h_scale
            min_y = t_Y[i] * w_scale
            max_x = min_x + h_scale
            max_y = min_y + w_scale
            
            fake_X = [str(min_y), str(max_y), str(max_y), str(min_y)]
            fake_Y = [str(min_x), str(min_x), str(max_x), str(max_x)]

            for t in range(4):
                coord = root.createElement('Coordinate')
                coords.appendChild(coord)
                coord.setAttribute('Order', str(t))
                coord.setAttribute('X', fake_X[t])
                coord.setAttribute('Y', fake_Y[t])

    # Add Group to AnnotationGroups
    class_names = ['tumor', 'normal']
    class_colors = ['#f4fa58', '#aa0000']
    for i in range(2):
        gr = root.createElement('Group')
        ann_groups.appendChild(gr)
        att = root.createElement('Attributes')
        gr.appendChild(att)
        gr.setAttribute('Name', class_names[i])
        gr.setAttribute('PartOfGroup', 'None')
        gr.setAttribute('Color', class_colors[i])
    
    xml_str = root.toprettyxml(indent ="\t") 
    save_path_file = os.path.join(outpath, slide_name + '.xml')
    
    with open(save_path_file, "w") as f:
        f.write(xml_str)


# def create_prob_xml(
#     prob_heatmap_path, 
#     slide_path, 
#     outpath):
#     """

#     Args:
#         prob_heatmap_path (str): path to probability map
#         slide_path (str): path to whole slide
#         outpath (str): output path
#     """
#     # Read slide
#     slide_img = open_slide(slide_path)
#     slide_name = os.path.basename(slide_path)[:-5]

#     # Resolution of whole-slide
#     max_size = slide_img.level_dimensions[0]
#     min_size = slide_img.level_dimensions[-1]

#     # Scale factor between min and max resolution
#     w_scale = max_size[0] / min_size[0]
#     h_scale = max_size[1] / min_size[1]
    
#     # Read heatmap
#     heatmap2 = cv2.imread(prob_heatmap_path)
#     gray_heatmap2 = cv2.cvtColor(heatmap2, cv2.COLOR_BGR2GRAY)
    
#     # Tumor coordinates
#     t_X, t_Y = np.where(gray_heatmap2 != 255)

#     root = minidom.Document()
    
#     # Create ASAP
#     asap = root.createElement('ASAP_Annotations') 
#     root.appendChild(asap)
    
#     # Create Anns
#     anns = root.createElement('Annotations')
#     asap.appendChild(anns)

#     # Create Groups
#     ann_groups = root.createElement('AnnotationGroups')
#     asap.appendChild(ann_groups)

#     # Add Annotation to Annotations
#     num_tumor = 200

#     for i in range(num_tumor):
#         # Probability
#         prob_color = heatmap2[t_X[i], t_Y[i], :] / 255
#         b, g, r = prob_color
#         hex_color = rgb2hex((r, g, b))
#         ann = root.createElement('Annotation')
#         anns.appendChild(ann)
#         ann.setAttribute('Name', 'tumor' + str(i + 1))
#         ann.setAttribute('Type', 'Rectangle')
#         ann.setAttribute('PartOfGroup', 'tumor')
#         ann.setAttribute('Color', str(hex_color))

#         # Add Coordinates to Annotation
#         for j in range(1):
#             coords = root.createElement('Coordinates')
#             ann.appendChild(coords)

#             # Add Coordinate to Coordinates
#             min_x = t_X[i] * h_scale
#             min_y = t_Y[i] * w_scale
#             max_x = min_x + h_scale
#             max_y = min_y + w_scale
#             fake_X = [str(min_y), str(max_y), str(max_y), str(min_y)]
#             fake_Y = [str(min_x), str(min_x), str(max_x), str(max_x)]

#             for t in range(4):
#                 coord = root.createElement('Coordinate')
#                 coords.appendChild(coord)
#                 coord.setAttribute('Order', str(t))
#                 coord.setAttribute('X', fake_X[t])
#                 coord.setAttribute('Y', fake_Y[t])

#     # Add Group to AnnotationGroups
#     class_names = ['tumor', 'normal']
#     class_colors = ['#f4fa58', '#aa0000']
#     for i in range(2):
#         gr = root.createElement('Group')
#         ann_groups.appendChild(gr)
#         att = root.createElement('Attributes')
#         gr.appendChild(att)
#         gr.setAttribute('Name', class_names[i])
#         gr.setAttribute('PartOfGroup', 'None')
#         gr.setAttribute('Color', class_colors[i])
    
#     xml_str = root.toprettyxml(indent ="\t") 
#     save_path_file = os.path.join(outpath, slide_name + '.xml')
    
#     with open(save_path_file, "w") as f:
#         f.write(xml_str)


if __name__ == '__main__':
    create_xml('/mnt/d/q/data_test/slide/predictions/B2018.28271_7C.binary.png', 
    '/mnt/d/q/data_test/slide/predictions/B2018.28271_7C.prob.png',
    '/mnt/d/q/data_test/slide/mrxs/B2018.28271_7C.mrxs',
    '/mnt/d/q/data_test/slide/mrxs')

    # show_3d("/mnt/d/q/data_test/slide/mrxs/B2018.28271_10B.heatmap2_.png")
    # show_3d("/mnt/d/q/data_test/slide/mrxs/B00.12534_A.heatmap2_.png")

    # compute_heatmap('/mnt/d/q/data_test/slide/mrxs/B2018.28271_7B.mrxs',
    # '/mnt/d/q/runs/exp_qresnet3/best_checkpoint_14.pth',
    # '/mnt/d/q/data_test/slide/mrxs')

    # debug_heatmap('/home/hades/Desktop/q/data/slide/mrxs/B2018.18540_10B.mrxs',
    # '/home/hades/Desktop/q/data/slide/mrxs/B2018.18540_10B.heatmap1_.png')