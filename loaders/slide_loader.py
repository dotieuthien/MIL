from openslide import open_slide
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def load_svs(slide_path):
    img = open_slide(slide_path)
    if img.level_count < 3:
        return
    sub_img = img.read_region((0, 0), 2, img.level_dimensions[2])
    return sub_img


def load_slide(slide_path):
    img = open_slide(slide_path)
    return img


# def get_box_from_min_size(slide_img, max_size_box=224):
#     """Split images into patches

#     Args:
#         slide_img : whole-slide image
#         max_size_box (int, optional): size of each patch to split. Defaults to 224.

#     Returns:
#         coords [list]: [j: width, i: height]
#         w_scale, h_scale
#     """
#     # Resolution of whole-slide
#     max_size = slide_img.level_dimensions[0]
#     min_size = slide_img.level_dimensions[-1]
#     # Scale factor between min and max resolution
#     w_scale = max_size[0] / min_size[0]
#     h_scale = max_size[1] / min_size[1]
#     # min_size_box = (max(1, max_size_box / w_scale), max(1, max_size_box / h_scale))

#     min_img = slide_img.read_region((0, 0), slide_img.level_count - 1, slide_img.level_dimensions[-1])
#     min_img = np.array(min_img)
#     debug_img = np.zeros((min_size[1], min_size[0], 3))

#     coords = []
#     for i in range(min_size[1]):
#         for j in range(min_size[0]):
#             # If the pixel in this position is white --> background
#             if sum(min_img[i, j]) / 4 > 250 or sum(min_img[i, j]) / 4 == 0:
#                 continue
#             else:
#                 coords.append([j, i])
#                 debug_img[i, j, :] = [0, 255, 255]

#     fig = plt.figure()
#     fig.add_subplot(1, 2, 1)
#     plt.imshow(min_img)
#     fig.add_subplot(1, 2, 2)
#     plt.imshow(debug_img)
#     plt.show()
    
#     return coords, w_scale, h_scale


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]
    a = np.asarray(a, dtype='float32') / 255.0
    R, G, B = background
    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')


class Slide(Dataset):
    def __init__(self, img_dir):
        self.block_size = 224
        self.img_dir = img_dir
        self.img_transforms = transforms.ToTensor()
        self.img_names = glob.glob(os.path.join(self.img_dir, '*.svs'))
        self.img_names = glob.glob(os.path.join(self.img_dir, '*.mrxs'))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img_path = self.img_names[item]
        slide = load_slide(img_path)

        img = np.array(load_svs(img_path))
        img = rgba2rgb(img)

        img_tensor = self.img_transforms(img)

        return img_tensor


class SlideLoader(object):
    def __init__(self, batch_size, shuffle, img_dir):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = Slide(img_dir)

    def loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)


if __name__ == "__main__":
    img_dir = '/home/hades/Desktop/q/data/slide/mrxs'
    dataset = Slide(img_dir)
    dataset.__getitem__(0)

    # dataloader = SlideLoader(1, True, img_dir).loader()
    #
    # for img in dataloader:
    #     print(img.size())
    #     # Convert to image
    #     debug_trans = transforms.ToPILImage()
    #     np_img = debug_trans(img[0])
    #     plt.imshow(np_img)
    #     plt.show()