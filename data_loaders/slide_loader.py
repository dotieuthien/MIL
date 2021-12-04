from openslide import open_slide
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def load_csv(slide_path):
    img = open_slide(slide_path)
    # print('Levels: ', img.level_count)
    # print('Size: ', img.level_dimensions)
    if img.level_count < 3:
        return
    sub_img = img.read_region((0, 0), 2, img.level_dimensions[2])
    # plt.imshow(sub_img)
    # plt.show()
    return sub_img


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

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img_path = self.img_names[item]
        img = np.array(load_csv(img_path))
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
    load_csv('/home/hades/Desktop/quantum-neural-network/data/slide/test.svs')
    img_dir = '/home/hades/Desktop/quantum-neural-network/data/slide'
    dataloader = SlideLoader(1, True, img_dir).loader()

    for img in dataloader:
        print(img.size())
        # Convert to image
        debug_trans = transforms.ToPILImage()
        np_img = debug_trans(img[0])
        plt.imshow(np_img)
        plt.show()