import numpy as np
from openslide import open_slide
import matplotlib.pyplot as plt
import os, glob

from utils.create_heatmap import segment_foreground
from loaders.slide_loader import rgba2rgb
import torch


def create_data_mil(slide_folder, output_path):
    libraryfile = {}
    img_paths = glob.glob(os.path.join(slide_folder, '*/*.mrxs'))
    libraryfile['slides'] = img_paths
    libraryfile['grid'] = []
    libraryfile['targets'] = []
    libraryfile['mult'] = 1.
    libraryfile['level'] = 0

    for p in img_paths:
        slide_cls = p.split('/')[-2]
        print(slide_cls)
        grid = []
        # Read slide
        slide = open_slide(p)
        dims = slide.level_dimensions
        # Slide image of min resolution
        min_dim = dims[-1]
        min_img = np.array(slide.read_region((0, 0), len(dims) - 1, min_dim))
        # Get size of level 0
        dim_0 = dims[0]
        scale_factor = [dim_0[0] / min_dim[0], dim_0[1] / min_dim[1]]
        print(scale_factor)
        
        # Segment slide image
        min_img = rgba2rgb(min_img)
        foreground_img = segment_foreground(min_img)
        foreground_img = np.array(foreground_img)

        # plt.imshow(foreground_img)
        # plt.show()

        # Create meshgrid
        w, h = min_dim
        for i in range(h):
            for j in range(w):
                if j > 20 and i > 10:
                    continue

                if foreground_img[i, j] == 255:
                    min_level_coord = [j, i]
                    level_0_coord = np.array(scale_factor) * np.array(min_level_coord)
                    level_0_coord = np.array(level_0_coord, dtype=np.int)
                    level_0_coord = tuple(level_0_coord)
                    grid.append(level_0_coord)
        print(len(grid))
        libraryfile['grid'].append(grid)
        if slide_cls == 'normal_slides':
            libraryfile['targets'].append(0)
        elif slide_cls == 'tumor_slides':
            libraryfile['targets'].append(1)
            break

    torch.save(libraryfile, output_path)


if __name__ == "__main__":
    data_folder = '/home/thiendo/Desktop/pathology/MIL/data/slide-dataset'
    output_path = '/home/thiendo/Desktop/pathology/MIL/data/slide-dataset/tile.pth'

    create_data_mil(data_folder, output_path)