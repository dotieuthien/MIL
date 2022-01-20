from utils.segment import segment_foreground
from openslide import open_slide
import numpy as np

def main(slide_path):
    slide_img = open_slide(slide_path)
    min_size = slide_img.level_dimensions[-1]
    min_img = slide_img.read_region((0, 0), slide_img.level_count - 1, slide_img.level_dimensions[-1])
    min_img = np.array(min_img)
    segment_foreground(min_img)


if __name__ == '__main__':
    main('/home/hades/Desktop/q/data/slide/mrxs/B00.12534_A.mrxs')