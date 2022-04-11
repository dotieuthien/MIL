# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from xml.dom import minidom
import xml.etree.ElementTree as ET
import multiresolutionimageinterface as mri
from openslide import open_slide
import cv2


def read_xml(mr_file, xml_file):
    """ Read file annotations xml for slide

    Args:
        file_path (str): path to xml file
    """
    mr_img = open_slide(mr_file)
    dim = mr_img.level_dimensions
    max_dim = dim[0]
    min_dim = dim[-1]
    num_level = mr_img.level_count

    print('Numbger of level in slide ', mr_img.level_count)
    print("Dimension of level 0 ", max_dim)
    print("Dimension of minimum level ", min_dim)

    min_level_img = np.array(mr_img.read_region((0, 0), num_level - 1, min_dim))
    min_level_anns = []

    tree = ET.parse(xml_file)
    # Get root of file
    root = tree.getroot()

    # Get Annotations
    anns = root[0]
    for ann in anns:
        min_level_ann = []
        # In each Annotation, we have a Coordinates
        # Coordinates means that a closed region in slide image (tumor/normal)
        coords = ann[0]
        class_name = ann.attrib['Name']

        for coord in coords:
            print('Origin annotations ', coord.tag, coord.attrib)
            # Scale annotation to minimum level of slide
            origin_coord = (float(coord.attrib['X']), float(coord.attrib['Y']))
            scale_x, scale_y = scale_ann(max_dim, min_dim, origin_coord)
            min_level_ann.append([scale_x, scale_y])
        
        min_level_anns.append(min_level_ann)
    
    # Draw anns
    h, w, _ = min_level_img.shape
    ann_img = np.ones((h, w)) * 255
    for ann in min_level_anns:
        points = np.array(ann, dtype=np.int32)
        print(points.shape)
        color = 0
        ann_img = cv2.polylines(ann_img, [points], True, color, 2)

    plt.imshow(ann_img)
    plt.show()

    # Convert ann img to mask
    mask = create_mask(ann_img)
    plt.imshow(mask)
    plt.show()

    return


def convert_annotations_to_mask(mr_file, xml_file, output_file):
    """Convert anootation to mask by ASAP

    Args:
        mr_file (str): slide image
        xml_file (str): _description_
        output_file (str): file.tif
    """
    reader = mri.MultiResolutionImageReader()

    mr_image = reader.open(mr_file)
    print(mr_image.getDimensions(), mr_image.getSpacing())

    annotation_list = mri.AnnotationList()

    xml_repository = mri.XmlRepository(annotation_list)
    xml_repository.setSource(xml_file)
    xml_repository.load()

    annotation_mask = mri.AnnotationToMask()
    print('Converting ...')
    annotation_mask.convert(annotation_list, output_file, mr_image.getDimensions(), mr_image.getSpacing())
    
    return


def scale_ann(max_dimension, min_dimension, origin_coord):
    """Scale down origin anns to anns with min level size

    Args:
        max_dimension (_type_): _description_
        min_dimension (_type_): _description_
        origin_coord (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Scale factor
    scale_w, scale_h = max_dimension[0] / min_dimension[0], max_dimension[1] / min_dimension[1]
    scale_x, scale_y = origin_coord[0] / scale_w, origin_coord[1] / scale_h
    return scale_x, scale_y


def create_mask(ann_img):
    """From anns to mask

    Args:
        ann_img (np.array):
    """
    h, w = ann_img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = ann_img.copy()
    im_flood_fill = im_flood_fill.astype("uint8")

    cv2.floodFill(im_flood_fill, mask, (0, 0), 128)

    rows, cols = np.where(im_flood_fill == 128)
    img_out = np.ones((h, w), dtype=np.uint8)
    img_out[rows, cols] = 0
    
    return img_out


def extract_tumor_tile():
    pass


def extract_normal_tile():
    pass


if __name__ == "__main__":
    mr_file = '/home/thiendo/Desktop/MIL/data/tumor_slides/B2016.9657 A.mrxs'
    file_xml = '/home/thiendo/Desktop/MIL/data/tumor_slides/B2016.9657 A.xml'
    mask_file = '/home/thiendo/Desktop/MIL/data/tumor_slides/B2016.9657 A.mask.tif'
    read_xml(mr_file, file_xml)
    # convert_annotations_to_mask(mr_file, file_xml, mask_file)

