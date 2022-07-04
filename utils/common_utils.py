import os
from shutil import which
from sys import prefix
import cv2
from imgviz import color
from numpy.core.defchararray import count, index
from numpy.lib.type_check import imag
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from imgviz.color import *
import os.path as osp
import csv
from tqdm import trange
from colorama import Fore

def get_new_color_map()->dict:
    color_map = {
        '192 192 255': '64 128 64',
        '255 128 192': '192 0 128',
        '85 107 47': '0 128 192',
        '0 200 255': '0 128 64',
        '128 0 0': '128 0 0',
        '196 202 211': '0 0 0',
        '0 255 255': '64 0 128',
        '255 128 128': '64 0 192',
        '255 0 128': '192 128 64',
        '255 0 0': '192 192 128',
        '0 255 0': '64 64 128',
        '128 0 128': '128 0 192',
        '0 192 0': '192 0 64',
        '0 128 128': '128 128 64',
        '0 116 172': '192 0 192',
        '0 0 160': '128 64 64',
        '255 0 153': '64 192 128',
        '255 255 0': '64 64 0',
        '128 128 255': '128 64 128'}
    return color_map

def split_and_save_rock_data(img_path: str, save_path: str = '', size: tuple=(512, 512), postfix='.bmp'):
    img = cv2.imread(img_path)
    # print("size: ", img.shape)
    height          =   img.shape[0]
    width           =   img.shape[1]
    row_batch       = height // size[0]
    column_batch    = width // size[1]
    i = 0
    print("batchSIze:", row_batch, column_batch)
    for row in tqdm(range(row_batch)):
        for column in range(column_batch):
            row_start       =   row*size[0]
            row_end         =   row*size[0]+ size[0]
            column_start    =   column*size[1]
            column_end      =   column* size[1] + size[1]
            data_block = img[row_start: row_end, column_start: column_end]
            cv2.imwrite(save_path + str(i) + postfix, img[row_start: row_end, column_start: column_end])
            i += 1


def get_unique_rgb_list(img_path: str):
    na = get_img_to_array(img_path)
    colours, counts = np.unique(na.reshape(-1,3), axis=0, return_counts=1)
    return colours, counts

def get_img_to_array(img_path: str):
    img = np.array(Image.open(img_path))
    return img

def get_index(src_data, tar_data):
    # index_group = np.where(src_data == tar_data)
    # print(index_group)
    # index = index_group[0][0]
    idx = -1
    for i in range(src_data.shape[0]):
        if tar_data[0] == src_data[i][0] and tar_data[1] == src_data[i][1] and tar_data[2] == src_data[i][2]:
            idx = i
            break
    return idx
def is_contain_color(color_set, color):
    for i in range(len(color_set)):
        if color[0] == color_set[i][0] and color[1] == color_set[i][1] and color[2] == color_set[i][2]:
            return True
    return False

def get_unique_all_rgb_list(img_path: str, save_color_path: str):
    img_list = os.listdir(img_path)
    print("process the color of ", len(img_list), " images")
    # print(img_list[0])
    init_color, counts = get_unique_rgb_list(img_path + img_list[0])
    img_list = img_list[1:]
    # print(init_color, counts)
    for img in tqdm(img_list):
        new_colours, new_counts = get_unique_rgb_list(img_path + img)
        # update the counts.
        for i in range(len(new_colours)):
            # print('new_color[i]: ', new_colours[i], init_color)
            if is_contain_color(init_color, new_colours[i]):
                index = get_index(init_color, new_colours[i])
                counts[index] += new_counts[i]
            else:
                init_color = np.concatenate((init_color, new_colours[i][np.newaxis, :]))
                counts = np.concatenate((counts, np.array([new_counts[i]])))
                # print(init_color, counts)

    print('==================== print colors ====================')
    print(init_color.shape, counts.shape)
    print('==================== print counts ====================')
    print(init_color, counts)
    if init_color.shape[0] == counts.shape[0]:
        with open(save_color_path, 'w+', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            for i in range(init_color.shape[0]):
                colors_str = str(init_color[i][0]) + ' ' + str(init_color[i][1]) + ' ' + str(init_color[i][2])
                counts_str = str(counts[i])
                csv_writer.writerow([colors_str, counts_str])
    else:
        raise ValueError(
            'failed to save color data.'
        )



def label_colormap(n_label=256, value=None):
    """Label colormap.
    Parameters
    ----------
    n_labels: int
        Number of labels (default: 256).
    value: float or int
        Value scale or value of label color in HSV space.

    Returns
    -------
    cmap: numpy.ndarray, (N, 3), numpy.uint8
        Label id to colormap.
    """
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0
    cmap = np.zeros((n_label, 3), dtype=np.uint8)
    print("generating colormap............")
    for i in tqdm(range(0, n_label)):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    if value is not None:
        hsv = rgb2hsv(cmap.reshape(1, -1, 3))
        if isinstance(value, float):
            hsv[:, 1:, 2] = hsv[:, 1:, 2].astype(float) * value
        else:
            assert isinstance(value, int)
            hsv[:, 1:, 2] = value
        cmap = hsv2rgb(hsv).reshape(-1, 3)
    print("colomap generated!")
    return cmap

def lblsave(filename, lbl):
    if osp.splitext(filename)[1] != '.bmp':
        filename += '.bmp'
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode='P')
        colormap = label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(filename)
    else:
        raise ValueError(
            '[%s] Cannot save the pixel-wise class label as PNG. '
            'Please consider using the .npy format.' % filename
        )


def save_rgb_as_8bit(mask_path: str, save_path: str):
    """ read a image and switch its color(RGB) to the color(8-bit) """
    colormap = label_colormap(256) # read a colormap.
    # print(colormap.shape)
    # img = Image.open(mask_path)
    # img_mat = np.array(img)
    img = cv2.imread(mask_path)
    img_mat = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lbls = np.zeros(shape=[img_mat.shape[0], img_mat.shape[1]], dtype=np.int32)
    len_colormap = len(colormap)
    indexes = np.nonzero(img_mat) # return a array which contains all the index of value which is non-zero.
    # print("indexes'shape: ", indexes[0], indexes[1])
    for i, j in zip(indexes[0], indexes[1]):
        for k in tqdm(range(len_colormap)):
            if all(img_mat[i, j, :3] == colormap[k]):
                # print("rgb: ", img_mat[i, j, :3], "-> ", k)
                lbls[i, j] = k
                break
    lblsave(save_path, lbls)

def save_mask_as_8bit(mask_dir: str, save_dir: str):
    mask_list = os.listdir(mask_dir)
    len_mask_list = len(mask_list)
    # for k in trange(len_colormap,  bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET)):
    for i in trange(len_mask_list,  bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET)):
        save_rgb_as_8bit(mask_path=mask_dir + mask_list[i], save_path= save_dir + mask_list[i])

def get_color_map(color_label_path: str):
    color_dict = {}
    with open(color_label_path, 'r') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            color_dict[row[0]] = row[3]
    return color_dict
def get_new_color(color_dict: dict, old_color):
    for key, value in color_dict.items():
        old_color_key = [ int(color_component) for color_component in key.split(' ') ]
        if old_color_key[0] == old_color[0] and old_color_key[1] == old_color[1] and old_color_key[2] == old_color[2]:
            return [int(color_component) for color_component in value.split(' ')]
        
            

def switchColor(mask_dir: str, color_label_path: str, save_path: str, postfix='.bmp'):
    mask_list = os.listdir(mask_dir)
    color_dict = get_color_map(color_label_path) # get dict contains the old color as key and new color as value.
    print("process the color of ", len(mask_list), " masks")
    for mask_name in tqdm(mask_list):
        mask_path = mask_dir + mask_name
        mask_data = get_img_to_array(mask_path)
        for i in range(mask_data.shape[0]):
            for j in range(mask_data.shape[1]):
                new_color_list = get_new_color(color_dict, mask_data[i, j, :])
                mask_data[i, j, 0] = new_color_list[2]
                mask_data[i, j, 1] = new_color_list[1]
                mask_data[i, j, 2] = new_color_list[0]
        mask_prefix, mask_postfix = mask_name.split('.')[0], '.' +mask_name.split('.')[1]
        mask_postfix = postfix if postfix != mask_postfix else mask_postfix
        # cv2.imwrite(save_path + mask_prefix + mask_postfix, mask_data, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(save_path + mask_prefix + mask_postfix, mask_data)

def switch_rgb_color(mask_dir: str,  save_path: str, postfix='.bmp'):
    mask_list = os.listdir(mask_dir)
    color_dict = get_new_color_map()
    print("process the color of ", len(mask_list), " masks")
    for mask_name in tqdm(mask_list):
        mask_path = mask_dir + mask_name
        mask_data = get_img_to_array(mask_path)
        for i in range(mask_data.shape[0]):
            for j in range(mask_data.shape[1]):
                old_color_list = str(mask_data[i, j, 0]) + " " + \
                                 str(mask_data[i, j, 1]) + " " + \
                                 str(mask_data[i, j, 2])
                new_color_list_str = color_dict[old_color_list]
                new_color_list = new_color_list_str.split(' ')
                mask_data[i, j, 0] = new_color_list[0]
                mask_data[i, j, 1] = new_color_list[1]
                mask_data[i, j, 2] = new_color_list[2]
        mask_prefix, mask_postfix = mask_name.split('.')[0], '.' +mask_name.split('.')[1]
        mask_postfix = postfix if postfix != mask_postfix else mask_postfix
        cv2.imwrite(save_path + mask_prefix + mask_postfix, mask_data)

def writeLabel2Csv(tar_path: str = ""):
    color_label = {
        'Calcite': '192 192 255',
        'Quartz': '255 128 192',
        'Chlorite': '85 107 47',
        'Oligoclase': '0 200 255',
        'Biotite': '128 0 0',
        'other': '196 202 211',
        'Albite': '0 255 255',
        'Apatite': '255 128 128',
        'Plagioclase': '255 0 128',
        'Rutile': '255 0 0',
        'Augite': '0 255 0',
        'Benitoite': '128 0 128',
        'Illite': '0 192 0',
        'Orthoclase': '0 128 128',
        'Ankerite': '0 116 172',
        'Clinochlore': '0 0 160',
        'Zircon': '255 0 153',
        'Pyrite': '255 255 0',
        'Dolomite': '128 128 255'}
    color_map = get_new_color_map()
    with open(tar_path, 'w+', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['name', 'r', 'g', 'b'])
        for key, value in color_map.items():
            color_list = value.split(' ')
            r = color_list[0]
            g = color_list[1]
            b = color_list[2]
            csv_writer.writerow([key, r, g, b])


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 


def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    
def preprocess_input(image):
    image = image / 127.5 - 1
    return image