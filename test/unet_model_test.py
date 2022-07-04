import os
import sys
from charset_normalizer import detect
from matplotlib import image
sys.path.append('..')
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.optimizers import Adam

from nets.unet import Unet
from nets.unet_training import CE, Generator, LossHistory, dice_loss_with_CE
from utils.metrics import Iou_score, f_score
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# %matplotlib inline
from unet import Unet
import scipy.signal
import argparse
import cv2
import imgviz.label as il
log_dir = "logs/"
inputs_size = [128,128,3]
num_classes = 2
dice_loss = True


parser = argparse.ArgumentParser(description='input a index')
parser.add_argument('-index', action='store', dest='id')
parser.add_argument('-class', action='store', dest='class_name')
parse_result = parser.parse_args()
image_id = parse_result.id
class_name = parse_result.class_name
if image_id == None:
    image_id = -1
if class_name == None:
    class_name = 'pore'

#functions
#=========================================================================================================

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    fig.tight_layout()
    return fig, ax

def save_lbl(mask_path: str, lbl):
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = Image.fromarray(lbl.astype(np.uint8), mode='P')
        colormap = il.label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(mask_path)
    else:
        raise ValueError(
            '[%s] Cannot save the pixel-wise class label as PNG. '
            'Please consider using the .npy format.' % mask_path
        )
    pass
#=========================================================================================================

# model_path = "../model_data/unet_voc.h5"
albite_model_path = "../log_albite_and_quartz/log_albite/ep070-loss0.289-val_loss0.360.h5"
quartz_model_path = '../log_albite_and_quartz/log_quartz/ep080-loss0.257-val_loss0.238.h5'
qa_model_path  = '../log_albite_quartz_total/ep069-loss0.266-val_loss0.355.h5'
pore_model_path = '../log_pore/ep060-loss0.100-val_loss0.108.h5'
pore_model_path = '../log_pore/pore_new.h5'
ckc_model_path = '../logs_ckc/ep069-loss0.205-val_loss0.331.h5'
# model backup for ckc
# ep069-loss0.205-val_loss0.331.h5
# ep067-loss0.219-val_loss0.168.h5

# model backup for pore
#ep020-loss0.089-val_loss0.125.h5
#ep060-loss0.100-val_loss0.108.h5
# model backup for qa
# ep059-loss0.228-val_loss0.480.h5

# dataset path.
albite_dataset_path = "../VOCdevkit_albite/VOC2007/"
quartz_dataset_path = "../VOCdevkit_quartz/VOC2007/"
qa_dataset_path = '../VOCdevkit_albite_qnd_quartz/VOC2007/'
pore_dataset_path = '../VOCdevkit_pore/VOC2007/'
ckc_dataset_path = '../VOCdevkit_ckc/VOC2007/'

# label path
quartz_label_path = quartz_dataset_path + "SegmentationClass/"
albite_label_path = albite_dataset_path + "SegmentationClass/"
qa_label_path = qa_dataset_path + 'SegmentationClass/'
pore_label_path = pore_dataset_path + 'SegmentationClass/'
ckc_label_path = ckc_dataset_path + 'SegmentationClass/'

# the path for saving plot.
qa_plot_save_path = '../results/detection_results_pa_0323/'
ckc_plot_save_path = '../results/detection_results_ckc/'
pore_plot_save_path = '../results/detection_results_pore/'
qa_loss_txt_path = '../log_albite_quartz_total/loss_2022_03_23_17_04_31/epoch_loss_2022_03_23_17_04_31.txt'
pore_loss_txt_path = '../log_pore/loss_2022_03_24_11_44_51/epoch_loss_2022_03_24_11_44_51.txt'
ckc_loss_txt_path = '../logs_ckc_0324/loss_2022_03_24_20_51_49/epoch_loss_2022_03_24_20_51_49.txt'
MODEL_PATH = None
LABEL_PATH = None
RESULT_SAVE_PATH =None
MSK_COLOR = None
LOSS_PATH = None
# set model_path
if class_name == 'pore':
    MODEL_PATH = pore_model_path
    LABEL_PATH = pore_label_path
    RESULT_SAVE_PATH = pore_plot_save_path
    MSK_COLOR = (159,94,255)
    LOSS_PATH = pore_loss_txt_path

elif class_name == 'qa':
    MODEL_PATH = qa_model_path
    LABEL_PATH = qa_label_path
    RESULT_SAVE_PATH = qa_plot_save_path
    MSK_COLOR = (21,251,213)
    LOSS_PATH = qa_loss_txt_path

elif class_name == 'ckc':
    MODEL_PATH = ckc_model_path
    LABEL_PATH = ckc_label_path
    RESULT_SAVE_PATH = ckc_plot_save_path
    MSK_COLOR = (255,192,192)
    LOSS_PATH = ckc_loss_txt_path
else:
    print('invalid class which could not be detected by model.')
    exit(0)


# MODEL_PATH = qa_model_path
# LABEL_PATH = qa_label_path
DETECT_AND_DRAW = True
PLOT_LOSS = False
OUTPUT_PRE = False
# load image
dataset_path = "../VOCdevkit_albite/VOC2007/"
img_path = dataset_path + "JPEGImages/"
mask_path = dataset_path + ''
img_list = os.listdir(img_path)
img_list.remove('.DS_Store')

image_id = int(image_id)
if image_id >= len(img_list):
    print("invalid index: out of range.")
    exit(0)

# for index, i in enumerate(img_list):
#     print(index, "->", i)

# detect.
if DETECT_AND_DRAW:
    fig, ax = get_ax(1, 3)
    model_obj = Unet()
    model_obj.model_path = MODEL_PATH
    model_obj.load_model()
    model_obj.set_mask_color(1, MSK_COLOR)
    img = Image.open(img_path + img_list[image_id])
    img_data = np.array(img)
    
    r_image = model_obj.detect_image(img)
    ax[0].imshow(img)
    ax[1].imshow(r_image)
    label = Image.open(LABEL_PATH + img_list[image_id])
    ax[2].imshow(label)

    fig.savefig(RESULT_SAVE_PATH + img_list[image_id])
    print("finished to save figure ->", RESULT_SAVE_PATH + img_list[image_id])
    # cv2.imwrite(RESULT_SAVE_PATH + 'pre_' + img_list[image_id], np.array(r_image))

if OUTPUT_PRE:
    model_obj = Unet()
    model_obj.blend = False
    model_obj.model_path = MODEL_PATH
    model_obj.load_model()
    model_obj.set_mask_color(1, MSK_COLOR)
    img = Image.open(img_path + img_list[image_id])
    img_data = np.array(img)
    r_image = model_obj.detect_image(img)
    output = np.array(r_image)
    print(output.shape)
    new_lbl = np.zeros(shape=[output.shape[0], output.shape[1]], dtype=np.int32)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if output[i][j][0] != 0 or output[i][j][1] != 0 or output[i][j][2] !=0:
                new_lbl[i][j] = 1
            else:
                new_lbl[i][j] = 0
    new_mask_save_path = '../miou_pr_dir/' +img_list[image_id]
    save_lbl(new_mask_save_path, new_lbl)


if PLOT_LOSS:
    f = open(LOSS_PATH)
    raw_loss = f.readlines()
    loss_list = [float(i) for i in raw_loss]
    iters = range(len(loss_list))
    plt.figure()
    plt.plot(iters, loss_list, 'red', linewidth = 2, label='train loss')
    if len(loss_list) < 25:
            num = 5
    else:
            num = 15
    plt.plot(iters, scipy.signal.savgol_filter(loss_list, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('A Loss Curve')
    plt.legend(loc="upper right")
    plt.savefig(RESULT_SAVE_PATH + 'loss.png')
    plt.cla()
    plt.close("all")
    # for loss_value in loss_list:
    #     print(loss_value)
        # loss_list.append(loss_value.split('\')[])
    f.close()