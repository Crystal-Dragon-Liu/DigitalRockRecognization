import datetime
import os

import keras.backend as K
import numpy as np
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.layers import Conv2D, Dense, DepthwiseConv2D
from keras.optimizers import SGD, Adam
from keras.regularizers import l2

from networks.deeplabv3 import Deeplabv3
from nets.deeplab_training import (CE, Focal_Loss, dice_loss_with_CE,
                                  dice_loss_with_Focal_Loss, get_lr_scheduler)
from utils.callbacks import ExponentDecayScheduler, LossHistory
from utils.dataloader import DeeplabDataset
from utils.utils_metrics import Iou_score, f_score