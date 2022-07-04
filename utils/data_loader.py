import os
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import numpy as np 
import pandas as pd
import random
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import shutil
from random import shuffle
import cv2
from common_utils import cvtColor, preprocess_input
import math
class DatasetManager:
    def __init__(self, dataset_path, class_csv_file, bs = 10):
        self.dataset_path = dataset_path
        self.class_csv_file = class_csv_file
        self._class_df = None
        self.label_dict = None
        self.batchsize = bs
        self.seed = 1
        self.dataset_dirname  = None
        self.dataset_raw = None
        self.buffer_size = 10000
        self.image_size = None
        self.dataset_img_filenames = None 
        self.transforms_image = None
        self.transforms_image_and_mask = None
        self._init_manager()
        
    def _init_manager(self):
        self._prepare_labeldict()
    """ parse .csv file and load label information. """
    def _prepare_labeldict(self):
        self._class_df = pd.read_csv(os.path.join(self.dataset_path, self.class_csv_file))
        colors = self._class_df[["r", "g", "b"]].values.tolist()
        colors = [tuple(color) for color in colors]
        category = self._class_df[["name"]].values.tolist()
        self.label_dict = {"COLORS": colors, "CATEGORIES": category}
    
    """ get the label information we defined """
    def get_label_info(self):
        return self.label_dict
    """ return data frame of label containing the type of rock and its color. """
    def get_label_encoding(self):
        return self._class_df

    """ return the img data by the path expected. """
    def _get_image(self, image_path: str):
        img = tf.io.read_file(image_path)
        # static_cast for the tf.tensor from int8 (maybe) to float32
        img = tf.cast(tf.image.decode_bmp(img, channels=3), dtype=tf.float32)
        # channels = 3 stands for this method
        # would output RGB image.
        return img
    
    @tf.function    
    def _set_shapes(self, img, mask):  
        img.set_shape((self.image_size[0],self.image_size[1],3))
        mask.set_shape((self.image_size[0],self.image_size[1],19))
        return img,mask

    def _get_filenpaths(self): 
        dataset_img_filenames = tf.data.Dataset.list_files(self.dataset_path + self.dataset_dirname+"/"+ "*.bmp", seed=self.seed)
        image_paths = os.path.join(self.dataset_path,self.dataset_dirname, "*")
        mask_paths = os.path.join(self.dataset_path,self.dataset_dirname+"_labels", "*")
        image_list = sorted(glob(image_paths))
        mask_list = sorted(glob(mask_paths))
        return image_list, mask_list
    
    """ return dataset """
    def get_dataset(self, dataset_dirname, image_size=(128, 128)):
        self.AUTOTUNE   = tf.data.experimental.AUTOTUNE
        self.image_size = image_size
        self.dataset_dirname = dataset_dirname
        self.image_list, self.mask_list = self._get_filenpaths()
        self.dataset = tf.data.Dataset.from_tensor_slices((self.image_list, self.mask_list))
        return self._get_prepared_dataset()

    def _process_data(self, image_path, mask_path):
        image, mask = self._get_image(image_path), self._get_image(mask_path)
        if self.dataset_dirname == "train":
            aug_img = tf.numpy_function(func=self._aug_training, inp=[image, mask], Tout=(tf.float32,tf.float32))
            datapoint = self._normalize_img_and_colorcorrect_mask(aug_img[0],aug_img[1])
            return datapoint[0], datapoint[1]
        else:
            aug_img = tf.numpy_function(func=self._aug_basic, inp=[image, mask], Tout=(tf.float32,tf.float32))
            datapoint = self._normalize_img_and_colorcorrect_mask(aug_img[0],aug_img[1])
            return datapoint[0], datapoint[1]

    def _aug_training(self,image, mask):
        # augment image and mask
        img_mask_data = {"image":image, "mask":mask}
        # aug_image_and_mask = self.transforms_image_and_mask(**img_mask_data)
        aug_image_and_mask = img_mask_data
        aug_img = aug_image_and_mask["image"]
        aug_mask = aug_image_and_mask["mask"]
        # augment image only
        img_data = {"image":aug_img}
        # aug_data =  self.transforms_image(**img_data)
        aug_data = img_data
        aug_img = aug_data["image"]

        aug_img = tf.cast(aug_img, tf.float32)
        aug_img = tf.image.resize(aug_img, size=self.image_size)
        aug_mask = tf.cast(aug_mask, tf.float32)
        aug_mask = tf.image.resize(aug_mask, size=self.image_size)
        aug_img = tf.clip_by_value(aug_img, 0,255)
        return (aug_img, aug_mask)

    def _aug_basic(self,image, mask):
        aug_img = tf.cast(image, tf.float32)
        aug_img = tf.image.resize(aug_img, size=self.image_size)
        aug_mask = tf.cast(mask, tf.float32)
        aug_mask = tf.image.resize(aug_mask, size=self.image_size)
        return (aug_img, aug_mask)

    def _normalize_img_and_colorcorrect_mask(self,input_image, input_mask): 
        input_image = tf.cast(input_image, tf.float32) / 255.0
        one_hot_map = []
        for color in self.label_dict["COLORS"]:
            class_map = tf.reduce_all(tf.equal(input_mask, color), axis=-1)
            one_hot_map.append(class_map)
        one_hot_map = tf.stack(one_hot_map, axis=-1)
        one_hot_map = tf.cast(one_hot_map, tf.float32)
        return (input_image, one_hot_map)

    def _get_prepared_dataset(self):
        if self.dataset_dirname == 'train':
            self.dataset = self.dataset.map(self._process_data, num_parallel_calls=self.AUTOTUNE).prefetch(self.AUTOTUNE)
            self.dataset = self.dataset.map(self._set_shapes, num_parallel_calls=self.AUTOTUNE).shuffle(150).repeat().batch(self.batchsize ).prefetch(self.AUTOTUNE)
            self.train_ds = self.dataset
            return self.train_ds
        elif self.dataset_dirname == "val":
            self.dataset = self.dataset.map(self._process_data, num_parallel_calls=self.AUTOTUNE).prefetch(self.AUTOTUNE)
            self.dataset = self.dataset.map(self._set_shapes, num_parallel_calls=self.AUTOTUNE).repeat().batch(self.batchsize ).prefetch(self.AUTOTUNE)
            self.val_ds = self.dataset
            return self.val_ds
    
    def _restore_original_mask_colors(self, mask):
        new_mask = mask
        h,w = new_mask.shape 
        new_mask = np.reshape(new_mask, (h*w,1))
        dummy_mask = np.ndarray(shape=(h,w, 3))
        dummy_mask =  np.reshape(dummy_mask, (h*w, 3))
        for idx, pixel in enumerate(new_mask):
            dummy_mask[idx] = np.asarray(self.label_dict["COLORS"][int(pixel)])
        return np.reshape(dummy_mask, (h,w,3))/255.

    def show_batch(self, ds, fsize = (15,5)):
        image_batch, label_batch = next(iter(ds)) 
        image_batch = image_batch.numpy()
        label_batch = label_batch.numpy()
        for i in range(len(image_batch)):
            fig, (ax1, ax2 )= plt.subplots(1, 2, figsize=fsize)
            fig.suptitle('Image Label')
            ax1.imshow(image_batch[i])
            ax2.imshow(self._restore_original_mask_colors(np.argmax(label_batch[i], axis=-1)))
    
def split_train_validation_list(root_path: str):
    train_path = root_path + 'train/'
    train_label_path = root_path + 'train_labels/'
    val_path = root_path + 'val/'
    val_label_path = root_path + 'val_labels/'

    imglist = os.listdir(train_path)
    count = len(imglist)
    np.random.seed(10101)
    np.random.shuffle(imglist)
    train_imglist = imglist[:int(count*0.9)]
    val_imglist = imglist[int(count*0.9):]


    #TODO split img files
    for val in val_imglist:
        shutil.move(train_path + val, val_path+val)
        shutil.move(train_label_path + val, val_label_path + val)



def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def letterbox_image(image, label , size):
    label = Image.fromarray(np.array(label))

    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    label = label.resize((nw,nh), Image.NEAREST)
    new_label = Image.new('L', size, (0))
    new_label.paste(label, ((w-nw)//2, (h-nh)//2))
    return new_image, new_label


class Generator(object):
    def __init__(self,batch_size,train_lines,image_size,num_classes,dataset_path):
        self.batch_size     = batch_size
        self.train_lines    = train_lines
        self.train_batches  = len(train_lines)
        self.image_size     = image_size
        self.num_classes    = num_classes
        self.dataset_path   = dataset_path

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        label = Image.fromarray(np.array(label))

        h, w = input_shape
        # resize image
        rand_jit1 = rand(1-jitter,1+jitter)
        rand_jit2 = rand(1-jitter,1+jitter)
        new_ar = w/h * rand_jit1/rand_jit2

        scale = rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        label = label.convert("L")
        
        # flip image or not
        flip = rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
        return image_data,label
        
    def generate(self, random_data = True):
        i = 0
        length = len(self.train_lines)
        inputs = []
        targets = []
        while True:
            if i == 0:
                shuffle(self.train_lines)
            annotation_line = self.train_lines[i]
            name = annotation_line.split()[0]

            # read img and mask from dir.
            jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "JPEGImages"), name + ".bmp"))
            png = Image.open(os.path.join(os.path.join(self.dataset_path, "SegmentationClass"), name + ".bmp"))

            if random_data:
                jpg, png = self.get_random_data(jpg,png,(int(self.image_size[1]),int(self.image_size[0])))
            else:
                jpg, png = letterbox_image(jpg, png, (int(self.image_size[1]),int(self.image_size[0])))
            
            inputs.append(np.array(jpg)/255)
            
            png = np.array(png)
            png[png >= self.num_classes] = self.num_classes
            # num_classes should be set to num_classes + 1 when there are many white strip bars around labels.
            seg_labels = np.eye(self.num_classes+1)[png.reshape([-1])]
            seg_labels = seg_labels.reshape((int(self.image_size[1]),int(self.image_size[0]),self.num_classes+1))
            
            targets.append(seg_labels)
            i = (i + 1) % length
            if len(targets) == self.batch_size:
                tmp_inp = np.array(inputs)
                tmp_targets = np.array(targets)
                inputs = []
                targets = []
                yield tmp_inp, tmp_targets

class DatasetManagerV2(keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, batch_size, num_classes, train, dataset_path):
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)
        self.input_shape        = input_shape
        self.batch_size         = batch_size
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def __getitem__(self, index):
        images  = []
        targets = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i           = i % self.length
            name        = self.annotation_lines[i].split()[0]
            #-------------------------------#
            #   从文件中读取图像
            #-------------------------------#
            jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".png"))
            png         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))
            #-------------------------------#
            #   数据增强
            #-------------------------------#
            jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)
            jpg         = preprocess_input(np.array(jpg, np.float64))
            png         = np.array(png)
            png[png >= self.num_classes] = self.num_classes
            #-------------------------------------------------------#
            #   转化成one_hot的形式
            #   在这里需要+1是因为voc数据集有些标签具有白边部分
            #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
            #-------------------------------------------------------#
            seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
            seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

            images.append(jpg)
            targets.append(seg_labels)

        images  = np.array(images)
        targets = np.array(targets)
        return images, targets

    def on_epoch_end(self):
        shuffle(self.annotation_lines)
        
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image   = cvtColor(image)
        label   = Image.fromarray(np.array(label))
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            iw, ih  = image.size
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label

