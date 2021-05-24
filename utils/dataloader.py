import math
from random import randint, shuffle

import keras
import numpy as np
from PIL import Image

def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class SRganDataset(keras.utils.Sequence):
    def __init__(self, train_lines, lr_shape, hr_shape, batch_size):
        super(SRganDataset, self).__init__()

        self.train_lines    = train_lines
        self.train_batches  = len(train_lines)
        self.lr_shape       = lr_shape
        self.hr_shape       = hr_shape

        self.batch_size     = batch_size
        self.global_index   = 0

    def __len__(self):
        return math.ceil(self.train_batches / float(self.batch_size))

    def pre_process(self, image, mean, std):
        image = (image/255 - mean)/std
        return image

    def random_crop(self, image, width, height):
        #--------------------------------------------#
        #   如果图像过小无法截取，先对图像进行放大
        #--------------------------------------------#
        if image.size[0]<self.hr_shape[1] or image.size[1]<self.hr_shape[0]:
            resized_width, resized_height = get_new_img_size(width, height, img_min_side=np.max(self.hr_shape))
            image = image.resize((resized_width, resized_height), Image.BICUBIC)

        #--------------------------------------------#
        #   随机截取一部分
        #--------------------------------------------#
        width1  = randint(0, image.size[0] - width)
        height1 = randint(0, image.size[1] - height)

        width2  = width1 + width
        height2 = height1 + height

        image   = image.crop((width1, height1, width2, height2))
        return image

    def __getitem__(self, index):
        if self.global_index == 0:
            shuffle(self.train_lines)

        images_l = []
        images_h = []
        lines   = self.train_lines
        n       = self.train_batches
        for _ in range(self.batch_size):
            image_origin = Image.open(lines[self.global_index].split()[0])

            img_h = self.random_crop(image_origin, self.hr_shape[1], self.hr_shape[0])
            img_l = img_h.resize((self.lr_shape[1], self.lr_shape[0]), Image.BICUBIC)

            img_h = np.array(img_h, dtype=np.float32)
            img_h = self.pre_process(img_h, [0.5,0.5,0.5], [0.5,0.5,0.5])
            
            img_l = np.array(img_l, dtype=np.float32)
            img_l = self.pre_process(img_l, [0.5,0.5,0.5], [0.5,0.5,0.5])

            if rand()<.5:
                img_h = np.fliplr(img_h)
                img_l = np.fliplr(img_l)
                
            images_h.append(img_h)
            images_l.append(img_l)

            self.global_index = (self.global_index + 1) % n
        return np.array(images_l), np.array(images_h)
