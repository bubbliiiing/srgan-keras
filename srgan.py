import numpy as np
from PIL import Image

from nets.srgan import build_generator


class SRGAN(object):
    #-----------------------------------------#
    #   注意修改model_path
    #-----------------------------------------#
    _defaults = {
        "model_path"        : 'model_data/Generator_SRGAN.h5',
        "scale_factor"      : 4, 
    }
    #---------------------------------------------------#
    #   初始化SRGAN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()

    def generate(self):
        self.net = build_generator([None,None,3], self.scale_factor)
        self.net.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))

    def pre_process(self, image, mean, std):
        image = (image/255 - mean)/std
        return image

    def generate_1x1_image(self, image):
        #-------------------------------------#
        #   对图像进行预处理
        #-------------------------------------#
        image = np.array(image)
        image = self.pre_process(image, [0.5,0.5,0.5], [0.5,0.5,0.5])
        image = np.expand_dims(image, 0)
        
        #-------------------------------------#
        #   将预处理后的图像传入网络进行预测
        #-------------------------------------#
        test_image = self.net.predict(image)
        #-------------------------------------#
        #   将归一化的结果再转成rgb格式
        #-------------------------------------#
        test_image = (test_image[0] * 0.5 + 0.5) * 255

        test_image = Image.fromarray(np.uint8(test_image))
        return test_image
