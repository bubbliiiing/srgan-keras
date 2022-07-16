import numpy as np
from PIL import Image

from nets.srgan import build_generator
from utils.utils import cvtColor, postprocess_output, preprocess_input


class SRGAN(object):
    #-----------------------------------------#
    #   注意修改model_path
    #-----------------------------------------#
    _defaults = {
        #-----------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #-----------------------------------------------#
        "model_path"        : 'model_data/Generator_SRGAN.h5',
        #-----------------------------------------------#
        #   上采样的倍数，和训练时一样
        #-----------------------------------------------#
        "scale_factor"      : 4, 
    }
    #---------------------------------------------------#
    #   初始化SRGAN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)  
        self.generate()

    #---------------------------------------------------#
    #   创建生成模型
    #---------------------------------------------------#
    def generate(self):
        self.net = build_generator([None, None, 3], self.scale_factor)
        self.net.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))

    def detect_image(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   添加上batch_size维度，并进行归一化
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image, dtype='float32')), 0)
        
        #---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        #---------------------------------------------------------#
        hr_image    = self.net.predict(image_data)[0]
        
        #---------------------------------------------------------#
        #   将归一化的结果再转成rgb格式
        #---------------------------------------------------------#
        hr_image    = postprocess_output(hr_image)

        hr_image    = Image.fromarray(np.uint8(hr_image))
        return hr_image
