import math

import tensorflow as tf
from keras import layers
from keras.applications import VGG19
from keras.initializers import random_normal
from keras.models import Model


def residual_block(inputs, filters):
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer = random_normal(stddev=0.02))(inputs)
    x = layers.BatchNormalization(momentum=0.5)(x)
    x = layers.advanced_activations.PReLU(shared_axes=[1,2])(x)

    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer = random_normal(stddev=0.02))(x)
    x = layers.BatchNormalization(momentum=0.5)(x)
    x = layers.Add()([x, inputs])
    return x

def SubpixelConv2D(scale=4):
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                None if input_shape[1] is None else input_shape[1] * scale,
                None if input_shape[2] is None else input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)

    return layers.Lambda(subpixel, output_shape=subpixel_shape)
    
def deconv2d(inputs):
    x = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer = random_normal(stddev=0.02))(inputs)
    x = SubpixelConv2D(scale=2)(x)
    x = layers.advanced_activations.PReLU(shared_axes=[1,2])(x)
    return x

def build_generator(lr_shape, scale_factor, num_residual=16):
    #-----------------------------------#
    #   获得进行上采用的次数
    #-----------------------------------#
    upsample_block_num = int(math.log(scale_factor, 2))
    img_lr = layers.Input(shape=lr_shape)

    #--------------------------------------------------------#
    #   第一部分，低分辨率图像进入后会经过一个卷积+PRELU函数
    #--------------------------------------------------------#
    x = layers.Conv2D(64, kernel_size=9, strides=1, padding='same', kernel_initializer = random_normal(stddev=0.02))(img_lr)
    x = layers.advanced_activations.PReLU(shared_axes=[1,2])(x)

    short_cut = x
    #-------------------------------------------------------------#
    #   第二部分，经过num_residual个残差网络结构。
    #   每个残差网络内部包含两个卷积+标准化+PRELU，还有一个残差边。
    #-------------------------------------------------------------#
    for _ in range(num_residual):
        x = residual_block(x, 64)

    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer = random_normal(stddev=0.02))(x)
    x = layers.BatchNormalization(momentum=0.5)(x)
    x = layers.Add()([x, short_cut])

    #-------------------------------------------------------------#
    #   第三部分，上采样部分，将长宽进行放大。
    #   两次上采样后，变为原来的4倍，实现提高分辨率。
    #-------------------------------------------------------------#
    for _ in range(upsample_block_num):
        x = deconv2d(x)

    gen_hr = layers.Conv2D(3, kernel_size=9, strides=1, padding='same', activation='tanh')(x)

    return Model(img_lr, gen_hr, name="generator")

def d_block(inputs, filters, strides=1):
    x = layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same', kernel_initializer = random_normal(stddev=0.02))(inputs)
    x = layers.BatchNormalization(momentum=0.5)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x

def build_discriminator(hr_shape):
    inputs = layers.Input(shape=hr_shape)

    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer = random_normal(stddev=0.02))(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = d_block(x, 64, strides=2)
    x = d_block(x, 128)
    x = d_block(x, 128, strides=2)
    x = d_block(x, 256)
    x = d_block(x, 256, strides=2)
    x = d_block(x, 512)
    x = d_block(x, 512, strides=2)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, kernel_initializer = random_normal(stddev=0.02))(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    validity = layers.Dense(1, activation='sigmoid', kernel_initializer = random_normal(stddev=0.02))(x)
    return Model(inputs, validity, name="discriminator")

def build_vgg():
    # 建立VGG模型，只使用第9层的特征
    vgg = VGG19(False, weights="imagenet")
    vgg.outputs = [vgg.layers[-2].output]

    img = layers.Input(shape=[None,None,3])
    img_features = vgg(img)

    return Model(img, img_features, name="vgg")

if __name__ == "__main__":
    model = build_generator([56,56,3])
    model.summary()
