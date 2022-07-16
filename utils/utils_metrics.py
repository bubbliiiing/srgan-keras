import tensorflow as tf
from keras import backend as K
from .utils import postprocess_output

def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(y_true, y_pred):
    y_true = postprocess_output(y_true)
    y_pred = postprocess_output(y_pred)
    max_pixel = 255.0
    y_pred = K.clip(y_pred, 0.0, 255.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))

def SSIM(y_true, y_pred):
    y_true = postprocess_output(y_true)
    y_pred = postprocess_output(y_pred)
    ssim = tf.image.ssim(y_pred, y_true, 255)
    return ssim
