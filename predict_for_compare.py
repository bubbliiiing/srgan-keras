#----------------------------------------#
#   对单张图片进行对比
#   该预测文件用于获得对比的图片
#   会保存bicbic上采样和srgan上采样的结果
#   如果想直接获得预测结果请用predict.py
#----------------------------------------#
import numpy as np
from PIL import Image

from srgan import SRGAN

srgan = SRGAN()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        height, width, _    = np.shape(image)
        image               = image.resize([int(width/2), int(height/2)], Image.BICUBIC)
        resize_image        = image.resize([int(width*2), int(height*2)], Image.BICUBIC)
        resize_image.save("predict_bic_bic.png")

        r_image = srgan.generate_1x1_image(image)
        r_image.save("predict_srgan.png")
        r_image.show()
