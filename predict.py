#-------------------------------------#
#   对单张图片进行预测
#   运行结果保存在根目录
#   保存文件为predict_srgan.png
#-------------------------------------#
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
        r_image = srgan.generate_1x1_image(image)
        r_image.save("predict_srgan.png")
        r_image.show()
