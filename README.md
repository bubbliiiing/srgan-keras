## Srgan：超分辨率图像复原模型在Keras当中的实现
---

### 目录
1. [所需环境 Environment](#所需环境)
3. [文件下载 Download](#文件下载)
4. [预测步骤 How2predict](#预测步骤)
5. [训练步骤 How2train](#训练步骤)
6. [参考资料 Reference](#Reference)

## 所需环境
tensorflow-gpu==1.13.1    
keras==2.1.5    

## 文件下载
为了验证模型的有效性，我使用了**Yahoo MirFlickr25k数据集**进行了训练。    
训练好的生成器与判别器模型[Generator_SRGAN.h5](https://github.com/bubbliiiing/srgan-keras/releases/download/v1.0/Generator_SRGAN.h5)、[Discriminator_SRGAN.h5](https://github.com/bubbliiiing/srgan-keras/releases/download/v1.0/Discriminator_SRGAN.h5)可以通过百度网盘下载或者通过GITHUB下载    
权值的百度网盘地址如下：    
链接: https://pan.baidu.com/s/1Jf8mmzprXv9GSOiTWgB_CA 提取码: fc53  

Yahoo MirFlickr25k数据集可以通过百度网盘下载：   
链接: https://pan.baidu.com/s/1kGGB-SClc44VkhKLR_GoVw 提取码: 58es  

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，直接运行predict.py，输入要提高分辨率的图片的路径，即可生成高分辨率图片，生成图片位于根目录的predict_srgan.png。如输入：
```python
img/before.jpg
```
### b、使用自己训练的权重 
1. 按照训练步骤训练。    
2. 在dcgan.py文件里面，在如下部分修改model_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件**。    
```python
#-----------------------------------------#
#   注意修改model_path
#-----------------------------------------#
_defaults = {
    "model_path"        : 'model_data/Generator_SRGAN.h5',
    "scale_factor"      : 4, 
}
```
3. 运行predict.py，输入要提高分辨率的图片的路径，即可生成高分辨率图片，生成图片位于根目录的predict_srgan.png。 

## 训练步骤
1. 训练前将期望生成的图片文件放在datasets文件夹下（参考Yahoo MirFlickr25k数据集）。  
2. 运行根目录下面的txt_annotation.py，生成train_lines.txt，保证train_lines.txt内部是有文件路径内容的。  
3. 运行train.py文件进行训练，训练过程中生成的图片可查看results文件夹下的图片。  


