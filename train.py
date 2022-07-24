import datetime
import os

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import layers
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.utils.multi_gpu_utils import multi_gpu_model

from nets.srgan import build_discriminator, build_generator, build_vgg
from utils.callbacks import LossHistory
from utils.dataloader import OrderedEnqueuer, SRganDataset
from utils.utils import get_lr_scheduler, show_config
from utils.utils_fit import fit_one_epoch
from utils.utils_metrics import PSNR, SSIM

if __name__ == "__main__":
    #---------------------------------------------------------------------#
    #   train_gpu   训练用到的GPU
    #               默认为第一张卡、双卡为[0, 1]、三卡为[0, 1, 2]
    #               在使用多GPU时，每个卡上的batch为总batch除以卡的数量。
    #---------------------------------------------------------------------#
    train_gpu       = [0,]
    #--------------------------------------------------------------------------#
    #   如果想要断点续练就将model_path设置成logs文件夹下已经训练的权值文件。 
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从0开始训练，则设置model_path = ''。
    #--------------------------------------------------------------------------#
    G_model_path    = ""
    D_model_path    = ""
    #-----------------------------------------------------#
    #   scale_factor    上采样的倍数，需要是2的n次方
    #                   即2、4、8、16，越大需要的显存越大
    #                   4代表进行四倍的上采样
    #-----------------------------------------------------#
    scale_factor    = 4
    #-----------------------------------------------------#
    #   lr_shape        训练时低分辨率图片输入大小
    #                   训练时高分辨率图片输出大小
    #                   获得输入与输出的图片的shape
    #-----------------------------------------------------#
    lr_shape        = [96, 96]
    hr_shape        = [lr_shape[0] * scale_factor, lr_shape[1] * scale_factor]

    #-----------------------------------------------------#
    #   训练参数设置
    #   Init_epoch      显示的起始世代，默认为0
    #                   断点续练时可调整，会自动调整学习率
    #   Epoch           总共训练的Epoch
    #   batch_size      每次输入多少张图片训练
    #-----------------------------------------------------#
    Init_Epoch      = 0
    Epoch           = 200
    batch_size      = 4
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    Init_lr         = 2e-4
    Min_lr          = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=2e-4
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.9
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   内存较小的电脑可以设置为2或者0  
    #------------------------------------------------------------------#
    num_workers         = 4
    #------------------------------#
    #   每隔50个step保存一次图片
    #------------------------------#
    photo_save_step     = 50
    
    #------------------------------------------------------------------#
    #   annotation_path     获得图片路径
    #------------------------------------------------------------------#
    annotation_path     = "train_lines.txt"

    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)
    print('Number of devices: {}'.format(ngpus_per_node))

    #---------------------------#
    #   生成网络和评价网络
    #---------------------------#
    G_model_body = build_generator([lr_shape[0], lr_shape[1], 3], scale_factor)
    D_model_body = build_discriminator([hr_shape[0], hr_shape[1], 3])
    #-----------------------------------#
    #   创建VGG模型，该模型用于提取特征
    #-----------------------------------#
    VGG_model_body = build_vgg()
    VGG_model_body.trainable = False
        
    #------------------------------------------#
    #   将训练好的模型重新载入
    #------------------------------------------#
    if G_model_path != '':
        G_model_body.load_weights(G_model_path, by_name=True, skip_mismatch=True)
    if D_model_path != '':
        D_model_body.load_weights(D_model_path, by_name=True, skip_mismatch=True)

    if ngpus_per_node > 1:
        G_model     = multi_gpu_model(G_model_body, gpus=ngpus_per_node)
        D_model     = multi_gpu_model(D_model_body, gpus=ngpus_per_node)
        VGG_model   = multi_gpu_model(VGG_model_body, gpus=ngpus_per_node)
    else:
        G_model     = G_model_body
        D_model     = D_model_body
        VGG_model   = VGG_model_body

    #--------------------------------------------#
    #   回调函数
    #--------------------------------------------#
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    callback        = TensorBoard(log_dir=log_dir)
    callback.set_model(G_model)
    loss_history    = LossHistory(log_dir)
    
    with open(annotation_path) as f:
        lines = f.readlines()
    num_train = len(lines)
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    #------------------------------------------------------#
    #   Init_Epoch为起始世代
    #   Epoch总训练世代
    #------------------------------------------------------#
    if True:
        #---------------------------------------#
        #   根据optimizer_type选择优化器
        #---------------------------------------#
        optimizer = {
            'adam'  : Adam(lr = Init_lr, beta_1 = momentum),
            'sgd'   : SGD(lr = Init_lr, momentum = momentum, nesterov=True)
        }[optimizer_type]
        
        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, Epoch)
        
        #---------------------------------------#
        #   判断每一个世代的长度
        #---------------------------------------#
        epoch_step      = min(num_train // batch_size, 2000)
        if epoch_step == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        
        #------------------------------#
        #   Adam optimizer
        #------------------------------#
        D_model.compile(loss="binary_crossentropy", optimizer=optimizer)
        
        D_model.trainable   = False
        img_lr              = layers.Input(shape=[lr_shape[0], lr_shape[1], 3])
        fake_hr             = G_model(img_lr)
        valid               = D_model(fake_hr)
        fake_features       = VGG_model(fake_hr)
        Combine_model_body  = Model(img_lr, [fake_hr, valid, fake_features])
        if ngpus_per_node > 1:
            Combine_model = multi_gpu_model(Combine_model_body, gpus=ngpus_per_node)
        else:
            Combine_model = Combine_model_body

        #-----------------------------------------------------------------#
        #   不同版本的keras与多gpu设置metrics方式不同，因此设置了多个try
        #-----------------------------------------------------------------#
        try:
            Combine_model.compile(loss=['mse', 'binary_crossentropy', 'mse'], loss_weights=[1, 1e-3, 2e-6], optimizer=optimizer,
                                    metrics={'generator': [PSNR, SSIM]})
        except:
            Combine_model.compile(loss=['mse', 'binary_crossentropy', 'mse'], loss_weights=[1, 1e-3, 2e-6], optimizer=optimizer,
                                    metrics=[[PSNR, SSIM], [], []])
            

        train_dataloader    = SRganDataset(lines, lr_shape, hr_shape, batch_size)
        
        #---------------------------------------#
        #   构建多线程数据加载器
        #---------------------------------------#
        gen_enqueuer        = OrderedEnqueuer(train_dataloader, use_multiprocessing=True if num_workers > 1 else False, shuffle=True)
        gen_enqueuer.start(workers=num_workers, max_queue_size=10)
        gen                 = gen_enqueuer.get()
        
        for epoch in range(Init_Epoch, Epoch):
            K.set_value(Combine_model.optimizer.lr, lr_scheduler_func(epoch))
            K.set_value(D_model.optimizer.lr, lr_scheduler_func(epoch))
    
            fit_one_epoch(G_model, D_model, Combine_model, VGG_model, G_model_body, D_model_body, loss_history, epoch, epoch_step, gen, Epoch, save_period, save_dir, photo_save_step)
