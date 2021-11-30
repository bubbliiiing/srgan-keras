import keras.backend as K
from keras import layers
from keras.models import Model
from keras.optimizers import Adam

from nets.srgan import build_discriminator, build_generator, build_vgg
from utils.dataloader import SRganDataset
from utils.utils_fit import fit_one_epoch
from utils.utils_metrics import PSNR, SSIM

if __name__ == "__main__":
    #-----------------------------------#
    #   代表进行四倍的上采样
    #-----------------------------------#
    scale_factor = 4
    #-----------------------------------#
    #   获得输入与输出的图片的shape
    #-----------------------------------#
    lr_shape    = [96, 96]
    hr_shape    = [lr_shape[0] * scale_factor, lr_shape[1] * scale_factor]
    #--------------------------------------------------------------------------#
    #   如果想要断点续练就将model_path设置成logs文件夹下已经训练的权值文件。 
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从0开始训练，则设置model_path = ''。
    #--------------------------------------------------------------------------#
    G_model_path    = ""
    D_model_path    = ""

    #------------------------------#
    #   训练参数设置
    #------------------------------#
    Init_epoch      = 0
    Epoch           = 200
    batch_size      = 4
    lr              = 0.0002
    #------------------------------#
    #   每隔50个step保存一次图片
    #------------------------------#
    save_interval   = 50
    #------------------------------#
    #   获得图片路径
    #------------------------------#
    annotation_path = "train_lines.txt"

    #---------------------------#
    #   生成网络和评价网络
    #---------------------------#
    G_model = build_generator([lr_shape[0], lr_shape[1], 3], scale_factor)
    D_model = build_discriminator([hr_shape[0], hr_shape[1], 3])
    #-----------------------------------#
    #   创建VGG模型，该模型用于提取特征
    #-----------------------------------#
    VGG_model = build_vgg()
    VGG_model.trainable = False

    #------------------------------------------#
    #   将训练好的模型重新载入
    #------------------------------------------#
    if G_model_path != '':
        G_model.load_weights(G_model_path, by_name=True, skip_mismatch=True)
    if D_model_path != '':
        D_model.load_weights(D_model_path, by_name=True, skip_mismatch=True)
    
    with open(annotation_path) as f:
        lines = f.readlines()
    num_train = len(lines)

    #------------------------------------------------------#
    #   Init_Epoch为起始世代
    #   Epoch总训练世代
    #------------------------------------------------------#
    if True:
        epoch_step = min(num_train // batch_size, 2000)
        if epoch_step == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        #------------------------------#
        #   Adam optimizer
        #------------------------------#
        D_model.compile(loss="binary_crossentropy", optimizer=Adam(lr, 0.9, 0.999))

        img_lr              = layers.Input(shape=[lr_shape[0], lr_shape[1], 3])
        fake_hr             = G_model(img_lr)
        fake_features       = VGG_model(fake_hr)
        D_model.trainable   = False
        valid               = D_model(fake_hr)
        Combine_model       = Model(img_lr, [fake_hr, valid, fake_features])

        Combine_model.compile(loss=['mse', 'binary_crossentropy', 'mse'], loss_weights=[1, 1e-3, 2e-6], optimizer=Adam(lr, 0.9, 0.999),
                                metrics={'model_1': [PSNR, SSIM]})

        gen                 = SRganDataset(lines, lr_shape, hr_shape, batch_size)

        for epoch in range(Init_epoch, Epoch):
            fit_one_epoch(G_model, D_model, Combine_model, VGG_model, epoch, epoch_step, gen, Epoch, batch_size, save_interval)

            lr = K.get_value(Combine_model.optimizer.lr) * 0.98
            K.set_value(Combine_model.optimizer.lr, lr)

            lr = K.get_value(D_model.optimizer.lr) * 0.98
            K.set_value(D_model.optimizer.lr, lr)
