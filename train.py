import keras.backend as K
import numpy as np
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from tqdm import tqdm

from nets.srgan import build_discriminator, build_generator, build_vgg
from utils.dataloader import SRganDataset
from utils.metrics import PSNR, SSIM
from utils.utils import show_result


def fit_one_epoch(G_model, D_model, Combine_model, VGG_model, epoch, epoch_size, gen, Epoch, batch_size, save_interval):
    G_total_loss = 0
    G_total_PSNR = 0
    G_total_SSIM = 0
    D_total_loss = 0

    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            imgs_lr, imgs_hr        = batch

            y_real                  = np.ones((batch_size,))
            y_fake                  = np.zeros((batch_size,))

            #-------------------------------------------------#
            #   训练判别器
            #-------------------------------------------------#
            G_result                = G_model.predict(imgs_lr)
            d_loss_real             = D_model.train_on_batch(imgs_hr, y_real)
            d_loss_fake             = D_model.train_on_batch(G_result, y_fake)
            d_loss                  = 0.5 * np.add(d_loss_real, d_loss_fake)

            #-------------------------------------------------#
            #   训练生成器
            #-------------------------------------------------#
            image_features          = VGG_model.predict(imgs_hr)
            g_loss                  = Combine_model.train_on_batch(imgs_lr, [imgs_hr, y_real, image_features])
            D_total_loss            += d_loss
            G_total_loss            += g_loss[0]
            G_total_PSNR            += g_loss[4]
            G_total_SSIM            += g_loss[5]

            pbar.set_postfix(**{'G_loss'        : G_total_loss / (iteration + 1), 
                                'D_loss'        : D_total_loss / (iteration + 1),
                                'G_PSNR'        : G_total_PSNR / (iteration + 1), 
                                'G_SSIM'        : G_total_SSIM / (iteration + 1),
                                'lr'            : K.get_value(D_model.optimizer.lr)},)
            pbar.update(1)

            if iteration % save_interval == 0:
                show_result(epoch+1, G_model, imgs_lr, imgs_hr)

    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('G Loss: %.4f || D Loss: %.4f ' % (G_total_loss/(epoch_size+1),D_total_loss/(epoch_size+1)))
    
    if (epoch+1) % 5 == 0:
        print('Saving state, iter:', str(epoch+1))
        G_model.save_weights('logs/G_Epoch%d-GLoss%.4f-DLoss%.4f.h5'%((epoch+1), G_total_loss/(epoch_size+1), D_total_loss/(epoch_size+1)))
        D_model.save_weights('logs/D_Epoch%d-GLoss%.4f-DLoss%.4f.h5'%((epoch+1), G_total_loss/(epoch_size+1), D_total_loss/(epoch_size+1)))


if __name__ == "__main__":
    #-----------------------------------#
    #   获得输入图片的高、宽、通道数
    #-----------------------------------#
    lr_height   = 96
    lr_width    = 96
    channels    = 3

    #-----------------------------------#
    #   代表进行四倍的上采样
    #-----------------------------------#
    scale_factor = 4
    #-----------------------------------#
    #   获得输入与输出的图片的shape
    #-----------------------------------#
    lr_shape    = (lr_height, lr_width, channels)
    hr_shape    = (lr_height * scale_factor, lr_width * scale_factor, channels)

    #---------------------------#
    #   生成网络和评价网络
    #---------------------------#
    G_model = build_generator(lr_shape, scale_factor)
    D_model = build_discriminator(hr_shape)
    #-----------------------------------#
    #   创建VGG模型，该模型用于提取特征
    #-----------------------------------#
    VGG_model = build_vgg()
    VGG_model.trainable = False

    # G_model_path = "model_data/Generator_SRGAN.h5"
    # D_model_path = "model_data/Discriminator_SRGAN.h5"
    # G_model.load_weights(G_model_path, by_name=True, skip_mismatch=True)
    # D_model.load_weights(D_model_path, by_name=True, skip_mismatch=True)
    
    #---------------------------#
    #   数据集存放路径
    #   指向训练用的txt
    #---------------------------#
    annotation_path = "train_lines.txt"
    with open(annotation_path) as f:
        lines = f.readlines()
    num_train = len(lines)

    #------------------------------------------------------#
    #   Init_Epoch为起始世代
    #   Epoch总训练世代
    #------------------------------------------------------#
    if True:
        lr          = 0.0002
        batch_size  = 4
        Init_epoch  = 0
        Epoch       = 200
        #------------------------------------------------------#
        #   每个50个step保存一次图片，保存在results里
        #------------------------------------------------------#
        save_interval = 50

        D_model.compile(loss="binary_crossentropy", optimizer=Adam(lr, 0.9, 0.999))

        img_lr              = layers.Input(shape=lr_shape)
        fake_hr             = G_model(img_lr)
        fake_features       = VGG_model(fake_hr)
        D_model.trainable   = False
        valid               = D_model(fake_hr)
        Combine_model       = Model(img_lr, [fake_hr, valid, fake_features])

        Combine_model.compile(loss=['mse', 'binary_crossentropy', 'mse'], loss_weights=[1, 1e-3, 2e-6], optimizer=Adam(lr, 0.9, 0.999),
                                metrics={'model_1': [PSNR, SSIM]})

        gen         = SRganDataset(lines, lr_shape, hr_shape, batch_size)

        epoch_size  = min(max(1, num_train//batch_size), 2000)

        for epoch in range(Init_epoch, Epoch):
            fit_one_epoch(G_model, D_model, Combine_model, VGG_model, epoch, epoch_size, gen, Epoch, batch_size, save_interval)

            lr = K.get_value(Combine_model.optimizer.lr) * 0.98
            K.set_value(Combine_model.optimizer.lr, lr)

            lr = K.get_value(D_model.optimizer.lr) * 0.98
            K.set_value(D_model.optimizer.lr, lr)
