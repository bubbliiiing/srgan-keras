import keras.backend as K
import numpy as np
from tqdm import tqdm

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
