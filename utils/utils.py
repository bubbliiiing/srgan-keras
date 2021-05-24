import itertools

import matplotlib.pyplot as plt

def show_result(num_epoch, G_net, imgs_lr, imgs_hr):
    test_images = G_net.predict(imgs_lr)

    fig, ax = plt.subplots(1, 2)

    for j in itertools.product(range(2)):
        ax[j].get_xaxis().set_visible(False)
        ax[j].get_yaxis().set_visible(False)
    
    ax[0].cla()
    ax[0].set_title("Fake_Hr_Images")
    ax[0].imshow((test_images[0] * 0.5 + 0.5))

    ax[1].cla()
    ax[1].set_title("True_Hr_Images")
    ax[1].imshow((imgs_hr[0] * 0.5 + 0.5))

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    plt.savefig("results/epoch_" + str(num_epoch) + "_results.png")
    plt.close('all')  #避免内存泄漏

