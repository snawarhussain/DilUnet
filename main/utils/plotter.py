import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
def imshow(train_loader =None, no_train_loader = False, img_in = None, label_in = None):
    axes = []

    fig  = plt.figure()
    if train_loader != None:
        img, label = next(iter(train_loader))


    if no_train_loader:
        img = img_in
        label = label_in


    img =  img.numpy()
    label = label.numpy()
    imgtem = img # numpy.concatenate((img,label), axis=0)
    rows= img.shape[0]
    for i in range(rows*2):

        axes.append(fig.add_subplot(rows, rows, i+1))
        subplot_title = ("img")
        subplot_lbl_title = ("label")


        img_num = imgtem[i].transpose((2,1,0))

        plt.imshow(img_num)

    for i in range(rows*2):
        num = (i + 1)

        if i< rows:

            axes[i].set_title(subplot_title+str(num))
            axes[i].set_xticklabels('')
            axes[i].set_yticklabels('')
        else:
            axes[i].set_title(subplot_lbl_title)
            axes[i].set_xticklabels('')
            axes[i].set_yticklabels('')


    plt.show()

def plot_img_and_mask(img, mask):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()
image_path = 'DRIVE/training/images/img/22_training.TIF'
mask_path = 'DRIVE/training/mask/22_training_mask.gif'
def preprocess(image_path, mask_path):
    image = cv2.imread(image_path)
    cv2.imshow('original', image)
    mask = Image.open(mask_path)
    mask =np.asarray(mask).astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_out = cv2.subtract(mask,image)
    mask_out = cv2.subtract(mask,mask_out)
    _,g,r = cv2.split(mask_out)
    hybrid_channel = cv2.addWeighted(g,0.8, r, 0.2, 0)
    hybrid_2 = 0.8*g + 0.2*r
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(hybrid_channel)
    cv2.imshow('hybrid',hybrid_channel)
    cv2.imshow('clahe',cl1)
    cv2.imshow('mask_out', mask_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    preprocess(image_path, mask_path)