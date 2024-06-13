import torch
from matplotlib import pyplot as plt
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from main.utils.Data_loader import CustomDataLoader


def visualize(image, mask=None, original_image=None, original_mask=None):
    fontsize = 18
    if mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))
        ax[0].imshow(image)
        plt.show()

    elif original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image, cmap='gray')
        ax[1].imshow(mask, cmap='gray')
        plt.show()

    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image, cmap='gray')
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask, cmap='gray')
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image, cmap='gray')
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask, cmap='gray')
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
        plt.show()


if __name__ == '__main__':
    import cv2
    import numpy as np
    import albumentations as A
    #
    # img_dir = 'DRIVE/training/images/img/22_training.tif'
    #
    # img = cv2.imread(img_dir)
    img_dir = 'DRIVE/training/images/img/'
    label_dir = 'DRIVE/training/images/label/'
    mask_dir = 'DRIVE/training/mask/'
    val_percent = 0.2
    batch_size = 1
    width_out = 420
    height_out = 420
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available..... training on CPU')
    else:
        print("CUDA is available..... training on GPU")
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    transform = A.Compose([
        # A.VerticalFlip(p=0.5),
        # A.RandomRotate90(p=0.5),
        # A.OneOf([A.ElasticTransform(p=0.5, alpha=90, sigma=120 * 0.05, alpha_affine=45 * 0.03),
        # A.GridDistortion(p=0.5)], p=0.6),
        A.CLAHE(p=1),
        # A.RandomBrightnessContrast(p=0.6),
        # A.RandomGamma(p=0.6),
        # A.Normalize(mean=0.485, std=0.229)
    ])

    transform_label = transforms.Compose([transforms.ToTensor()
                                          ])

    dataset = CustomDataLoader(img_dir, label_dir, mask_dir, transform, transform_label,image_scale=0.5)
    n_val = int(len(dataset) * val_percent)
    n_train = int(len(dataset) - n_val)
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=3, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val, batch_size=3, shuffle=False, num_workers=0, pin_memory=False)
    for img, label in (iter(train_loader)):
        img = np.squeeze(img[0].permute(1, 2, 0).numpy())
        label = np.squeeze(label[0].permute(1, 2, 0).numpy())
        img_green = img
        filterSize = (5, 5)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        bottom = cv2.morphologyEx(img_green, cv2.MORPH_BLACKHAT, kernel)
        top = cv2.morphologyEx(img_green, cv2.MORPH_TOPHAT, kernel2)
        im = bottom - top
        visualize(img, im)
#         visualize(img,label)
