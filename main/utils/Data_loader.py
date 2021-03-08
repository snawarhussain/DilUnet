import torch
from torch.utils.data import Dataset
import os
import natsort
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms as transforms
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt


class CustomDataLoader(Dataset):
    """the class for loading the dataset from the directories
    Arguments:

        img_dir: directory for the dataset images

        label_dir: labels of the images directory

        transform: the list of transformation applied to the images of dataset

        transform_label: transforms applied to the labels of dataset
    """

    def __init__(self, img_dir, label_dir, mask_dir, transform, transform_label):

        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.transform_label = transform_label
        all_images = os.listdir(self.img_dir)
        all_lables = os.listdir(self.label_dir)
        all_masks = os.listdir(self.mask_dir)
        self.total_imgs = natsort.natsorted(all_images)
        self.total_labels = natsort.natsorted(all_lables)
        self.total_masks = natsort.natsorted(all_masks)
        pass

    def __len__(self):
        return len(self.total_imgs)

    @classmethod
    def preprocessing(cls, image, label, mask, size=None, scale=None):
        """class method for preprocessing of the image and label
        usage: preprocessing the images before feeding to the network for training
        as well as before making predictions
        dataset class dishes out pre-processed bathes of images and labels """
        w, h = image[:, :, 0].shape
        if scale is not None and w!= h:
            newW = int(scale * w)
            newH = int(scale * w)
            assert newW > 0 and newH > 0, 'Scale is too small'
            image = cv2.resize(image, (newW, newH), interpolation=cv2.INTER_LINEAR)
            label = label.resize((newW, newH))
            mask = mask.resize((newW, newH))

        label = np.asarray(label).astype(np.uint8)
        # size = (int(512), int(512))
        # label = cv2.resize(label, size, interpolation=cv2.INTER_LINEAR)
        # label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
        ret, label = cv2.threshold(label, 80, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # label = np.expand_dims(np.array(label, np.float32), -1)
        # label = label / 255.
        label = Image.fromarray(label)

        mask = np.array(mask).astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        masked_out = cv2.subtract(mask, image)
        masked_out = cv2.subtract(mask, masked_out)
        # cv2.imshow('orignial', image)
        # cv2.imshow('mask_out', masked_out)

        hybrid_channel = cv2.addWeighted(masked_out[:, :, 2], 0.2, masked_out[:, :, 1], 0.8, 0)
        # cv2.imshow('hybird', hybrid_channel)
        # cv2.imshow('green', masked_out[:, :, 1])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl_img = clahe.apply(hybrid_channel)

        return cl_img, label

    def __getitem__(self, idx):
        """ Generator to yield a tuple of image and label
        idx: the index to iterate over the dataset in directories of both
        images and labels

        ---------------------

        :return:  image, label
        :rtype: torch tensor
        """
        # img_loc = os.path.join(self.img_dir,self.total_imgs[idx])
        # label_loc = os.path.join(self.label_dir,self.total_labels[idx])
        # image = Image.open(img_loc).convert("L")
        # label = Image.open(label_loc).convert("L")
        # image = self.transform(image)
        # label =  self.transform_label(label)

        img_loc = os.path.join(self.img_dir, self.total_imgs[idx])
        label_loc = os.path.join(self.label_dir, self.total_labels[idx])
        mask_loc = os.path.join(self.mask_dir, self.total_masks[idx])
        # image = Image.open(img_loc)
        image = cv2.imread(img_loc)
        mask = Image.open(mask_loc)

        # image = image.resize((512, 512), Image.ANTIALIAS)
        label = Image.open(label_loc)  # opening image and converting it to the grey scale
        # label = np.asarray(label).astype(np.uint8)
        # size = (int(512), int(512))
        # # label = cv2.resize(label, size, interpolation=cv2.INTER_LINEAR)
        # # label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
        # # label = cv2.GaussianBlur(label, (5, 5), 0)
        # ret, label = cv2.threshold(label, 80, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # # ret, label = cv2.adaptiveThreshold(label, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        # #   cv2.THRESH_BINARY,11,2)
        # # label = np.expand_dims(np.array(label, np.float32), -1)
        # # label = label / 255.
        image, label = self.preprocessing(image, label, mask, scale=1)

        label = self.transform_label(label)
        image = self.transform(image)

        #

        return image, label


if __name__ == '__main__':
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
    transform = transforms.Compose([
        transforms.ToTensor()])

    transform_label = transforms.Compose([

        transforms.ToTensor()])

    dataset = CustomDataLoader(img_dir, label_dir, mask_dir, transform, transform_label)
    n_val = int(len(dataset) * val_percent)
    n_train = int(len(dataset) - n_val)
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=15, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val, batch_size=3, shuffle=False, num_workers=0, pin_memory=True)
    img, label = next(iter(train_loader))
    # img = np.asarray(np.squeeze(label[0]))
    print(np.unique(img))
    img = np.squeeze(img[0].permute(1, 2, 0).numpy())
    plt.imshow(img, cmap='gray')
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
