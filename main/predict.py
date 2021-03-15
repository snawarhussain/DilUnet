import torch
import cv2
from main.utils.tem import *
from torchvision import transforms
from utils.Data_loader import CustomDataLoader
from matplotlib import pyplot as plt
from Network import Unet_variant
import numpy as np
from torch.utils.data import DataLoader, random_split
import albumentations as A
model_state_dict = torch.load('results/model_segmentation_last_epoch.pt'
                              ,map_location=torch.device('cpu')
                              )
print(model_state_dict.keys())
model = Unet_variant.UnetVariant(1, 1)
model.load_state_dict(model_state_dict)

model.eval()
folder = 'test'
img_dir = 'utils/DRIVE/'+folder+'/images/img/'
label_dir = 'utils/DRIVE/'+folder+'/images/label/'
mask_dir = 'utils/DRIVE/'+folder+'/mask/'
val_percent = 0.2
batch_size = 1
width_out = 420
height_out = 420
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available..... training on CPU')
else:
    print("CUDA is available..... training on GPU")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#model.to(device)
transform = A.Compose([
            # A.VerticalFlip(p=0.5),
            # A.RandomRotate90(p=0.5),
            # A.OneOf([A.ElasticTransform(p=0.5, alpha=90, sigma=120 * 0.05, alpha_affine=45 * 0.03),
            # A.GridDistortion(p=0.5)], p=0.6),
            # A.CLAHE(p=0.5),
            # A.RandomBrightnessContrast(p=0.6),
            # A.RandomGamma(p=0.6),
            #A.Normalize(mean=0.485, std=0.229)
        ])

transform_label = transforms.Compose([transforms.ToTensor()
                                      ])

dataset = CustomDataLoader(img_dir, label_dir, mask_dir,
                           transform, transform_label, image_scale= 0.5)
n_val = int(len(dataset) * val_percent)
n_train = int(len(dataset) - n_val)
train, val = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train, batch_size=2, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader = DataLoader(val, batch_size=2, shuffle=False,
                        num_workers=0, pin_memory=True)

# for img, lblin val_loader:
for images, labels in (val_loader):
    # if train_on_gpu:
    #     images = images.to(device)
    #     labels = labels.to(device)
    #img = img.detach().numpy()
    prediction = model(images)
    # if train_on_gpu:
    #     prediction = prediction.to(device)
    prediction = prediction.cpu().detach().numpy()
    n, c, h, w = (prediction.shape)


    for i in range(n):
        im = np.squeeze(prediction[i, :, :, :])
        im = (np.array(im) * 255).astype(np.uint8)
        ret, im = cv2.threshold(im, 60, 255, cv2.THRESH_TOZERO)
        # im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        #                      cv2.THRESH_BINARY, 11, 2)
        la = np.squeeze(labels[i, :, :, :])
        visualize(im, la)


# logits = logits.detach().numpy()


# print(prediction.shape)
# prediction = prediction[0][0]
# lbl = lbl[0][0]
# # prediction = (np.array(prediction) * 255).astype(np.uint8)
# # ret, prediction = cv2.threshold(prediction, 90, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# plt.imshow(prediction, cmap= 'gray')
#
# plt.show()
# plt.imshow(lbl, cmap='gray')
# plt.show()
# plt.imsave('results/pred.png', prediction, cmap='gray')

arr = np.load('results/train_loss_array.npy')
arr1 = np.load('results/stochastic_loss.npy')
arr2 = np.load('results/validation_loss_array.npy')
plt.plot(arr, label='training loss')

# plt.plot(arr1)
plt.plot(arr2, label=' val loss')
plt.legend()
plt.show()
