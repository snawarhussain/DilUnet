import torch
import cv2
from U_net import  UNet
from torchvision import transforms
from utils.Data_loader import CustomDataLoader
from matplotlib import pyplot as plt
model_state_dict = torch.load('model_segmentation_last_epoch.pt',
                              map_location=torch.device('cpu'))
print(model_state_dict.keys())
import numpy as np

import Unet_variant
model = Unet_variant.UnetVariant(1, 1)

model.load_state_dict(model_state_dict)

import os
import numpy as np

from torch.utils.data import  DataLoader, random_split
img_dir = 'utils/DRIVE/training/images/img/'
label_dir = 'utils/DRIVE/training/images/label/'
mask_dir = 'utils/DRIVE/training/mask/'
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
transform = transforms.Compose([
    transforms.ToTensor()])

transform_label = transforms.Compose([

    transforms.ToTensor()])

dataset = CustomDataLoader(img_dir, label_dir, mask_dir,
                           transform, transform_label)
n_val = int(len(dataset) * val_percent)
n_train = int(len(dataset) - n_val)
train, val = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train, batch_size=2, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader = DataLoader(val, batch_size=2, shuffle=False,
                        num_workers=0, pin_memory=True)



#for img, lbl  in val_loader:
img, lbl  = next(iter(val_loader))
prediction = model(img)
prediction = prediction.detach().numpy()

# logits = logits.detach().numpy()


print(prediction.shape)
prediction = prediction[0][0]
lbl = lbl[0][0]
prediction =  (np.array(prediction)*255).astype(np.uint8)
ret, prediction = cv2.threshold(prediction, 20, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(prediction, cmap='gray')

plt.show()
plt.imshow(lbl,cmap='gray')
plt.show()
plt.imsave('pred.png', prediction, cmap = 'gray')
# label = label.detach().numpy()
# label = label[0][0]
arr =  np.load('trainlossarray.npy')
arr1 = np.load('stochastic_loss.npy')
arr2 = np.load('validation_loss_array.npy')
plt.plot(arr, label='training loss')

#plt.plot(arr1)
plt.plot(arr2, label=' val loss')
plt.legend()
plt.show()
