import torch
import cv2
from torchvision import transforms
from utils.Data_loader import CustomDataLoader
from matplotlib import pyplot as plt
from Network import Unet_variant
import numpy as np
from torch.utils.data import DataLoader, random_split
model_state_dict = torch.load('results/model_segmentation_last_epoch.pt',
                              map_location=torch.device('cpu'))
print(model_state_dict.keys())
model = Unet_variant.UnetVariant(1, 1)
model.load_state_dict(model_state_dict)
model.eval()
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
transform = transforms.Compose([transforms.ToTensor()])

transform_label = transforms.Compose([transforms.ToTensor()])

dataset = CustomDataLoader(img_dir, label_dir, mask_dir,
                           transform, transform_label)
n_val = int(len(dataset) * val_percent)
n_train = int(len(dataset) - n_val)
train, val = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train, batch_size=2, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader = DataLoader(val, batch_size=2, shuffle=False,
                        num_workers=0, pin_memory=True)

# for img, lblin val_loader:
img, lbl = next(iter(val_loader))
prediction = model(img)
prediction = prediction.detach().numpy()

# logits = logits.detach().numpy()


print(prediction.shape)
prediction = prediction[0][0]
lbl = lbl[0][0]
prediction = (np.array(prediction) * 255).astype(np.uint8)
ret, prediction = cv2.threshold(prediction, 90, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(prediction, cmap= 'gray')

plt.show()
plt.imshow(lbl, cmap='gray')
plt.show()
plt.imsave('results/pred.png', prediction, cmap='gray')

arr = np.load('results/train_loss_array.npy')
arr1 = np.load('results/stochastic_loss.npy')
arr2 = np.load('results/validation_loss_array.npy')
plt.plot(arr, label='training loss')

# plt.plot(arr1)
plt.plot(arr2, label=' val loss')
plt.legend()
plt.show()
