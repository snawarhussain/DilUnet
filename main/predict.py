import torch
import cv2

from main.Network.Unet_variant_1 import UnetVariant_1
from main.evaluate import getIOU, get_roc_curve, roc_auc_score, getAcc
from main.utils.DataLoaderNoMask import CustomDataLoaderNoMask
from main.utils.tem import *
from torchvision import transforms
from utils.Data_loader import CustomDataLoader
from matplotlib import pyplot as plt
from Network import Unet_variant
import numpy as np
from torch.utils.data import DataLoader, random_split
import albumentations as A

from sklearn.metrics import jaccard_score as js
model_state_dict = torch.load('results/model_segmentation_last_epoch.pt'
                              , map_location=torch.device('cpu')
                              )
print(model_state_dict.keys())
model = UnetVariant_1(1, 1)
model.load_state_dict(model_state_dict)


folder = 'test'
img_dir = 'utils/DRIVE/'+folder+'/images/img/'
label_dir = 'utils/DRIVE/'+folder+'/images/2nd_manual/'
mask_dir = 'utils/DRIVE/'+folder+'/mask/'

# img_dir = 'utils/CHASEDB1/train/img/'
# label_dir = 'utils/CHASEDB1/train/1st_manual/'
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
             A.CLAHE(p=0),
            # A.RandomBrightnessContrast(p=0.6),
            # A.RandomGamma(p=0.6),
            #A.Normalize(mean=0.485, std=0.229)
        ])

transform_label = transforms.Compose([transforms.ToTensor()
                                      ])

dataset = CustomDataLoader(img_dir, label_dir, mask_dir,
                           transform, transform_label, image_scale=.5)
n_val = int(len(dataset) * val_percent)
n_train = int(len(dataset) - n_val)
train, val = random_split(dataset, [n_train, n_val])
data_loader = DataLoader(dataset, batch_size=1, shuffle=True,
                          num_workers=0, pin_memory=False)
val_loader = DataLoader(val, batch_size=2, shuffle=False,
                        num_workers=0, pin_memory=False)


# for img, lblin val_loader:
AUC=[]
whole_Acc=[]
whole_IOU=[]
whole_sensitivity=[]
whole_specificity=[]
Mes = []
Ori = []
with torch.no_grad():
    for images, labels in (data_loader):
        model.eval()
        # if train_on_gpu:
        #     images = images.to(device)
        #     labels = labels.to(device)
        #img = img.detach().numpy()
        prediction = model(images)
        # if train_on_gpu:
        #     prediction = prediction.to(device)
        prediction = prediction.cpu().detach().numpy()
        n, c, h, w = (prediction.shape)
        labels = np.array(labels.cpu().detach())

        for i in range(n):
            im = np.squeeze(prediction[i, :, :, :])
            im = (np.array(im) * 255).astype(np.uint8)
            #ret, im = cv2.threshold(im, 0, 255, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)


            ret, im_binarized = cv2.threshold(im, 45, 255, cv2.THRESH_BINARY)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 1))
            # im_binarized = cv2.morphologyEx(im_binarized, cv2.MORPH_ERODE, kernel)
            la = np.squeeze(labels[i, :, :, :])

            im = im/255.

            im_binarized = im_binarized/255
            visualize(im_binarized , la)
            auc = roc_auc_score(la.ravel(), im.ravel(), average='weighted')
            print('auc=', auc)
            # cv2.imwrite("pred/pred_{}.jpg".format(code), Predimage*255)
            # Predimage.save("pred/pred_{}.jpg".format(code))
            IOU = getIOU(im, la)
            Acc, sensitivity, specificity = getAcc(im_binarized, la)
            print('Acc=', Acc)
            print('sensitivity=', sensitivity)
            print('specificity=', specificity)
            la_flat = la.reshape(-1)
            im_flat = im_binarized.reshape(-1)
            #print('IoU =', js(la_flat,im_flat))

        Ori.append(la)
        Mes.append(im)
        AUC.append(auc)
        whole_IOU.append(IOU)
        whole_Acc.append(Acc)
        whole_sensitivity.append(sensitivity)
        whole_specificity.append(specificity)
    auc = roc_auc_score(np.array(Ori).ravel(), np.array(Mes).ravel(), average='weighted')
    get_roc_curve(np.array(Ori).ravel(), np.array(Mes).ravel(), average='macro')

    import scipy.io
    num = len(data_loader)
    # scipy.io.savemat('MesDrive9845.mat', {'target': Ori, 'Predimage': Mes})  # 写入mat文件
    mean_AUC = sum(AUC) / num
    mean_IOU = sum(whole_IOU) / num
    mean_Acc = sum(whole_Acc) / num
    mean_sensitivity = sum(whole_sensitivity) / num
    mean_specificity = sum(whole_specificity) / num
    print('mean_IOU', mean_IOU)
    print('mean_Acc', mean_Acc)
    print('mean_sensitivity', mean_sensitivity)
    print('mean_specificity', mean_specificity)
    print('mean_AUC', mean_AUC)



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
