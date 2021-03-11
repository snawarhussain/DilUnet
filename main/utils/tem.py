
from matplotlib import pyplot as plt




def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image, cmap='gray')
        ax[1].imshow(mask, cmap= 'gray')
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

# if __name__ == '__main__':
#     img_dir = 'DRIVE/training/images/img/'
#     label_dir = 'DRIVE/training/images/label/'
#     mask_dir = 'DRIVE/training/mask/'
#     val_percent = 0.2
#     batch_size = 1
#     width_out = 420
#     height_out = 420
#     train_on_gpu = torch.cuda.is_available()
#     if not train_on_gpu:
#         print('CUDA is not available..... training on CPU')
#     else:
#         print("CUDA is available..... training on GPU")
#     device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#     transform = transforms.Compose([
#         transforms.ToTensor()])
#     #
#     # transforms.RandomRotation(45),
#     # transforms.RandomHorizontalFlip(),
#     transform_label = transforms.Compose([
#                                           transforms.ToTensor()
#                                           ])
#
#     dataset = CustomDataLoader(img_dir, label_dir, mask_dir, transform, transform_label)
#     n_val = int(len(dataset) * val_percent)
#     n_train = int(len(dataset) - n_val)
#     train, val = random_split(dataset, [n_train, n_val])
#     train_loader = DataLoader(train, batch_size=10, shuffle=True, num_workers=0, pin_memory=True)
#     val_loader = DataLoader(val, batch_size=3, shuffle=False, num_workers=0, pin_memory=True)
#     for img, label in (iter(train_loader)):
#         img = np.squeeze(img[0].permute(1, 2, 0).numpy())
#         label = np.squeeze(label[0].permute(1, 2, 0).numpy())
#
#         visualize(img,label)