# segmentationUnet
DiluNET: Unet variant for segmentation

### Background and Objective:
 Retinal image segmentation can help clinicians detect pathological disorders by studying the change in retinal blood vessels.
 This early detection can help prevent blindness and many of the other vision impairments.
 Although several supervised and unsupervised methods have been proposed for the task of blood vessels segmentation; however, there is still much room for improvement. 

### Method:
 We proposed an automatic retinal blood vessels segmentation method based on the UNET architecture.
 This end-to-end framework utilizes pre-processing and data augmentation pipeline for training. 
 In this architecture, we proposed the use of multi-scale input and multi-output modules with improved skip paths and the correct use of dilated convolutions for effective feature extraction. 
 In multi-scale input, the input image is scaled down and concatenated with the output of convolutional blocks at different points in the encoder path to ensure the feature transfer of the original image while the multi-output module, output from each decoder block is up-sampled combined to get the final output.
 Skip paths connect each encoder block with the corresponding decoder block and the whole architecture utilizes different dilation rates for improvising the overall feature extraction  
## Results:
 The proposed method achieved results with **accuracies: 0.9680, 0.9694, 0.9701; sensitivities: 0.8837, 0.8263, 0.8713 and IOUs: 0.8698, 0.7951, 0.8184 on three of the publicly available datasets of DIVE, STARE and CHASE respectively**.
 An ablation study is performed to show the contribution of each proposed module and technique.
 ![DilUnet vs Unet](https://github.com/snawarhussain/segmentationUnet/blob/master/Screenshot%202021-09-25%20145758.png)
 
 
## Conclusion:
 The evaluation metrics revealed that the performance of the proposed method is higher than the original UNET and other UNET based architectures as well as many other state-of-the-art segmentation techniques.
 ![DilUnet](https://github.com/snawarhussain/segmentationUnet/blob/master/Screenshot%202021-09-25%20145832.png)
