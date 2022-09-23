# Ice Segmentation using Deep Learning
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Naereen/badges)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)<br>
This repository holds the various findings and experimentations that went into creating a preliminary Machine Learning model for Ice Segmentation using Deep Learning methods for the regions affected by this situation in the Inuit-Nunavik Region.

With the help of the CAIMAN network we had over 500 images at our disposal initially to work with and develop a model that would work similar, if not better, than pre-existing models for the same task found during the [literature review](https://docs.google.com/spreadsheets/d/1KfXr8ZhjXgLT4AE_IYlHgQrWP2ulJYbEjijWrnaoA2o/edit?usp=sharing).

Let us go through the key steps in creating the dataset to work upon and the experiments supporting them.

## Initial Experiments on Existing Models
For the initial part of the project, we looked upon recreating results from pre-exisitng models to gauge their own performance on their own dataset. The recreation was done using the publically available repositories.

![](https://lh5.googleusercontent.com/SIypAdQv6Cy6illKANGbTnnNiXEJh4lFN9R_qPGyrCkIkQFTtoFqoBY9rJjT39uvp5T3GiCa76n_OZbEaiLgxAMvTMmovkqG8BW0EloZvHPlj3TAdmTSqhQ-bKMxu4gv1zWk-ikaWz9YigDq4OH4pFjH4PyZTOfyXm1NWKtZCeZ3cfV5kWrp2fVLQg)

The key paper that caught our eye was the [Supplementing Remote Sensing of Ice: Deep Learning-Based Image Segmentation System for Automatic Detection and Localization of Sea-ice Formations From Close-Range Optical Images](https://ieeexplore.ieee.org/document/9443178) which used an ensemble model combining architectures like PSPNet, PSPDenseNet, DeepLabV3+, and UPerNet, to get its result on 450+ highly annotated images.

![](https://lh4.googleusercontent.com/6WK7gU8tKNY7IH9QXobLiRHuls6Q8y86lYhKz2VoQw4D0ocCDSANJpEc1KTc4KhV93ySZk0Bk2I3UzAfH9oHQeuLmMJxY4eupgAyaosc8qBlvvqCdoo0d0GH20PxWS0d6ewexKmVk3AS3zH4WTvWXyPLhSxNshNs4BJqFVeKOUZmkKLr-ltiQhceWA)

Using the similar premise, we decided to try to work on an ensemble of pre-known somewhat lightweighed models like PSPNet, U-Net and Mask R-CNN.

## Dealing with the Data

However, there was an abundance of data coming from the CAIMAN network, the job of manually annotating the images was a bit tricky given the weather conditions the images were taken in.

![](https://lh6.googleusercontent.com/LEDvyaAr2K6L2mVOd2EyCYn9kP7p7AR7NZcbtM1T5quWSHIkAeDGjhHZvtJ_FLupNrJR3Onf04Q9FmHUmEP1ba4YUGXfgxk2Kspiv0889iLk8tiiaQVfFQZQ71CZVH1z3ujuNlGlIBDVfVfSrgfdBoJU4zLTVx4d7Co9LTcOcHLwjsVJfhd3OqCCJQ)

I took upon annotating over a 100 images manually using the [Pixel Annotation Tool](https://github.com/abreheret/PixelAnnotationTool) which is easily buildable on both LINUX and Windows.

![](https://lh3.googleusercontent.com/DMPo51dphjkVyFV41HnzYvmWUDlWv1hoN34xBCJuUKZa15LRrmfdUFUyp4Y5YxT1mbp3VHgal7zXDWi-ERPVgTncjtbLl_R3xrHmhcBiXawFFm8C8UAJv9RGAKGkqOg__1iyuHyYNZ6bOF1cLeHyZN2PcZCOKe_Ad6q1jWemvLZ9OP2tG9S9Lsyo_Q)
<br>
![](https://lh5.googleusercontent.com/Pj8I2u_tmMKQ2QRPd_xYuLUhLbDcihH75ATpMLYCQd8dbA_H0aHVAJvcayL9TVvQjb7wIwbQFTo3KnGqT8yb88yRUnqQawKd8QpeTALOHVKKlSQ4MzI-IyHpua63ft9SCBlM-_YP2kNQjgtoaEoV0sOnh6HZca1LE5sELTooySLBrYcm7E1vcYS3mQ)

The annotation technique uses Watershedding feature from OpenCV. Instead of the traditional Thresholding or Contour detection, watershedding performs a marker-based image segmentation on its own depending on neighbor pixels.

Once the images were annotated, the images needed to be renamed and cropped due to their origins from different locations and types of cameras. (code to which can be found in the [Utils Folder](utils)).

## Training the Models

Before training the individual models on the desired dataset, I decided to go for a transfer learning method wherein, I aimed to get the weights from training the models on bigger datasets like ImageNet and the Caravan Dataset and then use them for Ice Segmentation on the Custom Dataset.

### U-Net
U-Net is a CNN architecture designed specifically to deal with image segmentation in Bio Medical Images. It can be used to highlight the points of interest all the while semantically segmenting each pixel to the desired class.

<img src= "https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png">

It usually is a pre-trained classification network like VGG/ResNet where you apply convolution blocks followed by a maxpool downsampling to encode the input image into feature representations at multiple different levels. The decoder is the second half of the architecture.
### PSP Net
PSPNet, or Pyramid Scene Parsing Network, is a semantic segmentation model that utilises a pyramid parsing module that exploits global context information by different-region based context aggregation. The local and global clues together make the final prediction more reliable.

<img src= "https://production-media.paperswithcode.com/methods/new_pspnet-eps-converted-to.jpg">

Given an input image, PSPNet use a pretrained CNN with the dilated network strategy to extract the feature map.
## Steps to run the Repository
Download the repository using the command: <br>
`git clone https://github.com/aryankargwal/ice_segmentation.git`

Download the Requirements to run the code using the command: <br>
`pip install -r requirements.txt`
## TODO
- [x] Literature Review
- [x] Experiment List
- [x] Experimentations
- [x] Data Collection
- [x] Data Annotation
- [x] Data Renaming
- [x] Data Cropping
- [x] Unet Training
- [x] PSPNet Training
- [ ] Mask RCNN Training
- [ ] Ensemble Model Training

## License
This project is under the Apache License. See [LICENSE](LICENSE) for more information.