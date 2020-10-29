# Development and Evaluation of Deep Learning for Screening Dental Caries from Oral Photos
![Ovreall Arthetecture](figure1.jpg)

This repository contains Pytorch implementation of a deep-learning-based detection model for localizing dental caries from oral photos. 
The multiple-GPU-based training/testing and CPU/single-GPU-based deployment are supported by this implementation. 


## Installation Guide
Our implementation only depends on `Pytorch` and `mmcv`. 

### Step 1: Clone the repository
```
$ git clone https://github.com/liangyuandg/DLCariesScreen.git
```

### Step 2: Install dependencies
**Note**: The implementation is built and tested under `Python3.6`.

**Note**: If you are using a Python virtualenv, make sure it is activated before running each command in this guide.

Install Pytorch by following the [official guidance](https://pytorch.org/). 

Install mmcv, which is a versatile image processing library designed for computer vision research, by following the [official guidance](https://github.com/open-mmlab/mmcv).

### Step 3: Download the pre-trained model weights
Download the model weights from [here](https://drive.google.com/file/d/1cuY783RCYNS0LwCTTGUlvMX7Z-XfKPNW/view?usp=sharing) and place it in the `checkpoints` directory.


## Instructions
The implementation supports multi-GPU-based training and testing for efficiency. The implementation also supports single-GPU-based or CPU-based inference for the deployment, e.g., on a model serveral or on edge. 

### Training
The implementation assumes a [COCO-formated](https://cocodataset.org/#format-data) json file for logging the raw image and annotation information. 

First, convert the json-formatted annotation file into pickle-formatted that supported by the training using `datasets/json_to_pickle.py`. Splitting the entries into training/validation/testing according to your experiment designs. 

Second, set the pathes for image repository, pickle-formatted annotation file, working directory, pre-trained model in `engines/engine_train_VGGSSD`. Other training parameters including augmentations and learning rates can also be modified in the header part of the file. 

Third, start trianing process by running the bash `engines/engine_train_multiple_GPU.sh`. 

### Testing and Evaluation

First, obtain the pickle-formatted annotation file similar as in the training procedure. 

Second, set the pathes for model checkpoint file, pickle-formatted annotation file, working directory, image repository in `engines/engine_infer_VGGSSD`. 

Third, start testing inference by running the bash `engines/engine_infer_multiple_GPU.sh`. 

For numerical evaluation, use the function provided in `evaluation/boxwise_relaxed_froc.py` for FROC; Use the function provided in `evaluation/imagewise_roc.py` for ROC. 

### Deployment

First, set the pathes for the testing image, model checkpoint file, and results in `deployment/engine_infer_Detection_single_GPU_draw_box.py`. 

Second, run `deployment/engine_infer_Detection_single_GPU_draw_box.py` for the single image inference and result visualization. 

We give one example input image and its inference result in the `deployment` folder.


## Notes
1. Since the privacy issue and commercial interest, the trained models and training images are not released. However, by following the instructions above, new models can be trained with individual's repo of data. 
2. More details about training strategy, model architecture descriptions, and result discussion will be released in a future publication. 
