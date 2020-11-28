# Artifact Regularization based on Fourier Transform for Fine-Tuning GANs
## Idea
## Background
### In-Domain GAN Inversion for Real Image Editing

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![TensorFlow 1.12.2](https://img.shields.io/badge/tensorflow-1.12.2-green.svg?style=plastic)
![Keras 2.2.4](https://img.shields.io/badge/keras-2.2.4-green.svg?style=plastic)

![image](idinvert/docs/assets/teaser.jpg)

**Figure:** *Real image editing using the proposed In-Domain GAN inversion with a fixed GAN generator.*

> **In-Domain GAN Inversion for Real Image Editing** <br>
> Jiapeng Zhu*, Yujun Shen*, Deli Zhao, Bolei Zhou <br>
> *European Conference on Computer Vision (ECCV) 2020*

In the repository, we propose an in-domain GAN inversion method, which not only faithfully reconstructs the input image but also ensures the inverted code to be **semantically meaningful** for editing. Basically, the in-domain GAN inversion contains two steps:

1. Training **domain-guided** encoder.
2. Performing **domain-regularized** optimization.

**NEWS: Please also find [this repo](https://github.com/genforce/idinvert_pytorch), which is friendly to PyTorch users!**

[[Paper](https://arxiv.org/pdf/2004.00049.pdf)]
[[Project Page](https://genforce.github.io/idinvert/)]
[[Demo](https://www.youtube.com/watch?v=3v6NHrhuyFY)]
[[Colab](https://colab.research.google.com/github/genforce/idinvert_pytorch/blob/master/docs/Idinvert.ipynb)]

#### Testing

##### Pre-trained Models

Please download the pre-trained models from the following links. For each model, it contains the GAN generator and discriminator, as well as the proposed **domain-guided encoder**.

| Path | Description
| :--- | :----------
|[face_256x256](https://drive.google.com/file/d/1azAzSZg6VfNydjWr4qfl8Z4LfxktTPqM/view?usp=sharing)    | In-domain GAN trained with [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset.
|[tower_256x256](https://drive.google.com/file/d/1USfaSLor5d71IRoC8CWTbKJagS0-MJEv/view?usp=sharing)   | In-domain GAN trained with [LSUN Tower](https://github.com/fyu/lsun) dataset.
|[bedroom_256x256](https://drive.google.com/file/d/1nRa4WAE1qF_j1CtH32hxjREK0o-rpucD/view?usp=sharing) | In-domain GAN trained with [LSUN Bedroom](https://github.com/fyu/lsun) dataset.

#### Inversion

```bash
MODEL_PATH='styleganinv_face_256.pkl'
IMAGE_LIST='examples/test.list'
python invert.py $MODEL_PATH $IMAGE_LIST
```

NOTE: We find that 100 iterations are good enough for inverting an image, which takes about 8s (on P40). But users can always use more iterations (much slower) for a more precise reconstruction.

##### Semantic Diffusion

```bash
MODEL_PATH='styleganinv_face_256.pkl'
TARGET_LIST='examples/target.list'
CONTEXT_LIST='examples/context.list'
python diffuse.py $MODEL_PATH $TARGET_LIST $CONTEXT_LIST
```

NOTE: The diffusion process is highly similar to image inversion. The main difference is that only the target patch is used to compute loss for **masked** optimization.

##### Interpolation

```bash
SRC_DIR='results/inversion/test'
DST_DIR='results/inversion/test'
python interpolate.py $MODEL_PATH $SRC_DIR $DST_DIR
```

##### Manipulation

```bash
IMAGE_DIR='results/inversion/test'
BOUNDARY='boundaries/expression.npy'
python manipulate.py $MODEL_PATH $IMAGE_DIR $BOUNDARY
```

NOTE: Boundaries are obtained using [InterFaceGAN](https://github.com/genforce/interfacegan).

##### Style Mixing

```bash
STYLE_DIR='results/inversion/test'
CONTENT_DIR='results/inversion/test'
python mix_style.py $MODEL_PATH $STYLE_DIR $CONTENT_DIR
```

#### Training

The GAN model used in this work is [StyleGAN](https://github.com/NVlabs/stylegan). Beyond the original repository, we make following changes:

- Change repleated $w$ for all layers to different $w$s (Line 428-435 in file `training/networks_stylegan.py`).
- Add the *domain-guided* encoder in file `training/networks_encoder.py`.
- Add losses for training the *domain-guided* encoder in file `training/loss_encoder.py`.
- Add schedule for training the *domain-guided* encoder in file `training/training_loop_encoder.py`.
- Add a perceptual model (VGG16) for computing perceptual loss in file `perceptual_model.py`.
- Add training script for the *domain-guided* encoder in file `train_encoder.py`.

##### Step-1: Train your own generator

```bash
python train.py
```

##### Step-2: Train your own encoder

```bash
TRAINING_DATA=PATH_TO_TRAINING_DATA
TESTING_DATA=PATH_TO_TESTING_DATA
DECODER_PKL=PATH_TO_GENERATOR
python train_encoder.py $TRAINING_DATA $TESTING_DATA $DECODER_PKL
```

Note that the file `dataset_tool.py`, which is borrowed from the [StyleGAN](https://github.com/NVlabs/stylegan) repo, is used to prepared a directory of data from all resolutions. The training of the encoder does not rely on the progressive strategy, therefore, the training data and the test data should be both specified as the `.tfrecords` file with the highest resolution.

### Detecting CNN-Generated Images [[Project Page]](https://peterwang512.github.io/CNNDetection/)

**CNN-generated images are surprisingly easy to spot...for now**  
[Sheng-Yu Wang](https://peterwang512.github.io/), [Oliver Wang](http://www.oliverwang.info/), [Richard Zhang](https://richzhang.github.io/), [Andrew Owens](http://andrewowens.com/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/).
<br>In [CVPR](https://arxiv.org/abs/1912.11035), 2020.

<img src='https://peterwang512.github.io/CNNDetection/images/teaser.png' width=1200>

This repository contains models, evaluation code, and training code on datasets from our paper. **If you would like to run our pretrained model on your image/dataset see [(2) Quick start](https://github.com/PeterWang512/CNNDetection#2-quick-start).**

**Jun 20 Update** Training code and dataset released; test results on uncropped images added (recommended for best performance).

**Oct 26 Update** Some reported the download link for training data does not work. If this happens, please try the updated alternative links: [1](https://drive.google.com/drive/u/2/folders/14E_R19lqIE9JgotGz09fLPQ4NVqlYbVc) and [2](https://cmu.app.box.com/folder/124997172518?s=4syr4womrggfin0tsfhxohaec5dh6n48)

#### (1) Setup

##### Install packages
- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

#### Download model weights
- Run `bash weights/download_weights.sh`


### (2) Quick start

#### Run on a single image

This command runs the model on a single image, and outputs the uncalibrated prediction.

```
# Model weights need to be downloaded.
python demo.py -f examples/real.png -m weights/blur_jpg_prob0.5.pth
python demo.py -f examples/fake.png -m weights/blur_jpg_prob0.5.pth
```

#### Run on a dataset

This command computes AP and accuracy on a dataset. See the [provided directory](cnn_detection/examples/realfakedir) for an example. Put your real/fake images into the appropriate subfolders to test.

```
python demo_dir.py -d examples/realfakedir -m weights/blur_jpg_prob0.5.pth
```

#### (3) Dataset

#### Testset
The testset evaluated in the paper can be downloaded [here](https://drive.google.com/file/d/1z_fD3UKgWQyOTZIBbYSaQ-hz4AzUrLC1/view?usp=sharing).

The zip file contains images from 13 CNN-based synthesis algorithms, including the 12 testsets from the paper and images downloaded from whichfaceisreal.com. Images from each algorithm are stored in a separate folder. In each category, real images are in the `0_real` folder, and synthetic images are in the `1_fake` folder. 

Note: ProGAN, StyleGAN, StyleGAN2, CycleGAN testset contains multiple classes, which are stored in separate subdirectories.

#### Training set
The training set used in the paper can be downloaded [here](https://drive.google.com/file/d/1iVNBV0glknyTYGA9bCxT_d0CVTOgGcKh/view?usp=sharing) (Try alternative links [1](https://drive.google.com/drive/u/2/folders/14E_R19lqIE9JgotGz09fLPQ4NVqlYbVc),[2](https://cmu.app.box.com/folder/124997172518?s=4syr4womrggfin0tsfhxohaec5dh6n48) if the previous link does not work). All images are from LSUN or generated by ProGAN, and they are separated in 20 object categories. Similarly, in each category, real images are in the `0_real` folder, and synthetic images are in the `1_fake` folder.

#### Validation set
The validation set consists of held-out ProGAN real and fake images, and can be downloaded [here](https://drive.google.com/file/d/1FU7xF8Wl_F8b0tgL0529qg2nZ_RpdVNL/view?usp=sharing). The directory structure is identical to that of the training set.

#### Download the dataset
A script for downloading the dataset is as follows: 
```
# Download the testset
cd dataset/test
bash download_testset.sh
cd ../..

# Download the training set
cd dataset/train
bash download_trainset.sh
cd ../..

# Download the validation set
cd dataset/val
bash download_valset.sh
cd ../..
```

**If the script doesn't work, an alternative will be to download the zip files manually from the above google drive links. One can place the testset, training, and validation set zip files in `dataset/test`, `dataset/train`, and `dataset/val` folders, respectively, and then unzip the zip files to set everything up.**

### (4) Train your models
We provide two example scripts to train our `Blur+JPEG(0.5)` and `Blur+JPEG(0.1)` models. We use `checkpoints/[model_name]/model_epoch_best.pth` as our final model.
```
# Train Blur+JPEG(0.5)
python train.py --name blur_jpg_prob0.5 --blur_prob 0.5 --blur_sig 0.0,3.0 --jpg_prob 0.5 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot ./dataset/ --classes airplane,bird,bicycle,boat,bottle,bus,car,cat,cow,chair,diningtable,dog,person,pottedplant,motorbike,tvmonitor,train,sheep,sofa,horse

# Train Blur+JPEG(0.1)
python train.py --name blur_jpg_prob0.1 --blur_prob 0.1 --blur_sig 0.0,3.0 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot ./dataset/ --classes airplane,bird,bicycle,boat,bottle,bus,car,cat,cow,chair,diningtable,dog,person,pottedplant,motorbike,tvmonitor,train,sheep,sofa,horse
```

### (5) Evaluation

After the testset and the model weights are downloaded, one can evaluate the models by running:

```
# Run evaluation script. Model weights need to be downloaded. See eval_config.py for flags
python eval.py
```

Besides print-outs, the results will also be stored in a csv file in the `results` folder. Configurations such as the path of the dataset, model weight are in `eval_config.py`, and one can modify the evaluation by changing the configurations.


**6/13/2020 Update** Additionally, we tested on uncropped images, and observed better performances on most categories. To evaluate without center-cropping:
```
# Run evaluation script without cropping. Model weights need to be downloaded.
python eval.py --no_crop --batch_size 1
```

The following are the models' performances on the released set, with cropping to 224x224 (as in the paper), and without cropping.

<b>[Blur+JPEG(0.5)]</b>

|Testset   |Acc (224)|  AP (224)  |Acc (No crop)|  AP  (No crop)|
|:--------:|:------:|:----:|:------:|:----:|
|ProGAN    | 100.0%	|100.0%|  100.0%	|100.0%|
|StyleGAN  | 73.4%	|98.5% |  77.5%	|99.3% |
|BigGAN    | 59.0%	|88.2% |  59.5%	|90.4% |
|CycleGAN  | 80.8%	|96.8% |  84.6%	|97.9% |
|StarGAN   | 81.0%	|95.4% |  84.7%	|97.5% |
|GauGAN    | 79.3%	|98.1% |  82.9%	|98.8% |
|CRN       | 87.6%	|98.9% |  97.8%	|100.0% |
|IMLE      | 94.1%	|99.5% |  98.8%	|100.0% |
|SITD      | 78.3%	|92.7% |  93.9%	|99.6% |
|SAN       | 50.0%	|63.9% |  50.0%	|62.8% |
|Deepfake  | 51.1%	|66.3% |  50.4%	|63.1% |
|StyleGAN2 | 68.4%	|98.0% |  72.4%	|99.1% |
|Whichfaceisreal| 63.9%	|88.8% |  75.2%	|100.0% |


<b>[Blur+JPEG(0.1)]</b>

|Testset   |Acc (224)|  AP (224)  |Acc (No crop)|  AP  (No crop)|
|:--------:|:------:|:----:|:------:|:----:|
|ProGAN    |100.0%	|100.0%| 100.0%	|100.0%|
|StyleGAN  |87.1%	|99.6%| 90.2%	|99.8% |
|BigGAN    |70.2%	|84.5%| 71.2%	|86.0% |
|CycleGAN  |85.2%	|93.5%| 87.6%	|94.9% |
|StarGAN   |91.7%	|98.2%| 94.6%	|99.0% |
|GauGAN    |78.9%	|89.5%| 81.4%	|90.8% |
|CRN       |86.3%	|98.2%| 86.3%	|99.8% |
|IMLE      |86.2%	|98.4%| 86.3%	|99.8% |
|SITD      |90.3%	|97.2%| 98.1%	|99.8% |
|SAN       |50.5%	|70.5%| 50.0%	|68.6% |
|Deepfake  |53.5%	|89.0%| 50.7%	|84.5% |
|StyleGAN2 |84.4%	|99.1%| 86.9%	|99.5% |
|Whichfaceisreal|83.6%	|93.2%| 91.6%	|99.8%|

