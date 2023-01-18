# Dancing In the Dark

PyTorch implementation of a modified-UNet for ultra low light edge extraction inferring, from the following poster for Learning From Data 2022 Fall, Tsinghua University.

Authors:
Chengwei Ren, Jinnan He and Yuzhu Zhang  
Math Imaging Lab, Tsinghua-Berkeley Shenzhen Institute, Tsinghua University

![DID](figures/Group_10_Dancing_in_the_dark.png)

## Catalog
- [x] INSTALLATION  
- [x] Image Inference Usage 
- [x] Video Inference Usage  
- [x] Web Video Demo

## INSTALLATION
We provide installation instructions for edge detection experiments here.
### Dependency setup
Install FFMPEG.  
* Please refer to [FFMPEG website](https://ffmpeg.org/download.html) to get installation guide.
* This repository is based on FFMPEG-3.4.11
### Clone this repository
~~~
git clone https://github.com/ChernweiRen/Dancing_In_the_Dark.git
cd Dancing_In_the_Dark
~~~

### Environment setup
Create an new conda virtual environment
~~~
conda create -n UNeXt python=3.6
conda activate UNeXt
~~~
Install required packages
~~~
pip install -r requirements.txt
~~~
Download the checkpoint from [GoogleDrive](https://drive.google.com/file/d/1ywQPIb91qurzWND07Z2fCQnutL8edcZb/view?usp=share_link), and put it in the `checkpoint` directory.

## Image Inference Usage
run the following shell script
~~~
python inference.py \
    --arch_type Sony \
    --input RAW_imgs/00184_04_0.04s.ARW \
    --resume checkpoint/model_best.pth.tar \
    --result_dir ./
    --gpu 0
~~~
~~~
python inference.py \
    --arch_type Sony \
    --input RAW_imgs/00065_09_0.1s.ARW \
    --resume checkpoint/model_best.pth.tar \
    --result_dir ./
    --gpu 0
~~~
Set `--input` to the input low-light RAW image path.  
Set `--result_dir` to the output edge map path.  
Set `--resume` to the model checkpoint path.
Set `--gpu` to specify the GPU for image inferring.

**Tips!!!**  
The input image must be a RAW image! If you only have RGB images, you should convert RGB images to RAW images through the following two works.  
[CycleISP](https://github.com/swz30/CycleISP) &
[InvertibleISP](https://github.com/yzxing87/Invertible-ISP)

## Video Inference Usage
Specify the configurations in `video_infer.sh`.  
**name** is the prefix the input video.  
**video_suffix** is the suffix of the input video.  
**suffix** should not be modified.  
**path** refers to the absolute path of the video parent directory.  
**H & W** is the Height and Width of the video.  
**ckpt** is the path to model checkpoint.  
then run this.
~~~
bash video_infer.sh
~~~
The output edge result video will be stored in the folder where this video_infer.sh script is located.

## Web Video Demo
[Video Demo](https://www.aliyundrive.com/s/fqscmgxuUEn)

![0.4LUX](https://media.giphy.com/media/f3hdKq8oV2pqo82ZG8/giphy.gif)  

![2](https://media.giphy.com/media/mi5G2hxFUj9C5IfgMc/giphy.gif)