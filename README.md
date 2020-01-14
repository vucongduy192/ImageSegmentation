### Overview
The programs in this repository train and use a fully convolutional neural network to take an image and classify 
its pixels. The network is transfer-trained basing on the VGG-16 model using the approach described in this paper 
by Jonathan Long et al. 

### Source kitti dataset + VGG base
Dataset + VGG url: https://drive.google.com/file/d/1je7YaKnb3dhEQ33tvCeOeqt6vy2M_Ubv/view?usp=sharing

Download and extract to root directory 

### Environment descriptions
* Tensorflow-gpu 1.13.1
* Cuda 10.0
* CuDNN 7.4.1

### Installation guide
Installation follow this guide https://gist.github.com/bogdan-kulynych/f64eb148eeef9696c70d485a76e42c3a
with some points of attention:
* Install specific cuda version
```bash
sudo apt install -y cuda-10-0    
```
* Search for available version of libcudnn7 package:
```bash 
sudo apt-cache policy libcudnn7
```
* Remember to install same specific version for libcudnn7 and libcudnn7-dev

* Check compatible version tensorflow-gpu

![GPU setup environment](./assets/GPUsetup.png)

### Reference: 
* https://github.com/ljanyst/image-segmentation-fcn
* https://nanonets.com/blog/how-to-do-semantic-segmentation-using-deep-learning/