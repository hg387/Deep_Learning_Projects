Final Project README:

Problem Statement: The applications of the image segmentation is
limitless. For this project, focus is towards understanding the
environment better especially the urban setting where we have a a lot of
movement that we don’t notice unless it is affecting us.\
It is very important to segmentize our urban environment to effectively
use in self-autonomous vehicles, Object tracking in a crowded space,
Precise calculations of various parameters such as number of trees,
number of small vs big buildings, number of cars parked etc. by just
using a video instead of a complex data set.

DataSet:

We are using a dataset called Cam Seq01 Dataset. It can be found at
http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamSeq01/ It consists
of high resolution images of our urban surrounding including images of
bicycles, pedestrians, cars, and their motions This dataset has 101
960X720 pixel images in which each pixel has been assigned to one of the
following 32 object classes

Architecture Used:

U-Net architecture is the extension of our traditional CNN model based
upon the ResNet34, 34 layers dimensions CNN downsizes our sample with
each layer U-Net upscales our sample after downsizing forming a
U-shaped. Downsizing process is called encoding while upscaling is
called decoding Objective Function: Cross Entropy Loss Activation
Function: Softmax activation function

More analysis is available in hg387Paper.pdf and CS 615\_ Deep Learning
Project.pptx. Sample video presentation is availlable at:
https://drive.google.com/file/d/19sXA\_bQalDLP5DkIDyXj9khD6\_ZBeiIQ/view?usp=sharing

1)  Source code is shared using Google Colab as thought this way is
    better instead of downloading all the libraries locally and
    attaching them. In case request not working, please use this link:
    https://colab.research.google.com/drive/1ysWLuYaOAT01ei6xLuLF831HLFSY4w0V?usp=sharing
    Locally the code is available as imagesegmentation.py

2)  Running the code is very easy as the environment is already set,
    please enable GPU if not enabled in your Google Colab.
3)  Made changes to the in-built resnet34 model inside of FastAI library
    so please don't unistall/re-install the FastAI library.


